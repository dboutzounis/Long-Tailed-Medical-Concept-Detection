#!/usr/bin/env python
"""Classification heads trained on frozen encoder embeddings (Section 4.4.5, Tables 5.9 and 5.11).

Heads:
    ridge         closed-form ridge regression, lambda selected by val MSE
    linear        Linear(d, C), zero bias
    linear_prior  Linear(d, C), bias initialized to the training log-odds
    cosine        tau * cos(x, w_c) + b_c with a learned temperature
    mlp           LayerNorm -> Linear(d, h) -> GELU -> Dropout(0.2) -> Linear(h, C)
Gradient heads use ASL, AdamW (lr 1e-4, weight decay 1e-4 on weight matrices), batch 512 and
early stopping on val loss (patience 5). A global threshold is selected on val and the test split
is evaluated once. concat3 / concat4 train on the concatenated per-encoder-normalized embeddings.

Usage:
    python train_frozen_heads.py --encoder medsiglip --head mlp --hidden 1024
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import random
import socket
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_DIRS = {
    "biomedclip": "biomedclip_features",
    "raddino": "raddino_features",
    "medsiglip": "medsiglip_features",
    "medimageinsight": "medimageinsight_features",
}
CONCAT_GROUPS = {
    "concat3": ["medimageinsight", "medsiglip", "biomedclip"],  # = KNN 3-way fusion
    "concat4": ["medimageinsight", "medsiglip", "biomedclip", "raddino"],
}
HEADS = ["ridge", "linear", "linear_prior", "cosine", "mlp"]
THRESHOLDS = [i / 100 for i in range(1, 100)]  # 0.01 .. 0.99, fixed grid


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def env_info(device):
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": str(device),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_num_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
    return info


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split(feat_dir, split, device):
    x_path = os.path.join(feat_dir, split + "_embs.pt")
    y_path = os.path.join(feat_dir, split + "_labels.pt")
    x = torch.load(x_path, map_location="cpu", weights_only=True).float()
    y = torch.load(y_path, map_location="cpu", weights_only=True).float()
    if x.dim() != 2:
        raise ValueError("%s: expected [N, D] embeddings, got %s" % (x_path, tuple(x.shape)))
    if y.dim() != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("%s: labels %s do not match embeddings %s"
                         % (split, tuple(y.shape), tuple(x.shape)))
    if not torch.all((y == 0) | (y == 1)):
        raise ValueError("%s: labels are not binary multi-hot" % y_path)
    x = F.normalize(x, dim=-1)  # idempotent if already normalised at extraction
    hashes = {"embs": sha256_file(x_path), "labels": sha256_file(y_path)}
    return x.to(device), y.to(device), hashes


def load_concat(feat_dirs, split, device):
    """One encoder: identical to load_split. Several: per-encoder L2-normalised
    blocks concatenated; rows must be the same images in the same order, which
    is verified by exact label-matrix equality (and test keys where saved)."""
    xs, hashes, y0 = [], {}, None
    keys0 = None
    for fd in feat_dirs:
        x, y, h = load_split(fd, split, device)
        name = os.path.basename(fd)
        if y0 is None:
            y0 = y
        elif not torch.equal(y, y0):
            raise ValueError("%s/%s: labels differ from %s -> rows are not aligned"
                             % (name, split, os.path.basename(feat_dirs[0])))
        kp = os.path.join(fd, split + "_keys.pkl")
        if os.path.exists(kp):
            with open(kp, "rb") as f:
                keys = list(pickle.load(f))
            if keys0 is None:
                keys0 = keys
            elif keys != keys0:
                raise ValueError("%s/%s: image keys differ in order" % (name, split))
        xs.append(x)
        hashes[name] = h
    if len(feat_dirs) == 1:
        return xs[0], y0, hashes[os.path.basename(feat_dirs[0])]
    return torch.cat(xs, dim=1), y0, hashes


def prior_logodds(y_train):
    n = y_train.shape[0]
    p = y_train.double().mean(0)
    n_zero = int((p == 0).sum())
    p = p.clamp(0.5 / n, 1 - 0.5 / n)
    return torch.log(p / (1 - p)).float(), n_zero


# ---------------------------------------------------------------------------
# Loss and metric
# ---------------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    """ASL (Ridnik et al. 2021) with sum over labels, mean over samples."""

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.02, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        p_neg = 1 - p
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)
        loss = y * torch.log(p.clamp(min=self.eps)) + (1 - y) * torch.log(p_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = p * y + p_neg * (1 - y)
            gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            loss = loss * torch.pow(1 - pt, gamma)
        return -loss.sum(dim=1).mean()


@torch.no_grad()
def sample_f1(pred, y):
    """Per-image F1 averaged over images; an image with empty gt and empty
    prediction scores 1 (never happens on ROCOv2, every image has a modality)."""
    tp = (pred & y).sum(1).double()
    denom = pred.sum(1).double() + y.sum(1).double()
    f1 = torch.where(denom > 0, 2 * tp / denom.clamp(min=1), torch.ones_like(denom))
    return f1.mean().item()


@torch.no_grad()
def tune_threshold(scores, y_bool):
    best_f1, best_t = -1.0, None
    for t in THRESHOLDS:  # ascending; ties keep the lowest threshold
        f1 = sample_f1(scores >= t, y_bool)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------
class LinearHead(nn.Module):
    def __init__(self, d, c, bias_init=None, weight_init=None):
        super().__init__()
        self.fc = nn.Linear(d, c)
        with torch.no_grad():
            if weight_init is not None:
                self.fc.weight.copy_(weight_init)
            if bias_init is not None:
                self.fc.bias.copy_(bias_init)
            else:
                self.fc.bias.zero_()

    def forward(self, x):
        return self.fc(x)


class CosineHead(nn.Module):
    def __init__(self, d, c, tau_init, bias_init):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c, d))
        nn.init.normal_(self.weight, std=d ** -0.5)  # only the direction matters
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.bias = nn.Parameter(bias_init.clone())

    def forward(self, x):
        cos = F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))
        return self.log_tau.exp() * cos + self.bias


class MLPHead(nn.Module):
    def __init__(self, d, c, hidden, dropout, bias_init):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, c),
        )
        with torch.no_grad():
            self.net[-1].bias.copy_(bias_init)

    def forward(self, x):
        return self.net(x)


def build_head(args, d, c, x_train, prior):
    zero = torch.zeros(c, device=x_train.device)
    if args.head == "linear":
        return LinearHead(d, c)
    if args.head == "linear_prior":
        return LinearHead(d, c, bias_init=prior)
    bias = prior if args.bias_init == "prior" else zero
    if args.head == "cosine":
        return CosineHead(d, c, args.tau_init, bias)
    if args.head == "mlp":
        return MLPHead(d, c, args.hidden, args.dropout, bias)
    raise ValueError(args.head)


def param_groups(model, weight_decay):
    """Weight decay on weight matrices only; biases, LayerNorm and log-tau are
    excluded so decay does not pull prior/temperature inits towards zero."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


# ---------------------------------------------------------------------------
# Training: gradient heads
# ---------------------------------------------------------------------------
def train_gradient_head(head, data, args, run_dir, device):
    x_tr, y_tr, x_va, y_va = data["x_tr"], data["y_tr"], data["x_va"], data["y_va"]
    y_va_bool = y_va.bool()
    crit = AsymmetricLoss(args.asl_gamma_neg, args.asl_gamma_pos, args.asl_clip)
    opt = torch.optim.AdamW(param_groups(head, args.weight_decay), lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)  # shuffle RNG only
    n = x_tr.shape[0]

    @torch.no_grad()
    def evaluate():
        head.eval()
        logits = head(x_va)
        loss = crit(logits, y_va).item()
        f1, t = tune_threshold(torch.sigmoid(logits), y_va_bool)
        return loss, f1, t

    log_path = os.path.join(run_dir, "log.csv")
    log_f = open(log_path, "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_f1_tuned", "val_thr", "seconds"])

    t0 = time.time()
    val_loss, val_f1, val_t = evaluate()
    writer.writerow([0, "", "%.6f" % val_loss, "%.6f" % val_f1, val_t, "%.1f" % 0.0])
    log_f.flush()
    print("epoch    0 | val loss %.4f | val F1 %.4f @ %.2f (init)" % (val_loss, val_f1, val_t))
    init_stats = {"val_loss": val_loss, "val_f1_tuned": val_f1, "val_thr": val_t}
    best_loss, best_epoch, bad = val_loss, 0, 0
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    epoch = 0
    for epoch in range(1, args.max_epochs + 1):
        head.train()
        perm = torch.randperm(n, generator=gen).to(device)
        running = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = crit(head(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            running += loss.item() * idx.numel()
        train_loss = running / n
        val_loss, val_f1, val_t = evaluate()
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        writer.writerow([epoch, "%.6f" % train_loss, "%.6f" % val_loss,
                         "%.6f" % val_f1, val_t, "%.1f" % (time.time() - t0)])
        log_f.flush()
        print("epoch %4d | train %.4f | val loss %.4f | val F1 %.4f @ %.2f%s"
              % (epoch, train_loss, val_loss, val_f1, val_t, "  *" if improved else ""))
        if bad >= args.patience:
            break
    log_f.close()

    hit_cap = (bad < args.patience) and epoch == args.max_epochs
    if hit_cap:
        print("WARNING: reached --max-epochs without triggering early stopping")
    head.load_state_dict(best_state)
    head.eval()
    torch.save(best_state, os.path.join(run_dir, "head.pt"))

    @torch.no_grad()
    def score(x):
        return torch.sigmoid(head(x))

    train_info = {"best_epoch": best_epoch, "epochs_run": epoch, "best_val_loss": best_loss,
                  "hit_max_epochs": hit_cap, "init": init_stats}
    if args.head == "cosine":
        train_info["tau_final"] = head.log_tau.exp().item()
    return score, train_info


# ---------------------------------------------------------------------------
# Training: ridge (closed form)
# ---------------------------------------------------------------------------
def fit_ridge(data, args, run_dir, device):
    x_tr, y_tr, x_va, y_va = data["x_tr"], data["y_tr"], data["x_va"], data["y_va"]
    X, Y = x_tr.double(), y_tr.double()
    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    gram = (Xc.T @ (Y - my))                      # X^T Y   [d, C]
    G = (Xc.T @ Xc).cpu()                         # X^T X   [d, d]
    evals, V = torch.linalg.eigh(G)               # CPU float64, deterministic
    evals, V = evals.to(device), V.to(device)
    VtXtY = V.T @ gram                            # [d, C]
    XvaV = (x_va.double() - mx) @ V               # [Nv, d]
    Yva = y_va.double()

    lambdas = [10 ** (k / 4) for k in range(-24, 17)]  # 1e-6 .. 1e4, 41 values
    rows, best = [], (float("inf"), None)
    for lam in lambdas:
        coef = VtXtY / (evals + lam).unsqueeze(1)
        mse = ((XvaV @ coef + my - Yva) ** 2).mean().item()
        rows.append((lam, mse))
        if mse < best[0]:
            best = (mse, lam)
    with open(os.path.join(run_dir, "log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "val_mse"])
        for lam, mse in rows:
            w.writerow(["%.6g" % lam, "%.8f" % mse])
    best_mse, best_lam = best
    at_edge = best_lam in (lambdas[0], lambdas[-1])
    print("ridge: best lambda %.4g (val MSE %.6f)%s"
          % (best_lam, best_mse, "  WARNING: at grid edge" if at_edge else ""))

    W = V @ (VtXtY / (evals + best_lam).unsqueeze(1))  # [d, C]
    b = my - mx @ W
    W, b = W.float(), b.float()
    torch.save({"W": W.cpu(), "b": b.cpu(), "lambda": best_lam}, os.path.join(run_dir, "head.pt"))

    @torch.no_grad()
    def score(x):
        return x @ W + b  # raw regression scores, thresholded on the same grid

    return score, {"best_lambda": best_lam, "best_val_loss": best_mse,
                   "loss_name": "val_mse", "lambda_at_grid_edge": at_edge}


def run_name(args):
    if args.head == "mlp":
        return "mlp_h%d_b%s" % (args.hidden, args.bias_init)
    if args.head == "cosine":
        return "cosine_b%s" % args.bias_init
    return args.head


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", required=True,
                   choices=sorted(ENCODER_DIRS) + sorted(CONCAT_GROUPS))
    p.add_argument("--head", required=True, choices=HEADS)
    p.add_argument("--features-root",
                   default="$WORK/Long-Tailed-Medical-Concept-Detection/artifacts")
    p.add_argument("--out-root", default=None,
                   help="default: <features-root>/frozen_heads")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm; 0 disables")
    p.add_argument("--asl-gamma-neg", type=float, default=4.0)
    p.add_argument("--asl-gamma-pos", type=float, default=0.0)
    p.add_argument("--asl-clip", type=float, default=0.02)
    p.add_argument("--hidden", type=int, default=1024, help="mlp hidden width")
    p.add_argument("--dropout", type=float, default=0.2, help="mlp dropout")
    p.add_argument("--bias-init", choices=["zero", "prior"], default="zero",
                   help="output-bias init for cosine and mlp heads")
    p.add_argument("--tau-init", type=float, default=None, help="cosine head temperature init")
    p.add_argument("--no-save-scores", action="store_true",
                   help="do not save test score matrix")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.head == "cosine" and args.tau_init is None:
        p.error("cosine requires --tau-init")
    args.features_root = os.path.expandvars(args.features_root)
    args.out_root = os.path.expandvars(args.out_root or
                                       os.path.join(args.features_root, "frozen_heads"))
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(args.seed)

    run_dir = os.path.join(args.out_root, args.encoder, run_name(args))
    if os.path.exists(os.path.join(run_dir, "results.json")) and not args.overwrite:
        print("results already exist in %s (use --overwrite); skipping" % run_dir)
        return
    os.makedirs(run_dir, exist_ok=True)
    print("run: %s / %s  ->  %s" % (args.encoder, run_name(args), run_dir))

    members = CONCAT_GROUPS.get(args.encoder, [args.encoder])
    feat_dir = [os.path.join(args.features_root, ENCODER_DIRS[m]) for m in members]
    x_tr, y_tr, h_tr = load_concat(feat_dir, "train", device)
    x_va, y_va, h_va = load_concat(feat_dir, "val", device)
    x_te, y_te, h_te = load_concat(feat_dir, "test", device)
    if len(members) > 1:
        print("concat of %s" % " + ".join(members))
    d, c = x_tr.shape[1], y_tr.shape[1]
    if not (x_va.shape[1] == x_te.shape[1] == d and y_va.shape[1] == y_te.shape[1] == c):
        raise ValueError("feature/label dimensions differ across splits")
    print("data: train %d, val %d, test %d | d=%d, C=%d"
          % (x_tr.shape[0], x_va.shape[0], x_te.shape[0], d, c))
    prior, n_zero = prior_logodds(y_tr)
    if n_zero:
        print("note: %d concepts have no positive training example" % n_zero)

    config = dict(vars(args))
    config["run_name"] = run_name(args)
    config["feature_dir"] = feat_dir
    config["feature_sha256"] = {"train": h_tr, "val": h_va, "test": h_te}
    config["env"] = env_info(device)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    data = {"x_tr": x_tr, "y_tr": y_tr, "x_va": x_va, "y_va": y_va,
            "x_te": x_te, "y_te": y_te}
    t0 = time.time()
    if args.head == "ridge":
        score, train_info = fit_ridge(data, args, run_dir, device)
    else:
        set_determinism(args.seed)  # head init independent of anything above
        head = build_head(args, d, c, x_tr, prior).to(device)
        score, train_info = train_gradient_head(head, data, args, run_dir, device)
        train_info["loss_name"] = "val_asl"

    s_va, s_te = score(x_va), score(x_te)
    val_f1, thr = tune_threshold(s_va, y_va.bool())
    test_f1 = sample_f1(s_te >= thr, y_te.bool())
    elapsed = time.time() - t0

    results = {
        "encoder": args.encoder,
        "run": run_name(args),
        "head": args.head,
        "val_f1": val_f1,
        "threshold": thr,
        "test_f1": test_f1,
        "mean_pred_per_image_test": (s_te >= thr).sum(1).double().mean().item(),
        "train_seconds": round(elapsed, 1),
        **train_info,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if not args.no_save_scores:
        torch.save(s_te.float().cpu(), os.path.join(run_dir, "test_scores.pt"))
    print("DONE %s / %s | val F1 %.4f @ %.2f | test F1 %.4f | %.0fs"
          % (args.encoder, run_name(args), val_f1, thr, test_f1, elapsed))


if __name__ == "__main__":
    main()
