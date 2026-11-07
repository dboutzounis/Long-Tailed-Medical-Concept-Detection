"""Zero-shot classification with and without label priors (Sections 4.4.2-4.4.3, Table 5.6).

Each concept is scored by the cosine between the image embedding and its four-template
prompt-ensemble text embedding. Decision rules, each with one hyperparameter selected on val:
  cos         global threshold on the cosine
  centered    cosine minus the concept's mean cosine over the (unlabelled) val images
  softmax     per-image softmax at temperature tau, then a global threshold
  topk        the k highest-scoring concepts
With the training label frequencies (train_stats.py, no image-label pairs):
  prior_only  no-image baseline: the n most frequent concepts for every image
  priors      sigmoid(alpha * (cos - mu_c) + logit(prior_c)), alpha and threshold on val

Usage:
    python train_stats.py                      # once
    python zeroshot.py --model medsiglip
    python zeroshot.py --model medsiglip --from-cache   # re-run the rules on cached scores
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from backends import MODEL_CHOICES, build_backend
from common import (apply_precision, load_config, load_cui_names, load_mlb, per_image_f1,
                    stochastic_modules_active, threshold_sweep, write_submission)
from data import build_loader
from train_stats import stats_path


@torch.no_grad()
def cosine_scores(backend, loader, text, device, chunk):
    use_amp = device.type == "cuda" and backend.amp
    t = text.to(device)
    scores, labels, keys = [], [], []
    for x, y, k in loader:
        for s in range(0, x.shape[0], chunk):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                f = backend.encode_image(x[s:s + chunk].to(device, non_blocking=True))
            scores.append((F.normalize(f.float(), dim=-1) @ t.t()).half().cpu())
        labels.append(y.bool())
        keys.extend(k)
    return torch.cat(scores).float(), torch.cat(labels), keys


def score_grid(scores, n_classes, n=80):
    """Thresholds at score quantiles giving ~0.1..30 predictions per image
    (works for any score range, e.g. SigLIP cosines around 0)."""
    flat = scores.flatten()
    if flat.numel() > 2_000_000:
        flat = flat[torch.randint(0, flat.numel(), (2_000_000,))]
    per_img = np.geomspace(0.1, 30, n)
    q = torch.tensor(1 - per_img / n_classes, dtype=torch.float32).clamp(0, 1)
    return sorted(set(torch.quantile(flat, q).tolist()))


def topk_matrix(scores, k):
    pred = torch.zeros_like(scores, dtype=torch.bool)
    pred.scatter_(1, scores.topk(k, dim=1).indices, True)
    return pred


def encode_scores(args, cfg, mlb, cui_order, device):
    """GPU part: encode val/test images and CUI prompts -> cosine matrices."""
    mcfg = cfg["models"].get(args.model, {})
    torch.manual_seed(args.seed)
    backend = build_backend(args.model, cfg["paths"].get("hf_cache"), mcfg)
    apply_precision(backend, cfg)
    if not backend.has_text:
        raise SystemExit(f"{args.model} is image-only (no text encoder): zero-shot does not "
                         "apply. Its no-training baseline is KNN retrieval.")
    text = backend.text_embeddings(load_cui_names(cfg["paths"].get("cui_mapping"), cui_order),
                                   device)
    backend.drop_text()

    backend.visual.to(device).eval()
    active = stochastic_modules_active(backend.visual)
    assert not active, f"stochastic modules active in eval: {active[:5]}"

    ex = cfg["extract"]
    chunk = mcfg.get("eval_chunk", 64)
    mk = lambda split: build_loader(cfg["paths"][f"{split}_shards"], mlb, backend,
                                    ex["batch_size"], ex["num_workers"], ex["prefetch_factor"])
    vs, vl, vk = cosine_scores(backend, mk("val"), text, device, chunk)
    ts, tl, tk = cosine_scores(backend, mk("test"), text, device, chunk)
    return vs, vl, vk, ts, tl, tk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "..", "common", "config.yaml"))
    ap.add_argument("--model", required=True, choices=MODEL_CHOICES)
    ap.add_argument("--from-cache", action="store_true",
                    help="reuse this run's cosine_scores.pt: no model, no GPU, seconds on CPU")
    ap.add_argument("--taus", type=float, nargs="+", default=[10, 30, 100])
    ap.add_argument("--prior-scales", type=float, nargs="+",
                    default=[1, 3, 10, 20, 30, 50, 100, 200])
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mlb = load_mlb(cfg["paths"]["mlb"])
    cui_order = list(mlb.classes_)
    out_dir = os.path.join(cfg["paths"]["out_root"], args.model)
    cache = os.path.join(out_dir, "cosine_scores.pt")

    if args.from_cache:
        z = torch.load(cache, weights_only=False)
        vs, vl, vk = z["val_scores"].float(), z["val_labels"].bool(), z["val_keys"]
        ts, tl, tk = z["test_scores"].float(), z["test_labels"].bool(), z["test_keys"]
        print(f"loaded cached scores from {cache}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vs, vl, vk, ts, tl, tk = encode_scores(args, cfg, mlb, cui_order, device)
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"val_scores": vs.half(), "val_labels": vl, "val_keys": vk,
                    "test_scores": ts.half(), "test_labels": tl, "test_keys": tk}, cache)

    C = vs.shape[1]
    print(f"val {tuple(vs.shape)}  test {tuple(ts.shape)}  "
          f"val cosine: mean {vs.mean():.3f}, class-mean range "
          f"[{vs.mean(0).min():.3f}, {vs.mean(0).max():.3f}]")
    results = {"model": args.model, "rules": {}}

    def finish(rule, val_f1, test_pred, params):
        tf1 = float(per_image_f1(test_pred, tl).mean())
        per_img = float(test_pred.float().sum(1).mean())
        write_submission(os.path.join(out_dir, f"test_{rule}.csv"), tk, test_pred, cui_order)
        results["rules"][rule] = {"val_f1": val_f1, "test_f1": tf1,
                                  "test_pred_per_img": per_img, **params}
        print(f"{rule:10s} val F1 {val_f1:.4f} -> test F1 {tf1:.4f}  "
              f"({per_img:.2f} CUIs/img)  {params}")

    # strict zero-shot: no labels used except one decision parameter tuned on val
    t, vf1, _ = threshold_sweep(vs, vl, grid=score_grid(vs, C))
    finish("cos", vf1, ts >= t, {"threshold": t})

    mu = vs.mean(0, keepdim=True)  # per-concept mean over UNLABELLED val images
    vc, tc = vs - mu, ts - mu
    t, vf1, _ = threshold_sweep(vc, vl, grid=score_grid(vc, C))
    finish("centered", vf1, tc >= t, {"threshold": t})

    best = None
    for tau in args.taus:
        vp = torch.softmax(tau * vs, dim=1)
        t, vf1, _ = threshold_sweep(vp, vl, grid=score_grid(vp, C))
        if best is None or vf1 > best[2]:
            best = (tau, t, vf1)
    tau, t, vf1 = best
    finish("softmax", vf1, torch.softmax(tau * ts, dim=1) >= t, {"tau": tau, "threshold": t})

    best = max(((k, float(per_image_f1(topk_matrix(vs, k), vl).mean()))
                for k in range(1, args.max_k + 1)), key=lambda kv: kv[1])
    finish("topk", best[1], topk_matrix(ts, best[0]), {"k": best[0]})

    # label PRIORS (training label frequencies; no image-label pairs)
    st_path = stats_path(cfg)
    if not os.path.exists(st_path):
        print(f"skipping prior rules: {st_path} missing (run train_stats.py)")
    else:
        st = torch.load(st_path, weights_only=False)
        assert st["classes"] == cui_order, "train_stats.pt was built with a different mlb"
        prior = ((st["counts"].float() + 1) / (st["n"] + 2)).clamp(1e-6, 1 - 1e-6)
        logit_prior = torch.log(prior / (1 - prior)).unsqueeze(0)

        order = st["counts"].argsort(descending=True)
        def top_n(n, rows):
            p = torch.zeros(rows, C, dtype=torch.bool)
            p[:, order[:n]] = True
            return p
        best = max(((n, float(per_image_f1(top_n(n, len(vl)), vl).mean()))
                    for n in range(1, 11)), key=lambda kv: kv[1])
        finish("prior_only", best[1], top_n(best[0], len(tl)), {"n": best[0]})

        best = None
        for sc in args.prior_scales:
            vp = torch.sigmoid(sc * vc + logit_prior)
            t, vf1, _ = threshold_sweep(vp, vl)
            if best is None or vf1 > best[2]:
                best = (sc, t, vf1)
        sc, t, vf1 = best
        finish("priors", vf1, torch.sigmoid(sc * tc + logit_prior) >= t,
               {"scale": sc, "threshold": t})

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
