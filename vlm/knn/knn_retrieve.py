"""Nearest-neighbor retrieval and score-level fusion on frozen embeddings (Section 4.4.4).

A query's score for concept c is the similarity-weighted fraction of its K nearest
training images that carry c (Eq. 4.5); c is predicted when the score reaches rho.
With --fuse-dir, the score matrices of several encoders are combined as a weighted sum.

Usage:
    # val: joint K x rho sweep for one encoder
    python knn_retrieve.py --data-dir $ART/medsiglip_features --split val --diagnose
    # test at the selected cell, writing the predictions
    python knn_retrieve.py --data-dir $ART/medsiglip_features --split test --k 50 --ratio 0.35 \
        --diagnose --mlb-path $ART/mlb.pkl --output knn_medsiglip.csv
    # fusion: sweep the weights on val, then pass --fuse-weights on test
    python knn_retrieve.py --data-dir $ART/medimageinsight_features \
        --fuse-dir $ART/medsiglip_features $ART/biomedclip_features --fuse-k 50 50 20 --split val
    python knn_retrieve.py --self-test
"""
import argparse
import csv
import os
import pickle

import torch

BAND_EDGES = (10, 100, 1000, 10 ** 9)  # rare, tail, mid, head (training occurrences)


def load_split(data_dir, split, need_keys=False):
    embs = torch.nn.functional.normalize(torch.load(os.path.join(data_dir, f"{split}_embs.pt")).float(), dim=-1)
    lab_path = os.path.join(data_dir, f"{split}_labels.pt")
    labels = torch.load(lab_path).float() if os.path.exists(lab_path) else None
    keys = None
    if need_keys:
        with open(os.path.join(data_dir, f"{split}_keys.pkl"), "rb") as f:
            keys = pickle.load(f)
    return embs, labels, keys


@torch.no_grad()
def retrieve_topk(query_embs, train_embs, k_max, device, batch=2048):
    """Cosine similarities and train indices of the k_max nearest neighbors of every query."""
    train_embs = train_embs.to(device)
    sims_out, idx_out = [], []
    for i in range(0, query_embs.shape[0], batch):
        s = query_embs[i:i + batch].to(device) @ train_embs.T
        s, idx = torch.topk(s, k_max, dim=1, largest=True, sorted=True)
        sims_out.append(s.cpu())
        idx_out.append(idx.cpu())
    return torch.cat(sims_out), torch.cat(idx_out)


@torch.no_grad()
def knn_scores(sims, idx, train_labels, k, device="cpu", batch=2048):
    """Similarity-weighted vote of the first k neighbors, in [0, 1] (Eq. 4.5)."""
    train_labels = train_labels.to(device)
    out = torch.empty(sims.shape[0], train_labels.shape[1], dtype=torch.float16)
    for b in range(0, sims.shape[0], batch):
        s = sims[b:b + batch, :k].to(device)
        lab = train_labels[idx[b:b + batch, :k].to(device)]
        score = torch.einsum("bk,bkc->bc", s, lab) / s.sum(1, keepdim=True).clamp_min(1e-8)
        out[b:b + batch] = score.cpu().half()
    return out


def sample_f1(preds, labels, eps=1e-8):
    preds, labels = preds.float(), labels.float()
    tp = (preds * labels).sum(1)
    prec = tp / (preds.sum(1) + eps)
    rec = tp / (labels.sum(1) + eps)
    return (2 * prec * rec / (prec + rec + eps)).mean().item()


def report_metrics(preds, labels, tag=""):
    preds, labels = preds.float(), labels.float()
    tp = (preds * labels).sum(1)
    p = (tp / (preds.sum(1) + 1e-8)).mean().item()
    r = (tp / (labels.sum(1) + 1e-8)).mean().item()
    f1 = sample_f1(preds, labels)
    print(f"[{tag}]  F1={f1:.4f}  P={p:.4f}  R={r:.4f}  "
          f"pred/img={preds.sum(1).mean():.2f}  gt/img={labels.sum(1).mean():.2f}")
    return f1


def recall_by_band(preds, labels, train_labels, edges=BAND_EDGES):
    """Micro recall and precision per concept-frequency band (Tables 5.8 and 5.13)."""
    preds, labels = preds.float(), labels.float()
    freq = train_labels.sum(0)
    tp, pos, pred_pos = (preds * labels).sum(0), labels.sum(0), preds.sum(0)
    print(f"{'band':>6} {'train_freq':>14} {'#CUIs':>6} {'gt+':>8} {'recall':>7} {'prec':>7} {'TP':>7}")
    lo = 0
    for name, hi in zip(["rare", "tail", "mid", "head"], edges):
        m = (freq > lo) & (freq <= hi)
        gt, tpp, pp = pos[m].sum().item(), tp[m].sum().item(), pred_pos[m].sum().item()
        band = f"{lo + 1}-{'inf' if hi >= 10 ** 9 else hi}"
        print(f"{name:>6} {band:>14} {int(m.sum()):>6} {int(gt):>8} "
              f"{tpp / (gt + 1e-8):>7.3f} {tpp / (pp + 1e-8):>7.3f} {int(tpp):>7}")
        lo = hi


def simplex_grid(n, step):
    """All weight tuples of length n on a grid of `step` that sum to one."""
    m = int(round(1.0 / step))
    out = []

    def rec(parts_left, rem, acc):
        if parts_left == 1:
            out.append(tuple(v * step for v in acc + [rem]))
            return
        for v in range(rem + 1):
            rec(parts_left - 1, rem - v, acc + [v])

    rec(n, m, [])
    return out


def best_ratio(scores, labels, ratio_grid):
    return max((sample_f1(scores >= r, labels), r) for r in ratio_grid)


def joint_sweep(sims, idx, train_labels, labels, k_grid, ratio_grid, device):
    """Retrieve once, then evaluate every (K, rho) cell."""
    print(f"{'K':>5} | " + " ".join(f"r={r:>4.2f}" for r in ratio_grid))
    best = (-1.0, None, None)
    for k in k_grid:
        scores = knn_scores(sims, idx, train_labels, k, device)
        row = [sample_f1(scores >= r, labels) for r in ratio_grid]
        for r, f1 in zip(ratio_grid, row):
            if f1 > best[0]:
                best = (f1, k, r)
        print(f"{k:>5} | " + " ".join(f"{v:>6.4f}" for v in row))
    print(f"\nBEST cell: K={best[1]}  ratio={best[2]:.2f}  ->  F1={best[0]:.4f}")
    return best


def write_submission(preds, keys, mlb_path, output):
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)
    cui_lists = mlb.inverse_transform(preds.float().numpy())
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "CUIs"])
        for key, cuis in zip(keys, cui_lists):
            w.writerow([key, ";".join(cuis)])
    print(f"Wrote {output} ({len(keys)} rows)")


def fuse(args, tr_e, tr_l, q_e, q_l, q_keys, device):
    dirs = [args.data_dir] + list(args.fuse_dir)
    ks = args.fuse_k or [args.k or 50] * len(dirs)
    if len(ks) != len(dirs):
        raise SystemExit(f"--fuse-k needs {len(dirs)} values, got {len(ks)}")
    score_mats = []
    for d, k in zip(dirs, ks):
        if d == args.data_dir:
            te, tl, qe = tr_e, tr_l, q_e
        else:
            te, tl, _ = load_split(d, "train")
            qe, ql, _ = load_split(d, args.split)
            # scores are combined row by row, so every cache must share the same image order
            if not (torch.equal(tl, tr_l) and torch.equal(ql, q_l)):
                raise SystemExit(f"labels differ for {d}: features are not in the same order")
        s, i = retrieve_topk(qe, te, k, device)
        score_mats.append(knn_scores(s, i, tl, k, device).float())
    print(f"\n{len(dirs)}-encoder fusion  |  K={ks}")

    combos = [tuple(args.fuse_weights)] if args.fuse_weights else simplex_grid(len(dirs), args.fuse_weight_step)
    results = sorted(((*best_ratio(sum(w_ * s_ for w_, s_ in zip(w, score_mats)), q_l, args.ratio_grid), w)
                      for w in combos), key=lambda x: -x[0])
    for f1, r, w in results[:10]:
        print(f"  w=[{' '.join(f'{x:.2f}' for x in w)}]  ratio={r:.2f}  F1={f1:.4f}")
    f1, r, w = results[0]
    print(f"\nBEST: w=[{' '.join(f'{x:.2f}' for x in w)}] ratio={r:.2f} -> F1={f1:.4f}")
    preds = (sum(w_ * s_ for w_, s_ in zip(w, score_mats)) >= r).float()
    report_metrics(preds, q_l, tag="fusion")
    recall_by_band(preds, q_l, tr_l)
    if args.output:
        write_submission(preds, q_keys, args.mlb_path, args.output)


def self_test():
    """Synthetic clustered data: retrieval must recover the cluster labels."""
    torch.manual_seed(0)
    D, C, n_clusters = 64, 40, 12
    centers = torch.nn.functional.normalize(torch.randn(n_clusters, D), dim=-1)
    cl_labels = (torch.rand(n_clusters, C) < 0.12).float()

    def make(n):
        cl = torch.randint(0, n_clusters, (n,))
        e = torch.nn.functional.normalize(centers[cl] + 0.25 * torch.randn(n, D), dim=-1)
        return e, (cl_labels[cl] + (torch.rand(n, C) < 0.03).float()).clamp(0, 1)

    tr_e, tr_l = make(4000)
    va_e, va_l = make(800)
    sims, idx = retrieve_topk(va_e, tr_e, 50, "cpu")
    best = joint_sweep(sims, idx, tr_l, va_l, [1, 5, 20, 50], [0.2, 0.3, 0.4, 0.5], "cpu")
    assert best[0] > 0.5, best
    scores = knn_scores(sims, idx, tr_l, 50)
    assert abs(report_metrics(scores >= 0.35, va_l, "selftest") - sample_f1(scores >= 0.35, va_l)) < 1e-6
    recall_by_band(scores >= 0.35, va_l, tr_l)
    assert len(simplex_grid(3, 0.1)) == 66
    print("\nSELF-TEST PASSED")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--data-dir")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--k-grid", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100, 200])
    p.add_argument("--ratio-grid", type=float, nargs="+",
                   default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    p.add_argument("--diagnose", action="store_true", help="print recall/precision by frequency band")
    p.add_argument("--fuse-dir", nargs="+", help="feature dirs of the other encoders to fuse")
    p.add_argument("--fuse-k", type=int, nargs="+", help="K per encoder, --data-dir first")
    p.add_argument("--fuse-weights", type=float, nargs="+",
                   help="fixed fusion weights (test); if omitted, sweep the weight simplex")
    p.add_argument("--fuse-weight-step", type=float, default=0.1)
    p.add_argument("--k", type=int)
    p.add_argument("--ratio", type=float)
    p.add_argument("--mlb-path")
    p.add_argument("--output")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_e, tr_l, _ = load_split(args.data_dir, "train")
    q_e, q_l, q_keys = load_split(args.data_dir, args.split, need_keys=args.output is not None)
    print(f"train={tuple(tr_e.shape)}  {args.split}={tuple(q_e.shape)}  C={tr_l.shape[1]}  device={device}")

    if args.fuse_dir:
        fuse(args, tr_e, tr_l, q_e, q_l, q_keys, device)
        return

    sims, idx = retrieve_topk(q_e, tr_e, args.k or max(args.k_grid), device)
    if args.k and args.ratio:
        preds = (knn_scores(sims, idx, tr_l, args.k, device).float() >= args.ratio).float()
        report_metrics(preds, q_l, tag=f"{args.split} K={args.k} ratio={args.ratio:.2f}")
        if args.diagnose:
            recall_by_band(preds, q_l, tr_l)
        if args.output:
            write_submission(preds, q_keys, args.mlb_path, args.output)
        return

    best = joint_sweep(sims, idx, tr_l, q_l, args.k_grid, args.ratio_grid, device)
    if args.diagnose:
        preds = (knn_scores(sims, idx, tr_l, best[1], device).float() >= best[2]).float()
        recall_by_band(preds, q_l, tr_l)


if __name__ == "__main__":
    main()
