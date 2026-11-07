"""Agreement and error correlation between the backbone models (Table 5.5, Figure 5.1).

Agreement is the mean Jaccard similarity of the predicted concept sets of two models.
Error correlation is the Pearson correlation of their per-image F1 scores.

Usage (from cnn/, after slurm/submit.sh eval configs/backbones/*.yaml):
    python backbone_agreement.py \
        --runs-dir $WORK/Long-Tailed-Medical-Concept-Detection/runs/cnn/backbones \
        --gt-csv $WORK/imageclefmed2026_webdataset/test_concepts.csv --out-dir agreement
"""
import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_sets(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    return {r.ID: set(c for c in r.CUIs.split(";") if c) for r in df.itertuples()}


def jaccard(a, b):
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def f1(pred, gt):
    return 1.0 if not pred and not gt else 2 * len(pred & gt) / (len(pred) + len(gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="one <backbone>_<loss>/test_predictions.csv per run")
    ap.add_argument("--gt-csv", required=True, help="ground truth of the test split (ID,CUIs)")
    ap.add_argument("--out-dir", default="agreement")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gt = read_sets(args.gt_csv)
    runs = sorted(d for d in os.listdir(args.runs_dir)
                  if os.path.exists(os.path.join(args.runs_dir, d, "test_predictions.csv")))
    preds = {r: read_sets(os.path.join(args.runs_dir, r, "test_predictions.csv")) for r in runs}
    ids = sorted(set(gt).intersection(*[set(p) for p in preds.values()]))
    print(f"{len(runs)} runs, {len(ids)} test images")

    f1s = {r: np.array([f1(preds[r][i], gt[i]) for i in ids]) for r in runs}
    rows = []
    for a, b in itertools.combinations(runs, 2):
        (bb_a, loss_a), (bb_b, loss_b) = a.rsplit("_", 1), b.rsplit("_", 1)
        rows.append({
            "a": a, "b": b,
            "agreement": np.mean([jaccard(preds[a][i], preds[b][i]) for i in ids]),
            "error_corr": np.corrcoef(f1s[a], f1s[b])[0, 1],
            "same_backbone": bb_a == bb_b, "loss_a": loss_a, "loss_b": loss_b,
        })
    pairs = pd.DataFrame(rows)
    pairs.to_csv(os.path.join(args.out_dir, "pairs.csv"), index=False)

    groups = {
        "All pairs": np.ones(len(pairs), bool),
        "Both trained with ASL": (pairs.loss_a == "asl") & (pairs.loss_b == "asl"),
        "Both trained with BCE": (pairs.loss_a == "bce") & (pairs.loss_b == "bce"),
        "Same backbone, different loss": pairs.same_backbone,
        "Same loss, different backbone": (pairs.loss_a == pairs.loss_b) & ~pairs.same_backbone,
    }
    summary = pd.DataFrame([{"pairs": name, "n": int(m.sum()),
                             "agreement": pairs.agreement[m].mean(),
                             "error_corr": pairs.error_corr[m].mean()} for name, m in groups.items()])
    print(summary.round(3).to_string(index=False))
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    # full error-correlation matrix, ordered by mean correlation with the other models
    corr = pd.DataFrame(np.eye(len(runs)), index=runs, columns=runs)
    for r in rows:
        corr.loc[r["a"], r["b"]] = corr.loc[r["b"], r["a"]] = r["error_corr"]
    order = (corr.sum(1) - 1).sort_values(ascending=False).index
    corr = corr.loc[order, order]
    labels = [f"{r.rsplit('_', 1)[0]} ({r.rsplit('_', 1)[1].upper()})" for r in order]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="Blues", vmin=corr.values[corr.values < 1].min(), vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson correlation of per-image F1")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"error_correlation.{ext}"), dpi=200)
    print(f"Wrote results to {args.out_dir}/")


if __name__ == "__main__":
    main()
