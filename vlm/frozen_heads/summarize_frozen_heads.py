#!/usr/bin/env python
"""Collect results.json of every frozen-head run into one table (stdout + CSV).

The MLP width with the highest validation F1 is marked as selected for each encoder.
"""
import argparse
import csv
import glob
import json
import os

ENC_ORDER = ["biomedclip", "medsiglip", "medimageinsight", "raddino", "concat3", "concat4"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root",
                   default="$WORK/Long-Tailed-Medical-Concept-Detection/artifacts/frozen_heads")
    args = p.parse_args()
    root = os.path.expandvars(args.out_root)

    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "results.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        print("no results under %s" % root)
        return

    # MLP width per encoder: highest validation F1
    selected = set()
    for enc in {r["encoder"] for r in rows}:
        mlps = [r for r in rows if r["encoder"] == enc and r["head"] == "mlp"]
        if mlps:
            best = max(mlps, key=lambda r: r["val_f1"])
            selected.add((enc, best["run"]))

    def key(r):
        e = ENC_ORDER.index(r["encoder"]) if r["encoder"] in ENC_ORDER else 99
        return (e, r["run"])

    rows.sort(key=key)
    header = ["encoder", "run", "best_epoch", "epochs_run", "val_loss", "val_f1",
              "threshold", "test_f1", "preds/img", "seconds", "note"]
    out = []
    for r in rows:
        note = []
        if (r["encoder"], r["run"]) in selected:
            note.append("selected width")
        if r.get("hit_max_epochs"):
            note.append("HIT MAX EPOCHS")
        if r.get("lambda_at_grid_edge"):
            note.append("LAMBDA AT GRID EDGE")
        if r["head"] == "ridge":
            note.append("lambda=%.3g (val MSE)" % r["best_lambda"])
        if "tau_final" in r:
            note.append("tau=%.2f" % r["tau_final"])
        out.append([r["encoder"], r["run"], r.get("best_epoch", "-"), r.get("epochs_run", "-"),
                    "%.5f" % r["best_val_loss"], "%.4f" % r["val_f1"], "%.2f" % r["threshold"],
                    "%.4f" % r["test_f1"], "%.2f" % r["mean_pred_per_image_test"],
                    "%.0f" % r["train_seconds"], "; ".join(note)])

    widths = [max(len(str(x)) for x in col) for col in zip(header, *out)]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    prev = None
    for row in out:
        if prev is not None and row[0] != prev:
            print()
        print(fmt.format(*[str(x) for x in row]))
        prev = row[0]

    csv_path = os.path.join(root, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out)
    print("\nwritten: %s" % csv_path)


if __name__ == "__main__":
    main()
