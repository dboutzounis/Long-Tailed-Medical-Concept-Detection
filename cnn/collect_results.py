"""Collect the evaluate.sbatch logs into one results table (Table 5.4).

Usage (from cnn/, after the eval jobs finish):
    python collect_results.py --logs-dir logs --out-csv backbones.csv --out-md backbones.md
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

RE_CONFIG = re.compile(r"Using config:\s*(\S+)\.yaml")
RE_BEST_VAL = re.compile(
    r"\*\*\* Best threshold:\s*([\d.]+)\s*->\s*"
    r"F1=([\d.]+)\s+Prec=([\d.]+)\s+Rec=([\d.]+)\s+MeanCUIs=([\d.]+)\s*\*\*\*"
)
RE_TEST = re.compile(
    r"Test metrics @ threshold=([\d.]+)\s*\|\s*"
    r"F1=([\d.]+)\s+Prec=([\d.]+)\s+Rec=([\d.]+)\s+MeanCUIs=([\d.]+)"
)
COLS = ["f1", "precision", "recall", "mean_cuis"]


def parse_log(path):
    with open(path, errors="replace") as f:
        text = f.read()
    m_cfg, m_val, m_test = RE_CONFIG.search(text), RE_BEST_VAL.search(text), RE_TEST.search(text)
    if not (m_cfg and m_val and m_test):
        return None
    row = {"run": os.path.basename(m_cfg.group(1)), "threshold": float(m_test.group(1))}
    row.update({f"val_{c}": float(v) for c, v in zip(COLS, m_val.groups()[1:])})
    row.update({f"test_{c}": float(v) for c, v in zip(COLS, m_test.groups()[1:])})
    row["_mtime"] = os.path.getmtime(path)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--pattern", default="eval_*.out")
    ap.add_argument("--out-csv", default="results.csv")
    ap.add_argument("--out-md", default="results.md")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.logs_dir, args.pattern))):
        row = parse_log(path)
        if row is None:
            print(f"SKIP {path}: incomplete log", file=sys.stderr)
        else:
            rows.append(row)
    if not rows:
        sys.exit("No complete eval logs found.")

    # keep the most recent log of each run
    df = (pd.DataFrame(rows).sort_values("_mtime").drop_duplicates("run", keep="last")
          .drop(columns="_mtime").sort_values("test_f1", ascending=False).reset_index(drop=True))
    print(df.to_string(index=False))
    df.to_csv(args.out_csv, index=False)
    with open(args.out_md, "w") as f:
        f.write(df.to_markdown(index=False) + "\n")
    print(f"\nWrote {args.out_csv} and {args.out_md}")


if __name__ == "__main__":
    main()
