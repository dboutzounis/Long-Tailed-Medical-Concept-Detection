"""Exploratory data analysis of the concept annotations (Section 3.4).

Writes Table 3.1 (most frequent concepts), Table 3.2 (concepts that appear once), Figure 3.2
(long-tail distribution) and Figure 3.3 (concepts per image), and prints per-image statistics.
Statistics are computed over the provided training and validation annotations (concepts.csv).

Usage:
    python eda.py --concepts concepts.csv --cui-mapping cui_mapping.csv --out-dir eda
"""
import argparse
import os
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#176BD7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", required=True, help="ID,CUIs with ';'-separated CUIs")
    ap.add_argument("--cui-mapping", required=True, help="CUI,Canonical name")
    ap.add_argument("--out-dir", default="eda")
    ap.add_argument("--n-rare", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.concepts)
    names = dict(pd.read_csv(args.cui_mapping)[["CUI", "Canonical name"]].values)
    cui_lists = df["CUIs"].dropna().str.split(";")
    counts = Counter(c for cuis in cui_lists for c in cuis)
    per_image = cui_lists.apply(len)

    print(f"images: {len(df)}  unique CUIs: {len(counts)}")
    print(f"concepts per image: mean {per_image.mean():.2f}, max {per_image.max()}, "
          f"images with one concept: {(per_image == 1).sum()}")

    top = pd.DataFrame([(i + 1, c, names.get(c, ""), n) for i, (c, n) in enumerate(counts.most_common(10))],
                       columns=["Rank", "CUI", "UMLS term", "Images"])
    top.to_csv(os.path.join(args.out_dir, "top10_concepts.csv"), index=False)
    print("\n" + top.to_string(index=False))

    singles = sorted(c for c, n in counts.items() if n == 1)
    rare = pd.DataFrame([(c, names.get(c, "")) for c in singles], columns=["CUI", "UMLS term"])
    rare.to_csv(os.path.join(args.out_dir, "single_occurrence_concepts.csv"), index=False)
    print(f"\n{len(rare)} concepts appear in a single image, e.g.:")
    print(rare.sample(min(args.n_rare, len(rare)), random_state=args.seed).to_string(index=False))

    freqs = sorted(counts.values(), reverse=True)
    ranks = np.arange(1, len(freqs) + 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(ranks, freqs, color=BLUE, linewidth=2)
    ax.fill_between(ranks, freqs, color=BLUE, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("class index")
    ax.set_ylabel("# of appearances")
    ax.set_title("Long Tail Distribution of Concepts")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "long_tail.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(7, 6))
    bins = np.arange(1, per_image.max() + 2) - 0.5
    ax.hist(per_image, bins=bins, color=BLUE, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("# of cuis per image")
    ax.set_ylabel("# of images")
    ax.set_title("Distribution of CUIs per Image")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cuis_per_image.png"), dpi=200)
    print(f"\nWrote tables and figures to {args.out_dir}/")


if __name__ == "__main__":
    main()
