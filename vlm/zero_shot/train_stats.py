"""Count how often each concept occurs in the training split (label priors of Section 4.4.3).

Writes <out_root>/train_stats.pt with the number of training images and per-concept counts.

Usage:
    python train_stats.py
"""

import argparse
import os
import sys

import torch
import webdataset as wds
from braceexpand import braceexpand

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from common import load_config, load_mlb, parse_cuis


def stats_path(cfg):
    return os.path.join(cfg["paths"]["out_root"], "train_stats.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "..", "common", "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    mlb = load_mlb(cfg["paths"]["mlb"])
    idx = {c: i for i, c in enumerate(mlb.classes_)}
    counts = torch.zeros(len(idx), dtype=torch.long)
    n = 0
    ds = wds.WebDataset(list(braceexpand(cfg["paths"]["train_shards"])), shardshuffle=False)
    for sample in ds.decode().to_tuple("json"):
        for c in parse_cuis(sample[0]):
            if c in idx:
                counts[idx[c]] += 1
        n += 1
        if n % 20000 == 0:
            print(f"  {n} samples...")
    out = stats_path(cfg)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"n": n, "counts": counts, "classes": list(mlb.classes_)}, out)
    print(f"{n} train images, {int((counts > 0).sum())}/{len(idx)} classes seen -> {out}")


if __name__ == "__main__":
    main()
