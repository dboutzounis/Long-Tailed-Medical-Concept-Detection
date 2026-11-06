"""Fit the MultiLabelBinarizer on the training shards (Section 3.6.1).

Every script reloads the saved binarizer (paths.mlb), so concept indices are the same for all
models. Also writes the per-concept pos_weight (paths.pos_weight); it is not used in the thesis.

Usage:
    python fit_label_binarizer.py --config ../cnn/configs/base.yaml
"""
import argparse
import os
import pickle

import numpy as np
import torch
import yaml
from braceexpand import braceexpand
from sklearn.preprocessing import MultiLabelBinarizer
import webdataset as wds


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    def expand(obj):
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        return obj

    return expand(cfg)


def parse_cuis(sample):
    raw = sample.get("cuis", [])
    if not raw:
        return []
    return [c.strip() for c in raw[0].split(";") if c.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    num_classes_expected = cfg["data"]["num_classes"]
    pw_min = cfg["pos_weight"]["min_clamp"]
    pw_max = cfg["pos_weight"]["max_clamp"]

    shard_urls = list(braceexpand(paths["train_shards"]))
    print(f"Streaming {len(shard_urls)} train shards...")

    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=False)
        .decode("pil")
        .to_tuple("json")
    )

    all_cui_lists = []
    n_samples = 0
    for (json_data,) in dataset:
        all_cui_lists.append(parse_cuis(json_data))
        n_samples += 1
        if n_samples % 10000 == 0:
            print(f"  {n_samples} samples streamed...")

    print(f"Total samples: {n_samples}")

    print("Fitting MultiLabelBinarizer...")
    mlb = MultiLabelBinarizer(sparse_output=False)
    mlb.fit(all_cui_lists)

    n_classes = len(mlb.classes_)
    if n_classes != num_classes_expected:
        print(f"WARNING: expected {num_classes_expected} classes, got {n_classes}. "
              f"Update config.data.num_classes.")
    print(f"{n_classes} classes found.")

    print("Computing pos_weight...")
    binary_matrix = mlb.transform(all_cui_lists)
    pos_counts = binary_matrix.sum(axis=0).astype(np.float32)
    n = float(n_samples)
    neg_counts = n - pos_counts
    raw_weight = neg_counts / np.maximum(pos_counts, 1.0)
    pos_weight = np.clip(raw_weight, pw_min, pw_max).astype(np.float32)

    print(f"pos_weight — min: {pos_weight.min():.2f}, max: {pos_weight.max():.2f}, "
          f"mean: {pos_weight.mean():.2f}")

    os.makedirs(os.path.dirname(paths["mlb"]), exist_ok=True)
    with open(paths["mlb"], "wb") as f:
        pickle.dump(mlb, f)
    print(f"Saved MLB -> {paths['mlb']}")

    torch.save(torch.tensor(pos_weight), paths["pos_weight"])
    print(f"Saved pos_weight -> {paths['pos_weight']}")


if __name__ == "__main__":
    main()
