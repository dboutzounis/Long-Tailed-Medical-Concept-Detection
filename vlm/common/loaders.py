"""Config loading and the ordered WebDataset loader used for feature extraction.

The loader reads the shards in a fixed order, so every encoder's cached features
share the same row order (checked by comparing their label matrices).
"""
import os
import pickle

import torch
import webdataset as wds
import yaml
from braceexpand import braceexpand
from torch.utils.data import DataLoader


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


def _parse_cuis(json_data):
    """CUIs of one sample; the shard JSON stores them as {"cuis": ["C0001;C0002"]}."""
    raw = json_data.get("cuis", [])
    if not raw:
        return []
    return [c.strip() for c in raw[0].split(";") if c.strip()]


def build_biomedclip_dataloader(split, config, biomedclip_preprocess,
                                return_keys=False, raw=False):
    """Ordered (unshuffled) loader of one split; raw=True yields PIL images instead of tensors."""
    paths    = config["paths"]
    data_cfg = config["data"]

    shard_urls = list(braceexpand(paths[f"{split}_shards"]))

    with open(paths["mlb"], "rb") as f:
        mlb = pickle.load(f)

    num_classes = data_cfg["num_classes"]

    def preprocess(sample):
        key = sample["__key__"]
        pil = sample["jpg"].convert("RGB")
        img = pil if raw else biomedclip_preprocess(pil)
        cuis = _parse_cuis(sample["json"])
        if cuis:
            label = torch.tensor(mlb.transform([cuis])[0], dtype=torch.float32)
        else:
            label = torch.zeros(num_classes, dtype=torch.float32)
        if return_keys:
            return img, label, key
        return img, label

    def collate_tensor(batch):
        images = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        if return_keys:
            return images, labels, [b[2] for b in batch]
        return images, labels

    def collate_raw(batch):
        images = [b[0] for b in batch]
        labels = torch.stack([b[1] for b in batch])
        if return_keys:
            return images, labels, [b[2] for b in batch]
        return images, labels

    if raw:
        collate = collate_raw
    elif return_keys:
        collate = collate_tensor
    else:
        collate = None

    pipeline = (
        wds.WebDataset(
            shard_urls,
            shardshuffle=False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        .decode("pil")
        .map(preprocess)
    )

    return DataLoader(
        pipeline,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        prefetch_factor=data_cfg["prefetch_factor"] if data_cfg["num_workers"] > 0 else None,
        drop_last=False,
        collate_fn=collate,
    )
