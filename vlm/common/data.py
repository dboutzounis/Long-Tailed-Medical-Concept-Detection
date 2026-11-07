"""Ordered WebDataset loader that applies each backend's extraction-time preprocessing."""
import torch
import webdataset as wds
from braceexpand import braceexpand
from torch.utils.data import DataLoader

from common import parse_cuis


def _collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch])


def build_loader(shards, mlb, backend, batch_size, num_workers, prefetch_factor):
    """Unshuffled loader of (images, label vectors, keys), in shard order."""
    urls = list(braceexpand(shards))
    tf = backend.eval_transform
    classes = {c: i for i, c in enumerate(mlb.classes_)}
    n_cls = len(classes)

    def preprocess(sample):
        img = tf(sample["jpg"].convert("RGB"))
        y = torch.zeros(n_cls, dtype=torch.float32)
        for c in parse_cuis(sample["json"]):
            if c in classes:
                y[classes[c]] = 1.0
        return img, y, sample["__key__"]

    kw = dict(shardshuffle=False, workersplitter=wds.split_by_worker)
    try:
        ds = wds.WebDataset(urls, empty_check=False, **kw)
    except TypeError:  # older webdataset without empty_check
        ds = wds.WebDataset(urls, **kw)
    ds = ds.decode("pil").map(preprocess)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      prefetch_factor=prefetch_factor if num_workers > 0 else None,
                      drop_last=False, collate_fn=_collate)
