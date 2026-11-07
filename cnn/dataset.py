"""WebDataset loader for the ImageCLEFmedical 2026 shards (JPEG + JSON with the CUIs)."""
import random

import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
import webdataset as wds
from braceexpand import braceexpand


def _parse_cuis(json_data):
    """Extract CUI list from decoded JSON dict. Format: {"cuis": ["C0001;C0002"]}"""
    raw = json_data.get("cuis", [])
    if not raw:
        return []
    return [c.strip() for c in raw[0].split(";") if c.strip()]


def _seed_worker(worker_id):
    """Seed each DataLoader worker deterministically from its torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _collate_with_keys(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    keys = [b[2] for b in batch]
    return images, labels, keys


def build_dataloader(split, config, local=False, return_keys=False):
    paths = config["paths"]
    data_cfg = config["data"]

    shard_pattern = paths[f"{split}_shards"]
    shard_urls = list(braceexpand(shard_pattern))

    with open(paths["mlb"], "rb") as f:
        mlb = pickle.load(f)

    num_classes = data_cfg["num_classes"]
    img_size = data_cfg["image_size"]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    is_train = split == "train"

    def preprocess(sample):
        key = sample["__key__"]
        img = transform(sample["jpg"].convert("RGB"))
        cuis = _parse_cuis(sample["json"])
        if cuis:
            label = torch.tensor(mlb.transform([cuis])[0], dtype=torch.float32)
        else:
            label = torch.zeros(num_classes, dtype=torch.float32)
        if return_keys:
            return img, label, key
        return img, label

    seed = config["training"]["seed"]

    pipeline = wds.WebDataset(
        shard_urls,
        shardshuffle=len(shard_urls) if is_train else False,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        seed=seed,
    )

    if is_train:
        pipeline = pipeline.repeat().shuffle(data_cfg["train_shuffle_buffer"])

    pipeline = pipeline.decode("pil").map(preprocess)

    batch_size = data_cfg["batch_size"]
    num_workers = 0 if local else data_cfg["num_workers"]
    prefetch_factor = None if num_workers == 0 else data_cfg["prefetch_factor"]

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        pipeline,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        drop_last=is_train,
        collate_fn=_collate_with_keys if return_keys else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )

    return loader
