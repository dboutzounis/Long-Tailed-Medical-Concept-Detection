"""Shared helpers for the vision-language encoder experiments: config loading, labels,
concept names, sample-averaged F1, threshold sweeps and reproducibility.

Sample-averaged F1 follows the ImageCLEF convention: an image with an empty ground truth
and an empty prediction scores 1.
"""

import csv
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import yaml

DOMINANT_CONCEPTS = {"C0040405", "C1306645", "C0024485", "C0041618"}

PROMPT_TEMPLATES = [
    "a radiology image showing {name}",
    "a biomedical image of {name}",
    "medical imaging: {name}",
    "{name}",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _expand(obj):
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def load_config(path):
    with open(path) as f:
        cfg = _expand(yaml.safe_load(f))
    base_path = cfg.get("base_config")
    if base_path:
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base_path)
        with open(base_path) as f:
            base = _expand(yaml.safe_load(f))
        bdata = base.get("data", {})
        cfg["extract"] = {"batch_size": bdata.get("batch_size", 256),
                          "num_workers": bdata.get("num_workers", 8),
                          "prefetch_factor": bdata.get("prefetch_factor", 2)}
        merged = dict(base.get("paths", {}))
        merged.update({k: v for k, v in cfg.get("paths", {}).items() if v is not None})
        cfg["paths"] = merged
    cfg.setdefault("extract", {"batch_size": 256, "num_workers": 8, "prefetch_factor": 2})
    for key in ("train_shards", "val_shards", "test_shards", "mlb"):
        if key not in cfg["paths"]:
            raise KeyError(f"paths.{key} missing (set it in config.yaml or base_config.yaml)")
    return cfg


# ---------------------------------------------------------------------------
# Labels / names
# ---------------------------------------------------------------------------

def parse_cuis(json_data):
    """CUIs of one sample; the shard JSON stores them as {"cuis": ["C0001;C0002"]}."""
    raw = json_data.get("cuis", [])
    if not raw:
        return []
    return [c.strip() for c in raw[0].split(";") if c.strip()]


def load_mlb(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_cui_names(csv_path, cui_order):
    name_map = {}
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name_map[row["CUI"].strip()] = row["Canonical name"].strip()
    missing = [c for c in cui_order if c not in name_map]
    if missing:
        print(f"WARNING: {len(missing)} CUIs have no canonical name; using the CUI string.")
    return [name_map.get(c, c) for c in cui_order]


def prompts_for(names, template):
    return [template.format(name=n) for n in names]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_image_f1(pred, gt, empty_both=1.0):
    """pred, gt: (N, C) {0,1} tensors or arrays. Returns (N,) float64 array."""
    pred = torch.as_tensor(pred).float()
    gt = torch.as_tensor(gt).float()
    tp = (pred * gt).sum(1)
    n_pred = pred.sum(1)
    n_gt = gt.sum(1)
    denom = n_pred + n_gt
    f1 = torch.where(denom > 0, 2 * tp / denom.clamp(min=1), torch.full_like(tp, empty_both))
    return f1.double().numpy()


def threshold_sweep(probs, labels, grid=None, chunk=4096):
    """Global threshold maximising mean per-image F1. probs/labels (N, C)."""
    if grid is None:
        grid = np.round(np.arange(0.02, 0.96, 0.01), 2)
    probs = torch.as_tensor(probs).float()
    labels = torch.as_tensor(labels).float()
    n_gt = labels.sum(1)
    best_t, best_f1, curve = None, -1.0, []
    for t in grid:
        f1_sum = 0.0
        for s in range(0, probs.shape[0], chunk):
            p = (probs[s:s + chunk] >= t).float()
            g = labels[s:s + chunk]
            tp = (p * g).sum(1)
            denom = p.sum(1) + n_gt[s:s + chunk]
            f1 = torch.where(denom > 0, 2 * tp / denom.clamp(min=1), torch.ones_like(tp))
            f1_sum += f1.sum().item()
        f1_mean = f1_sum / probs.shape[0]
        curve.append((float(t), f1_mean))
        if f1_mean > best_f1:
            best_t, best_f1 = float(t), f1_mean
    return best_t, best_f1, curve


def write_submission(path, keys, pred_matrix, cui_order):
    """ImageCLEF-style CSV: ID,CUIs (semicolon-joined)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pred_matrix = torch.as_tensor(pred_matrix).bool().numpy()
    cui_order = np.asarray(cui_order)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "CUIs"])
        for k, row in zip(keys, pred_matrix):
            w.writerow([k, ";".join(cui_order[row].tolist())])


# ---------------------------------------------------------------------------
# Reproducibility and precision
# ---------------------------------------------------------------------------

def set_reproducible(seed, rank=0, deterministic=True):
    """Seed every RNG and request deterministic kernels."""
    import random
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    s = int(seed) * 1000 + int(rank)
    random.seed(s)
    np.random.seed(s % 2 ** 32)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


PRECISIONS = ("bf16", "fp32")


def apply_precision(backend, cfg):
    """One precision for every encoder (training.precision in config.yaml).
    bf16 = autocast bf16 for the encoder forward; fp32 = true fp32 (TF32 off).
    Head, loss, probabilities and thresholds are always fp32."""
    prec = cfg.get("training", {}).get("precision", "bf16")
    if prec not in PRECISIONS:
        raise ValueError(f"training.precision must be one of {PRECISIONS}, got {prec}")
    backend.amp = prec == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return prec


def stochastic_modules_active(model):
    """Names of dropout / stochastic-depth modules that are active (must be empty at eval)."""
    names = []
    for n, m in model.named_modules():
        if not m.training:
            continue
        if isinstance(m, nn.Dropout) and m.p > 0:
            names.append(n)
        elif ("DropPath" in type(m).__name__ or type(m).__name__ == "StochasticDepth") \
                and getattr(m, "drop_prob", 0) > 0:
            names.append(n)
        elif isinstance(getattr(m, "dropout", None), float) and m.dropout > 0:
            names.append(n)
    return names
