"""Shared helpers for LoRA training and feature extraction."""
import os
import platform
import random
import socket
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml

THRESHOLDS = [i / 100 for i in range(1, 100)]  # 0.01 .. 0.99, same grid as frozen_heads


# ---------------------------------------------------------------------------
# Config and shared imports
# ---------------------------------------------------------------------------
def _expand(o):
    if isinstance(o, str):
        return os.path.expandvars(o)
    if isinstance(o, dict):
        return {k: _expand(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_expand(v) for v in o]
    return o


def load_config(path):
    with open(path) as f:
        return _expand(yaml.safe_load(f))


def import_common():
    """Import backends.py and common.py from vlm/common/."""
    code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import backends
    import common
    return backends, common


def build_backend(backends_mod, name, cfg):
    mcfg = cfg["models"].get(name, {})
    kw = {k: mcfg[k] for k in ("repo_dir", "model_dir", "vision_model_name",
                               "language_model_name") if mcfg.get(k) is not None}
    kw["grad_ckpt"] = False
    backend = backends_mod.build_backend(name, cfg["paths"]["hf_cache"], kw)
    if backend.has_text:
        backend.drop_text()
    for p in backend.visual.parameters():
        p.requires_grad_(False)
    return backend


# ---------------------------------------------------------------------------
# Distributed + reproducibility
# ---------------------------------------------------------------------------
def init_distributed():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return rank, world, device


def is_dist():
    return dist.is_available() and dist.is_initialized()


def all_reduce_(t):
    if is_dist():
        dist.all_reduce(t)
    return t


def barrier():
    if is_dist():
        dist.barrier()


def set_determinism(seed, deterministic=True, deterministic_attention=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # also exported in sbatch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if deterministic_attention and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def env_info(device, world):
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "hostname": socket.gethostname(),
            "platform": platform.platform(), "world_size": world,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


# ---------------------------------------------------------------------------
# Data: WebDataset shards, extraction-time transform only (no augmentation)
# ---------------------------------------------------------------------------
def _rank():
    return dist.get_rank() if is_dist() else 0


def seeded_shuffle(bufsize, seed, initial=1000):
    """Reproducible sample-level shuffle buffer; RNG = f(seed, rank, worker)."""
    def _shuffle(src):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = random.Random("%s-%s-%s" % (seed, _rank(), wid))
        it, buf = iter(src), []
        for sample in it:
            buf.append(sample)
            if len(buf) < bufsize:
                try:
                    buf.append(next(it))
                except StopIteration:
                    pass
            if len(buf) >= min(initial, bufsize):
                k = rng.randrange(len(buf))
                buf[k], buf[-1] = buf[-1], buf[k]
                yield buf.pop()
        rng.shuffle(buf)
        yield from buf
    return _shuffle


def _collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            [b[2] for b in batch])


def _seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(shards, classes, transform, parse_cuis, batch_size, num_workers,
                prefetch_factor, train, seed=42, shuffle_buffer=5000, distributed=False):
    """train=True: shard + sample shuffle, infinite stream, drop_last.
    train=False: fixed order, single pass. Same transform in both cases."""
    import webdataset as wds
    from braceexpand import braceexpand
    urls = list(braceexpand(shards))
    index = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)

    def preprocess(sample):
        img = transform(sample["jpg"].convert("RGB"))
        y = torch.zeros(n_cls, dtype=torch.float32)
        for c in parse_cuis(sample["json"]):
            if c in index:
                y[index[c]] = 1.0
        return img, y, sample["__key__"]

    kw = dict(shardshuffle=len(urls) if train else False,
              nodesplitter=wds.split_by_node if distributed else (lambda src: src),
              workersplitter=wds.split_by_worker, seed=seed)
    try:
        ds = wds.WebDataset(urls, empty_check=False, **kw)
    except TypeError:
        ds = wds.WebDataset(urls, **kw)
    if train:
        ds = ds.repeat().compose(seeded_shuffle(shuffle_buffer, seed))
    ds = ds.decode("pil").map(preprocess)
    g = torch.Generator()
    g.manual_seed(int(seed) * 1000 + _rank())
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=train, collate_fn=_collate, pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker if num_workers > 0 else None, generator=g,
        persistent_workers=train and num_workers > 0)


# ---------------------------------------------------------------------------
# Loss and metric (identical to frozen_heads/train_frozen_heads.py)
# ---------------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.02, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        p_neg = 1 - p
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)
        loss = y * torch.log(p.clamp(min=self.eps)) + (1 - y) * torch.log(p_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = p * y + p_neg * (1 - y)
            gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            loss = loss * torch.pow(1 - pt, gamma)
        return -loss.sum(dim=1).mean()


@torch.no_grad()
def f1_sums(probs, y_bool):
    """Sum over images of per-image F1, for every threshold in THRESHOLDS.
    Summable across batches and ranks -> exact global tuned F1 without gathering."""
    n_gt = y_bool.sum(1)
    out = torch.empty(len(THRESHOLDS), dtype=torch.float64, device=probs.device)
    for i, t in enumerate(THRESHOLDS):
        pred = probs >= t
        tp = (pred & y_bool).sum(1)
        denom = pred.sum(1) + n_gt
        f1 = torch.where(denom > 0, 2 * tp / denom.clamp(min=1), torch.ones_like(denom))
        out[i] = f1.double().sum()
    return out


def best_threshold(f1_vec):
    vals = f1_vec.tolist()
    i = max(range(len(vals)), key=lambda k: (vals[k], -k))  # ties -> lowest threshold
    return i, THRESHOLDS[i], vals[i]


# ---------------------------------------------------------------------------
# Model: encoder (+LoRA) -> L2-normalised embedding -> linear head
# ---------------------------------------------------------------------------
class EncoderClassifier(nn.Module):
    def __init__(self, backend, head, amp):
        super().__init__()
        self.visual = backend.visual        # registered: LoRA params are trained
        self.head = head
        self._encode = backend.encode_image  # uses backend.visual (same object)
        self.amp = amp

    def embed(self, x):
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if self.amp and x.is_cuda else nullcontext())
        with ctx:
            e = self._encode(x)
        return F.normalize(e.float(), dim=-1)

    def forward(self, x):
        return self.head(self.embed(x))
