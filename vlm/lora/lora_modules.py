"""LoRA for the nn.Linear layers inside transformer blocks: y = W x + (alpha / r) B A x, with
B = 0 at initialization. After training the update is merged into W, so the adapted encoder
has the original architecture. (A DoRA variant is included but not used in the thesis.)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dora=False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.scaling, self.use_dora = rank, alpha / rank, dora
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
        if dora:
            self.lora_magnitude = nn.Parameter(base.weight.detach().float().norm(p=2, dim=1))

    def delta_weight(self):
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x):
        if not self.use_dora:
            return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        w = self.base.weight.float() + self.delta_weight()
        w = w * (self.lora_magnitude / w.norm(p=2, dim=1).detach()).unsqueeze(1)
        return F.linear(x, w, self.base.bias)

    @torch.no_grad()
    def merged_linear(self):
        w = self.base.weight.float() + self.delta_weight()
        if self.use_dora:
            w = w * (self.lora_magnitude / w.norm(p=2, dim=1)).unsqueeze(1)
        b = self.base.bias
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=b is not None,
                        device=w.device, dtype=self.base.weight.dtype)
        lin.weight.copy_(w.to(self.base.weight.dtype))
        if b is not None:
            lin.bias.copy_(b)
        lin.requires_grad_(False)
        return lin


def lora_blocks(backend):
    """The transformer blocks LoRA is attached to (and checkpointed at)."""
    v = backend.visual
    if backend.name == "biomedclip":
        return list(v.trunk.blocks)
    if backend.name == "medsiglip":
        return list(v.encoder.layers)
    if backend.name == "raddino":
        return list(v.encoder.layer)
    if backend.name == "medimageinsight":
        return [blk for stage in v.image_encoder.blocks for blk in stage]
    if hasattr(backend, "lora_blocks"):
        return backend.lora_blocks()
    raise ValueError("no LoRA block mapping for backend %r" % backend.name)


def _set_child(root, qualname, new):
    parent = root.get_submodule(qualname.rsplit(".", 1)[0]) if "." in qualname else root
    setattr(parent, qualname.rsplit(".", 1)[-1], new)


def attach_lora(blocks, rank, alpha, dora=False):
    """Wrap every nn.Linear inside every block. Returns the number wrapped."""
    n = 0
    for blk in blocks:
        targets = [(name, m) for name, m in blk.named_modules()
                   if isinstance(m, nn.Linear) and name]
        for name, m in targets:
            _set_child(blk, name, LoRALinear(m, rank, alpha, dora))
            n += 1
    return n


def merge_lora(root):
    """Replace every LoRALinear under root by its merged nn.Linear."""
    targets = [(name, m) for name, m in root.named_modules() if isinstance(m, LoRALinear)]
    for name, m in targets:
        _set_child(root, name, m.merged_linear())
    return len(targets)


def lora_state_dict(root):
    return {k: v.detach().cpu().clone() for k, v in root.state_dict().items() if ".lora_" in k}


def load_lora_state_dict(root, sd):
    own = {k for k in root.state_dict() if ".lora_" in k}
    missing, unexpected = own - set(sd), set(sd) - own
    if missing or unexpected:
        raise KeyError("LoRA state mismatch: %d missing (e.g. %s), %d unexpected (e.g. %s)"
                       % (len(missing), sorted(missing)[:2], len(unexpected), sorted(unexpected)[:2]))
    root.load_state_dict(sd, strict=False)


def enable_block_checkpointing(blocks):
    """Activation checkpointing per block, independent of train/eval mode
    (HF only checkpoints when module.training is True, but the backbone stays
    in eval mode here). Non-reentrant, so LoRA weights inside get gradients."""
    for blk in blocks:
        orig = blk.forward

        def fwd(*args, _orig=orig, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _orig(*args, **kwargs)

        blk.forward = fwd
