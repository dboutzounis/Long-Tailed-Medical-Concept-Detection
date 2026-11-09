"""Grad-ECLIP (Zhao et al., 2025; https://github.com/Cyang-Zhao/Grad-Eclip) and the baseline maps,
for the frozen ViT image encoder of BiomedCLIP.

Unlike the official code, which recomputes the last attention layer as a single head, every
quantity is read with hooks from the unmodified multi-head forward pass, so the explained score
is exactly the model's score. For a scalar score S, the map of patch i is
    ReLU(sum_c g_c * v_i,c * lam_i),
with g = dS/d(attention output of the CLS token), v_i the value vector of patch i and lam_i the
min-max normalized cosine between the projected CLS query and patch key.
Baselines: Grad-CAM on the input of the last block, CLS attention (mean over heads), random.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hooks on the last block of a timm VisionTransformer
# ---------------------------------------------------------------------------

class TimmViTTap:

    def __init__(self, trunk, block_idx=-1):
        self.trunk = trunk
        self.block = trunk.blocks[block_idx]
        self.attn = self.block.attn
        self.num_heads = self.attn.num_heads
        self.n_prefix = int(getattr(trunk, "num_prefix_tokens", 1))
        self.grid = tuple(trunk.patch_embed.grid_size)
        self.patch = tuple(trunk.patch_embed.patch_size)
        self._check_supported()
        self.block_in = self.qkv = self.attn_out = None
        self._handles = [
            self.block.register_forward_pre_hook(self._h_block_in),
            self.attn.qkv.register_forward_hook(self._h_qkv),
            self.attn.proj.register_forward_pre_hook(self._h_attn_out),
        ]

    def _check_supported(self):
        for name in ("q_norm", "k_norm", "norm"):
            m = getattr(self.attn, name, None)
            if m is not None and not isinstance(m, nn.Identity):
                raise NotImplementedError("attn.%s is %s (expected Identity)"
                                          % (name, type(m).__name__))
        if getattr(self.attn, "gate", None) is not None:
            raise NotImplementedError("gated attention: proj input != attention output")
        gp = getattr(self.trunk, "global_pool", "token")
        if gp != "token":
            raise NotImplementedError("trunk.global_pool=%r: map assumes CLS pooling" % gp)
        if getattr(self.trunk, "attn_pool", None) is not None:
            raise NotImplementedError("trunk has an attention-pooling head")
        if self.n_prefix != 1:
            raise NotImplementedError("num_prefix_tokens=%d (register tokens?)" % self.n_prefix)

    def _h_block_in(self, module, args):
        self.block_in = args[0]

    def _h_qkv(self, module, args, out):
        self.qkv = out

    def _h_attn_out(self, module, args):
        self.attn_out = args[0]

    def remove(self):
        for h in self._handles:
            h.remove()

    # derived quantities (detached)
    def split_qkv(self):
        B, N, C3 = self.qkv.shape
        C = C3 // 3
        qkv = self.qkv.detach().reshape(B, N, 3, C)
        return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    @torch.no_grad()
    def recompute_attn_out(self):
        """Attention output from the captured q, k, v -- must match attn_out."""
        q, k, v = self.split_qkv()
        B, N, C = q.shape
        H = self.num_heads
        hd = C // H
        sh = lambda t: t.reshape(B, N, H, hd).transpose(1, 2)
        a = torch.softmax((sh(q) * hd ** -0.5) @ sh(k).transpose(-2, -1), dim=-1)
        return (a @ sh(v)).transpose(1, 2).reshape(B, N, C)

    @torch.no_grad()
    def cls_attention(self):
        """(B, P) softmax attention CLS -> patches, averaged over heads."""
        q, k, _ = self.split_qkv()
        B, N, C = q.shape
        H = self.num_heads
        hd = C // H
        qc = q[:, :1].reshape(B, 1, H, hd).transpose(1, 2)
        kk = k.reshape(B, N, H, hd).transpose(1, 2)
        a = torch.softmax((qc * hd ** -0.5) @ kk.transpose(-2, -1), dim=-1)  # B,H,1,N
        return a[:, :, 0, self.n_prefix:].mean(1)


# ---------------------------------------------------------------------------
# Explanation maps (one image, one scalar score; graph must be retained)
# ---------------------------------------------------------------------------

def grad_eclip_map(score, tap, with_ksim=True):
    """(P,) Grad-ECLIP map for batch element 0."""
    g = torch.autograd.grad(score, tap.attn_out, retain_graph=True)[0].detach()
    g_cls = g[0, 0]                                        # (C,)
    q, k, v = tap.split_qkv()
    p = tap.n_prefix
    contrib = g_cls[None, :] * v[0, p:]                    # (P, C)
    if with_ksim:
        W, b = tap.attn.proj.weight.detach(), tap.attn.proj.bias
        b = None if b is None else b.detach()
        q_out = F.linear(q[0, :1], W, b)
        k_out = F.linear(k[0, p:], W, b)
        lam = (F.normalize(q_out, dim=-1) * F.normalize(k_out, dim=-1)).sum(-1)
        lam = (lam - lam.min()) / (lam.max() - lam.min()).clamp(min=1e-12)
        contrib = contrib * lam[:, None]
    return F.relu(contrib.sum(-1))


def grad_cam_map(score, tap):
    """(P,) Grad-CAM on the input tokens of the last block (official baseline)."""
    x = tap.block_in
    g = torch.autograd.grad(score, x, retain_graph=True)[0].detach()
    w = g[0].mean(0, keepdim=True)                          # mean over ALL tokens
    return F.relu((w * x.detach()[0, tap.n_prefix:]).sum(-1))


def random_map(n, seed):
    g = torch.Generator().manual_seed(int(seed))
    return torch.rand(n, generator=g)


# ---------------------------------------------------------------------------
# Deletion / insertion (patch level, blurred-image baseline)
# ---------------------------------------------------------------------------

def gaussian_blur(x, sigma):
    """Separable Gaussian blur of a (B, 3, H, W) tensor. Linear and with
    unit-sum kernel, so blurring the normalised tensor == normalising the
    blurred image."""
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def patch_order(emap):
    """Patch indices by descending saliency; ties broken by index (stable)."""
    e = emap.detach().double().cpu().numpy()
    return np.argsort(-e, kind="stable")


def perturbation_batch(x, x_base, order, grid, patch, n_steps, mode):
    """(n_steps+1, 3, H, W): step s has the top round(s*P/n_steps) patches
    removed (deletion: replaced by x_base) or inserted (insertion: x placed
    onto x_base)."""
    gh, gw = grid
    ph, pw = patch
    P = gh * gw
    counts = [int(round(s * P / n_steps)) for s in range(n_steps + 1)]
    masks = torch.zeros(n_steps + 1, P, device=x.device, dtype=x.dtype)
    for s, m in enumerate(counts):
        masks[s, torch.as_tensor(order[:m], device=x.device)] = 1.0
    m = masks.view(-1, 1, gh, gw)
    m = m.repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)
    H, W = x.shape[-2:]
    if m.shape[-2:] != (H, W):
        raise ValueError("patch grid %s x patch %s does not tile the %dx%d input"
                         % (grid, patch, H, W))
    if mode == "deletion":
        return x * (1 - m) + x_base * m
    if mode == "insertion":
        return x_base * (1 - m) + x * m
    raise ValueError(mode)


def auc(curve):
    """Trapezoidal area under a curve sampled at fractions 0, 1/n, ..., 1."""
    c = np.asarray(curve, dtype=np.float64)
    # explicit trapezoid rule (np.trapz was removed in NumPy 2.0)
    return float((c[:-1] + c[1:]).sum() * 0.5 / (len(c) - 1))


def denormalize_to_uint8(x, mean, std):
    """(3, H, W) normalised tensor -> (H, W, 3) uint8: what the model saw."""
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    img = (x * s + m).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def find_normalize(transform):
    """mean/std of the Normalize step inside a torchvision Compose."""
    ts = getattr(transform, "transforms", [])
    for t in ts:
        if type(t).__name__ == "Normalize":
            return list(t.mean), list(t.std)
    raise ValueError("no Normalize in eval_transform: %r" % (transform,))
