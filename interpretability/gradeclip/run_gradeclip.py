#!/usr/bin/env python
"""Grad-ECLIP maps and faithfulness evaluation for frozen BiomedCLIP (Section 4.8, Table 5.18).

For a fixed random subset of test images and each of their ground-truth concepts, two scores
are explained: the zero-shot cosine with the prompt-ensemble text embedding ("zs") and the logit
of the frozen-feature linear probe ("lin"). Maps: Grad-ECLIP, Grad-ECLIP without the spatial
weight, Grad-CAM, attention and random. Faithfulness uses patch deletion / insertion against a
blurred copy of the image, compared pair by pair with the random map; confidence intervals come
from a bootstrap over images. A few sanity checks are written to sanity.json.

Usage:
    python run_gradeclip.py --config config.yaml
    python run_gradeclip.py --config config.yaml --summarize-only
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA init

import argparse
import copy
import csv
import json
import pickle
import random
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
VLM_COMMON = os.path.join(HERE, "..", "..", "vlm", "common")
sys.path.insert(0, HERE)
from core import (TimmViTTap, auc, denormalize_to_uint8, find_normalize,  # noqa: E402
                  gaussian_blur, grad_cam_map, grad_eclip_map, patch_order,
                  perturbation_batch, random_map)

# (map name, target the map explains); attn and random are target-free
ORDERS = [("gradeclip", "lin"), ("gradeclip_noksim", "lin"), ("gradcam", "lin"),
          ("gradeclip", "zs"), ("gradcam", "zs"), ("attn", None), ("random", None)]
TARGETS = ["lin", "zs"]          # index 0 / 1 in the last axis of the curves
MODES = ["deletion", "insertion"]


def order_name(m, t):
    return m if t is None else "%s__%s" % (m, t)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def load_yaml(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    def expand(o):
        if isinstance(o, str):
            return os.path.abspath(os.path.expandvars(o)) if ("/" in o or "$" in o) else o
        if isinstance(o, dict):
            return {k: expand(v) for k, v in o.items()}
        if isinstance(o, list):
            return [expand(v) for v in o]
        return o
    cfg["paths"] = expand(cfg["paths"])
    return cfg


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False   # fp32 throughout: gradients
    torch.backends.cudnn.allow_tf32 = False


def import_common():
    """Import backends.py, common.py and data.py from vlm/common/."""
    sys.path.insert(0, VLM_COMMON)
    import backends
    import common
    import data
    return backends, common, data

def load_head(head_dir, encoder):
    with open(os.path.join(head_dir, "config.json")) as f:
        hcfg = json.load(f)
    if hcfg.get("encoder") != encoder or hcfg.get("head") != "linear":
        raise SystemExit("%s is encoder=%s head=%s, expected %s / linear"
                         % (head_dir, hcfg.get("encoder"), hcfg.get("head"), encoder))
    with open(os.path.join(head_dir, "results.json")) as f:
        res = json.load(f)
    sd = torch.load(os.path.join(head_dir, "head.pt"), map_location="cpu", weights_only=True)
    W, b = sd["fc.weight"].float(), sd["fc.bias"].float()
    return W, b, float(res["threshold"]), res


def load_text_embs(path):
    meta_path = path[:-3] + "_meta.json"
    if not os.path.exists(path):
        raise SystemExit("%s missing: run extract_text_embs.py first" % path)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get("all_checks_passed", False):
            raise SystemExit("%s: alignment checks did not pass" % meta_path)
    return F.normalize(torch.load(path, map_location="cpu", weights_only=True).float(), dim=-1)


def stochastic_modules(module):
    bad = []
    for name, m in module.named_modules():
        if m.training:
            bad.append(name or "<root>")
    return bad


# ---------------------------------------------------------------------------
# Per-image explanation
# ---------------------------------------------------------------------------
def forward_scores(backend, x, T, W, b):
    """x: (1, 3, H, W) on device. Returns (zs (C,), lin (C,)) with graph."""
    f = backend.encode_image(x).float()
    fn = F.normalize(f, dim=-1)
    return (fn @ T.t())[0], (fn @ W.t() + b)[0], fn.detach()[0]


def explain_image(backend, tap, x, cuis, T, W, b, seed_base, with_maps=None):
    """All maps for one image and a list of CUI indices.
    Returns dict order_name -> (n_cuis, P) tensor, plus per-CUI scores."""
    x = x.clone().requires_grad_(True)       # builds a graph through the frozen model
    zs, lin, fn = forward_scores(backend, x, T, W, b)
    if tap.attn_out is None or tap.qkv is None or tap.block_in is None:
        raise RuntimeError("hooks did not fire")
    out = {order_name(m, t): [] for m, t in ORDERS}
    attn = tap.cls_attention()[0]
    P = attn.numel()
    for c in cuis:
        s = {"lin": lin[c], "zs": zs[c]}
        out["gradeclip__lin"].append(grad_eclip_map(s["lin"], tap, with_ksim=True))
        out["gradeclip_noksim__lin"].append(grad_eclip_map(s["lin"], tap, with_ksim=False))
        out["gradcam__lin"].append(grad_cam_map(s["lin"], tap))
        out["gradeclip__zs"].append(grad_eclip_map(s["zs"], tap, with_ksim=True))
        out["gradcam__zs"].append(grad_cam_map(s["zs"], tap))
        out["attn"].append(attn)
        out["random"].append(random_map(P, zlib.crc32(("%s|%d" % (seed_base, c)).encode())))
    maps = {k: torch.stack([m.float().cpu() for m in v]) for k, v in out.items()}
    scores = {"zs": zs.detach()[cuis].cpu(), "lin": lin.detach()[cuis].cpu()}
    return maps, scores, fn


@torch.no_grad()
def perturbed_scores(backend, batch, T, W, b, chunk):
    """(n, 2) scores [lin prob, zs cosine] for a batch of perturbed images."""
    outs = []
    for s in range(0, batch.shape[0], chunk):
        f = F.normalize(backend.encode_image(batch[s:s + chunk]).float(), dim=-1)
        outs.append(torch.stack([torch.sigmoid(f @ W.t() + b), f @ T.t()], dim=-1))
    return torch.cat(outs)          # (n, C, 2)


@torch.no_grad()
def faithfulness(backend, tap, x, maps, cuis, T, W, b, fcfg):
    """curves[pair, order, mode, step, target] for one image."""
    n_steps = int(fcfg["n_steps"])
    x_base = gaussian_blur(x, float(fcfg["blur_sigma"]))
    names = [order_name(m, t) for m, t in ORDERS]
    curves = np.zeros((len(cuis), len(names), len(MODES), n_steps + 1, len(TARGETS)),
                      dtype=np.float32)
    cidx = torch.as_tensor(cuis, device=x.device)
    for j in range(len(cuis)):
        batches = []
        for name in names:
            order = patch_order(maps[name][j])
            for mode in MODES:
                batches.append(perturbation_batch(x[0], x_base[0], order, tap.grid,
                                                  tap.patch, n_steps, mode))
        sc = perturbed_scores(backend, torch.cat(batches), T, W, b, int(fcfg["batch_size"]))
        sc = sc[:, cidx[j], :].view(len(names), len(MODES), n_steps + 1, 2)
        curves[j] = sc.cpu().numpy()
    return curves


# ---------------------------------------------------------------------------
# Randomisation test
# ---------------------------------------------------------------------------
def disable_fused_attention(module):
    n = 0
    for m in module.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
            n += 1
    return n


def randomized_copy(visual, block_idx, seed):
    """Deep copy of the image tower with the target block re-initialised
    (Linear: trunc-normal std 0.02, bias 0; LayerNorm: weight 1, bias 0)."""
    v = copy.deepcopy(visual)
    g = torch.Generator(device="cpu").manual_seed(seed)
    blk = v.trunk.blocks[block_idx]
    with torch.no_grad():
        for m in blk.modules():
            if isinstance(m, nn.Linear):
                w = torch.empty(m.weight.shape).normal_(0, 0.02, generator=g).clamp_(-0.04, 0.04)
                m.weight.copy_(w.to(m.weight))
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
                m.bias.zero_()
            elif hasattr(m, "gamma") and isinstance(getattr(m, "gamma"), torch.Tensor):
                m.gamma.fill_(1.0)          # LayerScale, if present
    return v


class _VisualOnly:
    """encode_image() over a given image tower (for the randomised copy)."""

    def __init__(self, visual):
        self.visual = visual

    def encode_image(self, x):
        return self.visual(x)


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(a), np.asarray(b)).correlation
    return float(r) if np.isfinite(r) else float("nan")


# ---------------------------------------------------------------------------
# Summary (cluster bootstrap over images)
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(values, groups, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(out_dir, n_boot=2000):
    z = np.load(os.path.join(out_dir, "curves.npz"), allow_pickle=False)
    curves, names = z["curves"], [str(s) for s in z["orders"]]
    with open(os.path.join(out_dir, "pairs.csv")) as f:
        pairs = list(csv.DictReader(f))
    groups = np.array([p["key"] for p in pairs])
    rows = []
    for oi, name in enumerate(names):
        explained = name.split("__")[1] if "__" in name else None
        for ti, tgt in enumerate(TARGETS):
            if explained is not None and explained != tgt:
                continue      # a target-specific map is evaluated on its own target
            d = np.array([auc(c) for c in curves[:, oi, 0, :, ti]])
            i = np.array([auc(c) for c in curves[:, oi, 1, :, ti]])
            rnd = names.index("random")
            dr = np.array([auc(c) for c in curves[:, rnd, 0, :, ti]])
            ir = np.array([auc(c) for c in curves[:, rnd, 1, :, ti]])
            row = {"map": name, "eval_target": tgt, "n_pairs": len(d),
                   "n_images": len(np.unique(groups))}
            for lab, vals in (("deletion_auc", d), ("insertion_auc", i),
                              ("ins_minus_del", i - d),
                              ("deletion_vs_random", d - dr),
                              ("insertion_vs_random", i - ir)):
                m, lo, hi = cluster_bootstrap_ci(vals, groups, n_boot)
                row[lab] = m
                row[lab + "_ci_lo"] = lo
                row[lab + "_ci_hi"] = hi
            rows.append(row)
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%-24s %-4s %22s %22s" % ("map", "tgt", "deletion AUC (lower)", "insertion AUC (higher)"))
    for r in rows:
        print("%-24s %-4s %8.4f [%.4f, %.4f] %8.4f [%.4f, %.4f]"
              % (r["map"], r["eval_target"], r["deletion_auc"], r["deletion_auc_ci_lo"],
                 r["deletion_auc_ci_hi"], r["insertion_auc"], r["insertion_auc_ci_lo"],
                 r["insertion_auc_ci_hi"]))
    print("written: %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(os.path.abspath(args.config))
    P_ = cfg["paths"]
    enc = cfg["encoder"]
    out_dir = P_["out_dir"]
    if args.summarize_only:
        summarize(out_dir, int(cfg["summary"]["n_boot"]))
        return
    if os.path.exists(os.path.join(out_dir, "curves.npz")) and not args.overwrite:
        raise SystemExit("%s already has results (use --overwrite)" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    seed = int(cfg["subset"]["seed"])
    set_determinism(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_dir = os.path.join(P_["features_root"], "%s_features" % enc)
    W, b, thr, head_res = load_head(P_["head_dir"], enc)
    T = load_text_embs(P_["text_embs"])
    with open(os.path.join(feat_dir, "test_keys.pkl"), "rb") as f:
        test_keys = list(pickle.load(f))
    cached = F.normalize(torch.load(os.path.join(feat_dir, "test_embs.pt"),
                                    map_location="cpu", weights_only=False).float(), dim=-1)
    train_counts = torch.load(os.path.join(feat_dir, "train_labels.pt"),
                              map_location="cpu", weights_only=False).float().sum(0).long()

    backends, common, data = import_common()
    lcfg = common.load_config(os.path.join(VLM_COMMON, "config.yaml"))
    mlb = common.load_mlb(lcfg["paths"]["mlb"])
    cui_order = list(mlb.classes_)
    names = common.load_cui_names(lcfg["paths"].get("cui_mapping"), cui_order)
    C = len(cui_order)
    if W.shape[0] != C or T.shape[0] != C or train_counts.numel() != C:
        raise SystemExit("C mismatch: head %d, text %d, train labels %d, mlb %d"
                         % (W.shape[0], T.shape[0], train_counts.numel(), C))
    if W.shape[1] != T.shape[1] or W.shape[1] != cached.shape[1]:
        raise SystemExit("embedding dims differ: head %d, text %d, cached %d"
                         % (W.shape[1], T.shape[1], cached.shape[1]))

    only = cfg["subset"].get("cuis")
    only_idx = None
    if only:
        pos = {c: i for i, c in enumerate(cui_order)}
        missing = [c for c in only if c not in pos]
        if missing:
            raise SystemExit("CUIs not in mlb: %s" % missing)
        only_idx = set(pos[c] for c in only)

    # model
    backend = backends.build_backend(enc, lcfg["paths"].get("hf_cache"),
                                     lcfg.get("models", {}).get(enc, {}))
    backend.drop_text()
    visual = backend.visual.to(device).eval()
    for p in visual.parameters():
        p.requires_grad_(False)
    n_fused = disable_fused_attention(visual)
    print("fused (SDPA) attention disabled in %d timm attention modules" % n_fused)
    active = stochastic_modules(visual)
    assert not active, "modules in training mode: %s" % active[:5]
    block_idx = int(cfg["method"]["block_idx"])
    tap = TimmViTTap(visual.trunk, block_idx)
    mean, std = find_normalize(backend.eval_transform)
    W, b, T = W.to(device), b.to(device), T.to(device)
    print("encoder %s | grid %s patch %s heads %d | head thr %.2f (test F1 %.4f)"
          % (enc, tap.grid, tap.patch, tap.num_heads, thr, head_res["test_f1"]))

    # image subset: fixed random sample of test keys
    rng = np.random.default_rng(seed)
    n_img = min(int(cfg["subset"]["n_images"]), len(test_keys))
    pick = sorted(rng.choice(len(test_keys), size=n_img, replace=False).tolist())
    wanted = {test_keys[i]: i for i in pick}
    ex = lcfg["extract"]
    loader = data.build_loader(lcfg["paths"]["test_shards"], mlb, backend,
                               int(cfg["data"]["batch_size"]), ex["num_workers"],
                               ex["prefetch_factor"])

    fcfg = cfg["faithfulness"]
    scfg = cfg["sanity"]
    pair_rows, maps_acc, curves_acc, images, image_keys = [], {}, [], [], []
    sanity = {"hook_layout_max_abs_diff": None, "cache_match_cos": [], "determinism": [],
              "randomization": {}}
    keep_for_rand = []                 # (key, x, cuis, original gradeclip__lin maps)
    t0 = time.time()
    done = 0
    for xb, yb, kb in loader:
        for x1, y1, k in zip(xb, yb, kb):
            if k not in wanted:
                continue
            cuis = torch.nonzero(y1).flatten().tolist()
            if only_idx is not None:
                cuis = [c for c in cuis if c in only_idx]
            if not cuis:
                wanted.pop(k)
                continue
            x = x1.unsqueeze(0).to(device).float()
            maps, scores, fn = explain_image(backend, tap, x, cuis, T, W, b, "%d|%s" % (seed, k))

            if sanity["hook_layout_max_abs_diff"] is None:
                d = (tap.recompute_attn_out() - tap.attn_out.detach()).abs().max().item()
                sanity["hook_layout_max_abs_diff"] = d
                print("hook layout check: max |recomputed - captured| = %.3g" % d)
                if d > 1e-3:
                    raise SystemExit("captured attention output does not match q,k,v layout")
            sanity["cache_match_cos"].append(float(fn.cpu() @ cached[wanted[k]]))
            if len(sanity["determinism"]) < int(scfg["n_determinism"]):
                again, _, _ = explain_image(backend, tap, x, cuis[:1], T, W, b, "det")
                a, r = maps["gradeclip__lin"][0], again["gradeclip__lin"][0]
                sanity["determinism"].append(
                    float((a - r).abs().max() / a.abs().max().clamp(min=1e-12)))
            if len(keep_for_rand) < int(scfg["n_randomization_images"]):
                keep_for_rand.append((k, x.cpu(), cuis, maps["gradeclip__lin"].clone(),
                                      maps["gradeclip__zs"].clone()))

            curves = faithfulness(backend, tap, x, maps, cuis, T, W, b, fcfg)
            curves_acc.append(curves)
            for name, m in maps.items():
                maps_acc.setdefault(name, []).append(m.view(len(cuis), *tap.grid).numpy())
            images.append(denormalize_to_uint8(x[0], mean, std))
            image_keys.append(k)
            prob = torch.sigmoid(scores["lin"])
            for j, c in enumerate(cuis):
                pair_rows.append({
                    "key": k, "cui": cui_order[c], "name": names[c],
                    "train_count": int(train_counts[c]), "n_gt": int(y1.sum()),
                    "zs_cos": "%.6f" % float(scores["zs"][j]),
                    "lin_logit": "%.6f" % float(scores["lin"][j]),
                    "lin_prob": "%.6f" % float(prob[j]),
                    "lin_pred": int(float(prob[j]) >= thr)})
            wanted.pop(k)
            done += 1
            if done % 25 == 0:
                print("%d/%d images | %d pairs | %.0fs" % (done, n_img, len(pair_rows),
                                                            time.time() - t0))
        if not wanted:
            break
    if wanted:
        print("WARNING: %d selected keys not found in the test shards" % len(wanted))
    tap.remove()

    # randomisation test on the kept images
    if keep_for_rand:
        rvis = randomized_copy(visual, block_idx, int(scfg["randomization_seed"])).to(device).eval()
        rtap = TimmViTTap(rvis.trunk, block_idx)
        rb = _VisualOnly(rvis)
        rho = {"gradeclip__lin": [], "gradeclip__zs": []}
        for k, x, cuis, m_lin, m_zs in keep_for_rand:
            rm, _, _ = explain_image(rb, rtap, x.to(device), cuis, T, W, b, "rand")
            for j in range(len(cuis)):
                rho["gradeclip__lin"].append(spearman(m_lin[j], rm["gradeclip__lin"][j]))
                rho["gradeclip__zs"].append(spearman(m_zs[j], rm["gradeclip__zs"][j]))
        rtap.remove()
        for name, v in rho.items():
            v = np.array(v, dtype=np.float64)
            sanity["randomization"][name] = {
                "n_pairs": int(np.isfinite(v).sum()),
                "spearman_mean": float(np.nanmean(v)), "spearman_median": float(np.nanmedian(v)),
                "abs_spearman_mean": float(np.nanmean(np.abs(v)))}
        del rvis

    # save
    cm = np.array(sanity["cache_match_cos"])
    sanity["cache_match_cos"] = {"min": float(cm.min()), "mean": float(cm.mean()),
                                 "n": int(cm.size)}
    det = np.array(sanity["determinism"])
    sanity["determinism"] = {"max_rel_diff": float(det.max()) if det.size else None,
                             "n": int(det.size)}
    sanity["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                     "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "sanity.json"), "w") as f:
        json.dump(sanity, f, indent=2)
    if cm.min() < 0.99:
        print("WARNING: fresh embeddings differ from cached features (min cos %.4f): "
              "preprocessing or precision differs from extract_features.py" % cm.min())

    with open(os.path.join(out_dir, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    np.savez_compressed(os.path.join(out_dir, "maps.npz"),
                        **{k: np.concatenate(v) for k, v in maps_acc.items()})
    np.savez_compressed(os.path.join(out_dir, "curves.npz"),
                        curves=np.concatenate(curves_acc),
                        orders=np.array([order_name(m, t) for m, t in ORDERS]),
                        modes=np.array(MODES), targets=np.array(TARGETS),
                        n_steps=int(fcfg["n_steps"]))
    np.savez_compressed(os.path.join(out_dir, "images.npz"),
                        images=np.stack(images), keys=np.array(image_keys))
    print("sanity: %s" % json.dumps({k: sanity[k] for k in
                                     ("hook_layout_max_abs_diff", "cache_match_cos",
                                      "determinism", "randomization")}))
    summarize(out_dir, int(cfg["summary"]["n_boot"]))


if __name__ == "__main__":
    main()
