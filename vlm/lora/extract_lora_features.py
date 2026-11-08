#!/usr/bin/env python
"""Extract the embeddings of a LoRA-adapted encoder (LoRA merged into the weights) in the same
format and row order as the frozen features, so the trained heads and KNN run on them unchanged.

Before writing, checks that merged and unmerged LoRA agree and that encoding is deterministic.
--frozen skips LoRA and, with --compare-to, checks that the stored frozen features are reproduced.

Usage:
    python extract_lora_features.py --config config.yaml --model medsiglip
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import pickle

import torch
import torch.nn.functional as F

from lora_common import (EncoderClassifier, build_backend, import_common, load_config,
                         make_loader, set_determinism)
from lora_modules import attach_lora, load_lora_state_dict, lora_blocks, merge_lora
from train_lora import run_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True,
                    choices=["biomedclip", "medsiglip", "medimageinsight", "raddino"])
    ap.add_argument("--tag", default="", help="run-directory suffix used at training")
    ap.add_argument("--run-dir", default=None, help="default: <out_root>/<run name>")
    ap.add_argument("--frozen", action="store_true", help="no LoRA (pipeline check)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out-root", default=None,
                    help="default: <features_root> (LoRA) or <features_root>/frozen_check")
    ap.add_argument("--compare-to", default=None,
                    help="stored features dir to compare the test split against (by key)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    P, T, L, E = cfg["paths"], cfg["training"], cfg["lora"], cfg["extract"]
    M = cfg["models"][args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(int(T["seed"]), T.get("deterministic", True), T.get("deterministic_attention", True))
    backends_mod, common_mod = import_common()
    classes = list(common_mod.load_mlb(P["mlb"]).classes_)
    backend = build_backend(backends_mod, args.model, cfg)
    amp = T.get("precision", "bf16") == "bf16" and device.type == "cuda"
    chunk = M["eval_chunk"]

    def loader(split):
        return make_loader(P[split + "_shards"], classes, backend.eval_transform,
                           common_mod.parse_cuis, E["batch_size"], E["num_workers"],
                           E["prefetch_factor"], train=False)

    def make_core():
        core = EncoderClassifier(backend, torch.nn.Identity(), amp).to(device)
        return core.eval()

    @torch.no_grad()
    def embed(core, x):
        return torch.cat([core.embed(x[s:s + chunk].to(device)) for s in range(0, x.shape[0], chunk)])

    probe_x = next(iter(loader("val")))[0][:16]
    meta = {"model": args.model, "frozen": args.frozen}

    if not args.frozen:
        run_dir = args.run_dir or os.path.join(P["out_root"], run_name(args.model, L, args.tag))
        ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
        lc = ck["lora_cfg"]
        blocks = lora_blocks(backend)
        attach_lora(blocks, lc["rank"], lc["alpha"], lc["dora"])
        load_lora_state_dict(backend.visual, ck["lora"])
        backend.visual.to(device).eval()
        core = make_core()

        def embed_both(c):
            c.amp = False
            e32 = embed(c, probe_x)
            c.amp = amp
            return e32, embed(c, probe_x)

        u32, u16 = embed_both(core)
        n_merged = merge_lora(backend.visual)
        backend.visual.to(device).eval()
        m32, m16 = embed_both(core)
        cos32 = (u32 * m32).sum(1)
        cos16 = (u16 * m16).sum(1)
        maxdiff32 = (u32 - m32).abs().max().item()
        print("[merge] %d LoRA linears merged | fp32: cos min %.8f, max |diff| %.2e | "
              "bf16 (informational): cos min %.6f"
              % (n_merged, cos32.min(), maxdiff32, cos16.min()))
        if cos32.min() < 0.99999:
            raise SystemExit("merge check failed in fp32: merged weights do not reproduce "
                             "LoRA outputs (this is a real bug, not bf16 rounding)")
        meta.update({"run_dir": run_dir, "best_epoch": ck["epoch"], "lora_cfg": lc,
                     "merge_cos_min_fp32": cos32.min().item(),
                     "merge_max_abs_diff_fp32": maxdiff32,
                     "merge_cos_min_bf16": cos16.min().item()})
    else:
        backend.visual.to(device).eval()
    core = make_core()

    a, b = embed(core, probe_x), embed(core, probe_x)
    cos = (a * b).sum(1)
    print("[determinism] repeat-encode cos min %.6f" % cos.min())
    if cos.min() < 0.9999:
        raise SystemExit("encoder is not deterministic: refusing to write features")
    meta["determinism_cos_min"] = cos.min().item()

    root = args.out_root or (os.path.join(P["features_root"], "frozen_check") if args.frozen
                             else P["features_root"])
    out_dir = os.path.join(root, args.model + "_features")
    os.makedirs(out_dir, exist_ok=True)
    for split in args.splits:
        embs, labels, keys = [], [], []
        with torch.no_grad():
            for i, (x, y, k) in enumerate(loader(split)):
                embs.append(embed(core, x).cpu())
                labels.append(y)
                keys.extend(k)
                if (i + 1) % 50 == 0:
                    print("  %s: %d images" % (split, len(keys)), flush=True)
        embs, labels = torch.cat(embs), torch.cat(labels)
        torch.save(embs, os.path.join(out_dir, split + "_embs.pt"))
        torch.save(labels, os.path.join(out_dir, split + "_labels.pt"))
        with open(os.path.join(out_dir, split + "_keys.pkl"), "wb") as f:
            pickle.dump(keys, f)
        print("saved %s: embeddings %s, labels %s" % (split, tuple(embs.shape), tuple(labels.shape)))

    if args.compare_to and "test" in args.splits:
        ref_dir = os.path.expandvars(args.compare_to)
        ref = torch.load(os.path.join(ref_dir, "test_embs.pt"), map_location="cpu").float()
        ref_y = torch.load(os.path.join(ref_dir, "test_labels.pt"), map_location="cpu")
        with open(os.path.join(ref_dir, "test_keys.pkl"), "rb") as f:
            ref_keys = list(pickle.load(f))
        pos = {k: i for i, k in enumerate(keys)}
        if set(ref_keys) != set(keys):
            raise SystemExit("test keys differ from %s" % ref_dir)
        idx = torch.tensor([pos[k] for k in ref_keys])
        e = torch.load(os.path.join(out_dir, "test_embs.pt"))[idx]
        lab = torch.load(os.path.join(out_dir, "test_labels.pt"))[idx]
        c = (F.normalize(e, dim=-1) * F.normalize(ref, dim=-1)).sum(1)
        same_order = keys == ref_keys
        labels_ok = torch.equal(lab.bool(), ref_y.bool())
        print("[compare] vs %s | cos min %.6f mean %.6f | labels identical: %s | same row order: %s"
              % (ref_dir, c.min(), c.mean(), labels_ok, same_order))
        meta.update({"compare_to": ref_dir, "compare_cos_min": c.min().item(),
                     "compare_cos_mean": c.mean().item(), "compare_labels_ok": labels_ok,
                     "compare_same_order": same_order})

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("-> %s" % out_dir)


if __name__ == "__main__":
    main()
