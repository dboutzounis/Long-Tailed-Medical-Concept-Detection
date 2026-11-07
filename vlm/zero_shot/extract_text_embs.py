#!/usr/bin/env python
"""Save the four-template prompt-ensemble text embedding of every concept, as used by
zeroshot.py and by the Grad-ECLIP zero-shot score.

Also checks, on the test split, that the cached image features and these text embeddings
reproduce the cosine scores cached by zeroshot.py (same images, labels and space).

Usage (GPU node, after zeroshot.py):
    python extract_text_embs.py --encoder biomedclip
"""
import argparse
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common")

import torch
import torch.nn.functional as F

ENCODER_DIRS = {
    "biomedclip": "biomedclip_features",
    "medsiglip": "medsiglip_features",
    "medimageinsight": "medimageinsight_features",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(COMMON, "config.yaml"))
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODER_DIRS))
    ap.add_argument("--model", default=None,
                    help="backends.py model name (default: same as --encoder)")
    ap.add_argument("--features-root",
                    default="$WORK/Long-Tailed-Medical-Concept-Detection/artifacts")
    ap.add_argument("--out-dir", default=None, help="default: <features-root>/text_embs")
    args = ap.parse_args()

    config = os.path.abspath(os.path.expandvars(args.config))
    feat_root = os.path.abspath(os.path.expandvars(args.features_root))
    out_dir = os.path.abspath(os.path.expandvars(args.out_dir or os.path.join(feat_root, "text_embs")))
    feat_dir = os.path.join(feat_root, ENCODER_DIRS[args.encoder])
    model = args.model or args.encoder
    sys.path.insert(0, COMMON)

    from backends import build_backend
    from common import apply_precision, load_config, load_cui_names, load_mlb

    cfg = load_config(config)
    cui_order = list(load_mlb(cfg["paths"]["mlb"]).classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text embeddings, exactly as zeroshot.encode_scores builds them
    torch.manual_seed(0)
    backend = build_backend(model, cfg["paths"].get("hf_cache"), cfg["models"].get(model, {}))
    apply_precision(backend, cfg)
    if not backend.has_text:
        raise SystemExit("%s has no text encoder" % model)
    with torch.no_grad():
        text = backend.text_embeddings(
            load_cui_names(cfg["paths"].get("cui_mapping"), cui_order), device)
    text = text.float().cpu()
    norms = text.norm(dim=-1)
    print("text embeddings %s | norm min %.4f max %.4f"
          % (tuple(text.shape), norms.min(), norms.max()))
    text = F.normalize(text, dim=-1)

    # check against cached image features and the zero-shot cache
    feats = torch.load(os.path.join(feat_dir, "test_embs.pt"), map_location="cpu").float()
    flabels = torch.load(os.path.join(feat_dir, "test_labels.pt"), map_location="cpu").bool()
    with open(os.path.join(feat_dir, "test_keys.pkl"), "rb") as f:
        fkeys = list(pickle.load(f))
    if text.shape != (len(cui_order), feats.shape[1]):
        raise SystemExit("text %s does not match (C=%d, d=%d of cached features)"
                         % (tuple(text.shape), len(cui_order), feats.shape[1]))
    if flabels.shape[1] != len(cui_order):
        raise SystemExit("cached labels have %d columns, mlb has %d"
                         % (flabels.shape[1], len(cui_order)))

    zs_dir = os.path.join(cfg["paths"]["out_root"], model)
    meta = {"encoder": args.encoder, "model": model, "shape": list(text.shape)}
    cache = os.path.join(zs_dir, "cosine_scores.pt")
    ok = True
    if not os.path.exists(cache):
        print("WARNING: %s not found; cannot verify alignment" % cache)
        ok = False
    else:
        z = torch.load(cache, map_location="cpu", weights_only=False)
        zkeys, zscores = list(z["test_keys"]), z["test_scores"].float()
        zlabels = z["test_labels"].bool()
        pos = {k: i for i, k in enumerate(fkeys)}
        missing = [k for k in zkeys if k not in pos]
        keys_ok = not missing and len(zkeys) == len(fkeys)
        print("[keys]   %d cached / %d zero-shot, %d missing -> %s"
              % (len(fkeys), len(zkeys), len(missing), "OK" if keys_ok else "MISMATCH"))
        if missing:
            raise SystemExit("test keys differ between cached features and zero-shot cache")
        idx = torch.tensor([pos[k] for k in zkeys])
        feats, flabels = F.normalize(feats[idx], dim=-1), flabels[idx]

        labels_ok = torch.equal(flabels, zlabels)
        print("[labels] identical after key alignment -> %s"
              % ("OK" if labels_ok else "MISMATCH (different mlb order or labels)"))

        cos = feats @ text.t()
        d = (cos - zscores).abs()
        a, b = cos.flatten(), zscores.flatten()
        a, b = a - a.mean(), b - b.mean()
        r = float((a @ b) / (a.norm() * b.norm()))
        top1 = float((cos.argmax(1) == zscores.argmax(1)).float().mean())
        space_ok = r > 0.98 and top1 > 0.90
        print("[space]  Pearson r %.5f | mean |diff| %.5f | max |diff| %.4f | "
              "top-1 agreement %.4f -> %s"
              % (r, d.mean(), d.max(), top1, "OK" if space_ok else "MISMATCH"))
        ok = keys_ok and labels_ok and space_ok
        meta.update({"keys_ok": keys_ok, "labels_ok": labels_ok, "pearson_r": r,
                     "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()),
                     "top1_agreement": top1})

    res_path = os.path.join(zs_dir, "results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            pri = json.load(f)["rules"].get("priors")
        if pri:
            meta["zs_priors_scale"] = pri["scale"]
            meta["zs_priors_test_f1"] = pri["test_f1"]
            print("[zero-shot priors] scale %s | test F1 %.4f"
                  % (pri["scale"], pri["test_f1"]))
    meta["all_checks_passed"] = ok

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, args.encoder + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if not ok:
        print("checks FAILED: text embeddings NOT saved (see %s_meta.json)" % args.encoder)
        sys.exit(2)
    out = os.path.join(out_dir, args.encoder + ".pt")
    torch.save(text, out)
    print("saved %s  (all checks passed)" % out)


if __name__ == "__main__":
    main()
