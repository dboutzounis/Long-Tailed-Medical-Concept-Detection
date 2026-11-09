#!/usr/bin/env python
"""Figures 5.6-5.9 from the outputs of run_gradeclip.py (no model, no GPU needed).

Map grids show one row per (image, concept) with one overlay per method; curves show the mean
deletion and insertion curves per method. Pairs are chosen by fixed rules: --select tp
(correct detections), fn (near misses) or random, ordered by probe probability.

Usage:
    python plot_gradeclip.py --out-dir RESULTS_DIR
    python plot_gradeclip.py --out-dir RESULTS_DIR --select fn --n 8 --tag misses
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SAL_CMAP = LinearSegmentedColormap.from_list("sal_blue", ["#08306b", "#0057ff", "#00c2ff"])
MAX_ALPHA = 0.85

# the 4 modality CUIs (common.DOMINANT_CONCEPTS): CT, X-ray, MRI, ultrasound.
MODALITY_CUIS = {"C0040405", "C1306645", "C0024485", "C0041618"}

METHOD_LABELS = {
    "gradeclip__lin": "Grad-ECLIP (probe)",
    "gradeclip_noksim__lin": "Grad-ECLIP w/o spatial w. (probe)",
    "gradcam__lin": "Grad-CAM (probe)",
    "gradeclip__zs": "Grad-ECLIP (zero-shot)",
    "gradcam__zs": "Grad-CAM (zero-shot)",
    "attn": "Last-layer attention",
    "random": "Random",
}
# curve styles: blue shades + line styles so methods stay distinguishable in print
CURVE_STYLE = {
    "gradeclip": dict(color="#08306b", ls="-", lw=2.2),
    "gradeclip_noksim": dict(color="#2171b5", ls="--", lw=1.8),
    "gradcam": dict(color="#4292c6", ls="-.", lw=1.8),
    "attn": dict(color="#6baed6", ls=(0, (5, 2, 1, 2)), lw=1.8),
    "random": dict(color="#9ecae1", ls=":", lw=2.0),
}


def load(out_dir):
    with open(os.path.join(out_dir, "pairs.csv")) as f:
        pairs = list(csv.DictReader(f))
    maps = dict(np.load(os.path.join(out_dir, "maps.npz")))
    im = np.load(os.path.join(out_dir, "images.npz"))
    images = {str(k): i for i, k in enumerate(im["keys"])}
    cv = np.load(os.path.join(out_dir, "curves.npz"))
    for name, m in maps.items():
        if m.shape[0] != len(pairs):
            raise SystemExit("maps[%s] has %d rows, pairs.csv %d" % (name, m.shape[0], len(pairs)))
    return pairs, maps, im["images"], images, cv


def select_pairs(pairs, args):
    idx = list(range(len(pairs)))
    if args.cuis:
        idx = [i for i in idx if pairs[i]["cui"] in set(args.cuis)]
    if args.pairs:
        want = [tuple(p.rsplit(":", 1)) for p in args.pairs]
        lookup = {(pairs[i]["key"], pairs[i]["cui"]): i for i in idx}
        missing = [w for w in want if w not in lookup]
        if missing:
            raise SystemExit("pairs not in results: %s" % missing)
        return [lookup[w] for w in want]
    if not args.include_modality:
        idx = [i for i in idx if pairs[i]["cui"] not in MODALITY_CUIS]
    if args.min_train is not None:
        idx = [i for i in idx if int(pairs[i]["train_count"]) >= args.min_train]
    if args.max_train is not None:
        idx = [i for i in idx if int(pairs[i]["train_count"]) <= args.max_train]
    prob = lambda i: float(pairs[i]["lin_prob"])
    if args.select == "tp":
        idx = sorted([i for i in idx if pairs[i]["lin_pred"] == "1"], key=prob, reverse=True)
    elif args.select == "fn":
        idx = sorted([i for i in idx if pairs[i]["lin_pred"] == "0"], key=prob, reverse=True)
    elif args.select == "random":
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(idx).tolist()
    out, seen, per_cui = [], set(), {}
    for i in idx:
        k, c = pairs[i]["key"], pairs[i]["cui"]
        if k in seen or per_cui.get(c, 0) >= args.max_per_cui:
            continue
        seen.add(k)
        per_cui[c] = per_cui.get(c, 0) + 1
        out.append(i)
        if len(out) == args.n:
            break
    if not out:
        raise SystemExit("no pairs match the selection")
    return out


def upsample(m, size):
    """Bilinear upsampling of a (gh, gw) map to (size, size), align_corners=False."""
    gh, gw = m.shape
    ys = (np.arange(size) + 0.5) * gh / size - 0.5
    xs = (np.arange(size) + 0.5) * gw / size - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, gh - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, gw - 1)
    y1, x1 = np.clip(y0 + 1, 0, gh - 1), np.clip(x0 + 1, 0, gw - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]
    a, b = m[y0][:, x0], m[y0][:, x1]
    c, d = m[y1][:, x0], m[y1][:, x1]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


def overlay(ax, img, m, style="paper"):
    u = upsample(m.astype(np.float64), img.shape[0])
    rng = u.max() - u.min()
    if style == "paper":
        if rng > 0:
            u = (u - u.min()) / rng
        else:
            u = np.zeros_like(u)
        color = plt.get_cmap("jet")(u)[..., :3]
        blend = np.clip(0.5 * img.astype(np.float64) / 255.0 + 0.5 * color, 0, 1)
        ax.imshow(blend, interpolation="nearest")
    else:
        gray = img.astype(np.float64).mean(-1) / 255.0
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if rng > 0 and style != "paper":
        u = (u - u.min()) / rng
        rgba = SAL_CMAP(u)
        rgba[..., 3] = MAX_ALPHA * u
        ax.imshow(rgba, interpolation="nearest")
    elif rng <= 0:
        ax.text(0.5, 0.5, "all-zero map", transform=ax.transAxes, ha="center",
                va="center", color="white", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_grid(pairs, maps, images, img_index, sel, methods, path, style="paper"):
    n, k = len(sel), len(methods) + 1
    fig, axes = plt.subplots(n, k, figsize=(2.1 * k, 2.25 * n), squeeze=False)
    for r, i in enumerate(sel):
        p = pairs[i]
        img = images[img_index[p["key"]]]
        ax = axes[r, 0]
        ax.imshow(img.astype(np.float64).mean(-1) / 255.0, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        name = p["name"] if len(p["name"]) <= 28 else p["name"][:26] + "..."
        ax.set_ylabel("%s\n%s\np=%.2f  cos=%.3f" % (name, p["cui"], float(p["lin_prob"]),
                                                   float(p["zs_cos"])), fontsize=7)
        for c, mname in enumerate(methods, start=1):
            overlay(axes[r, c], img, maps[mname][i], style)
            if r == 0:
                axes[r, c].set_title(METHOD_LABELS.get(mname, mname), fontsize=8)
    if n:
        axes[0, 0].set_title("Input", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path + "." + ext, dpi=200, bbox_inches="tight")
    plt.close(fig)


def trapz_auc(c):
    """Trapezoid AUC along the LAST axis (steps); c is (..., n_steps+1)."""
    return (c[..., :-1] + c[..., 1:]).sum(-1) * 0.5 / (c.shape[-1] - 1)


def plot_curves(cv, path):
    curves = cv["curves"]                       # pair, order, mode, step, target
    orders = [str(o) for o in cv["orders"]]
    targets = [str(t) for t in cv["targets"]]
    modes = [str(m) for m in cv["modes"]]
    x = np.linspace(0, 1, curves.shape[3])
    ylab = {"lin": "probe probability", "zs": "zero-shot cosine"}
    fig, axes = plt.subplots(len(targets), len(modes), figsize=(9.5, 7), squeeze=False)
    for ti, tgt in enumerate(targets):
        for mi, mode in enumerate(modes):
            ax = axes[ti, mi]
            for oi, name in enumerate(orders):
                base, _, explained = name.partition("__")
                if explained and explained != tgt:
                    continue
                c = curves[:, oi, mi, :, ti]
                a = trapz_auc(c).mean()
                ax.plot(x, c.mean(0), label="%s (AUC %.3f)" % (METHOD_LABELS.get(name, name)
                                                               .replace(" (probe)", "")
                                                               .replace(" (zero-shot)", ""), a),
                        **CURVE_STYLE.get(base, {}))
            ax.set_title("%s, %s target" % (mode.capitalize(),
                                            "probe" if tgt == "lin" else "zero-shot"), fontsize=10)
            ax.set_xlabel("fraction of patches %s" % ("removed" if mode == "deletion" else "inserted"))
            ax.set_ylabel("mean %s" % ylab.get(tgt, tgt))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path + "." + ext, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--select", choices=["tp", "fn", "random"], default="tp")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--cuis", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+",
                    default=["gradeclip__lin", "gradeclip__zs", "gradcam__lin", "attn"])
    ap.add_argument("--tag", default=None, help="figure file-name suffix")
    ap.add_argument("--style", choices=["paper", "blue"], default="paper",
                    help="paper: JET blended 50/50 as in the Grad-ECLIP paper; blue: previous overlay")
    ap.add_argument("--include-modality", action="store_true",
                    help="allow the 4 modality CUIs (excluded by default)")
    ap.add_argument("--max-per-cui", type=int, default=1,
                    help="max grid rows showing the same CUI")
    ap.add_argument("--min-train", type=int, default=None,
                    help="only CUIs with at least this many training images")
    ap.add_argument("--max-train", type=int, default=None,
                    help="only CUIs with at most this many training images")
    args = ap.parse_args()

    out_dir = os.path.expandvars(args.out_dir)
    pairs, maps, images, img_index, cv = load(out_dir)
    for m in args.methods:
        if m not in maps:
            raise SystemExit("unknown method %s (have: %s)" % (m, sorted(maps)))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    tag = args.tag or ("pairs" if args.pairs else args.select)
    sel = select_pairs(pairs, args)
    plot_grid(pairs, maps, images, img_index, sel, args.methods,
              os.path.join(fig_dir, "maps_%s" % tag), args.style)
    plot_curves(cv, os.path.join(fig_dir, "faithfulness_curves"))
    print("written: %s/{maps_%s,faithfulness_curves}.{pdf,png}" % (fig_dir, tag))


if __name__ == "__main__":
    main()
