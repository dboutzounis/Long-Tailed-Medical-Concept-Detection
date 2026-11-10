"""Figures 5.2-5.3 and Table 5.16 from the µMoE ablation reports.

Writes concept_dependence.{pdf,png} (largest relative AP drop per concept),
expert_concept_heatmap.{pdf,png} (busiest experts x affected concepts) and
expert_profiles.tex for one model.

Usage:
    python plot_mumoe_concepts.py --ablation-dir ablations \
        --taxonomy ../../data/taxonomy/concept_classifications.json --model tr_n32 --out-dir figures
"""
import argparse
import collections
import csv
import glob
import json
import os
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Style: blue sequential ramp for magnitude, fixed categorical order for groups
# ---------------------------------------------------------------------------
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
CAT_MARKERS = ["o", "s", "D", "^", "v", "P"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "pdf.fonttype": 42,
})

# Fallback grouping (used only without --taxonomy)
FALLBACK_GROUPS = {
    "Modality": {"C0002978", "C0024485", "C0040405", "C0041618", "C0920367", "C1306645", "C1699633", "C0032743"},
    "Broad region": {"C0000726", "C0030797", "C0817096", "C0023216", "C1140618", "C0027530", "C0006141",
                     "C0006104", "C0018787", "C0037949", "C0037303"},
    "Finding/device": {"C0025062", "C0031039", "C0032227", "C0747635", "C0034065", "C0444611", "C0027551"},
}
FALLBACK_ORDER = ["Modality", "Broad region", "Fine anatomy", "Finding/device"]

# Taxonomy categories, mapped the same way as in the other thesis scripts
TAXONOMY_ORDER = ["Imaging Modality", "Anatomy", "Radiological Finding", "Pathology",
                  "Device & Intervention", "Clinical Context"]

# Shorter display names for long UMLS preferred names
SHORT = {
    "Bone structure of humerus": "Humerus", "Bone structure of clavicle": "Clavicle",
    "Bone structure of cranium": "Cranium", "Left coronary artery structure": "Left coronary artery",
    "Right coronary artery structure": "Right coronary artery", "Ascending aorta structure": "Ascending aorta",
    "Pulmonary artery structure": "Pulmonary artery", "Pleural effusion disorder": "Pleural effusion",
    "Bilateral pleural effusion": "Bilateral pleural effusion",
    "Structure of left shoulder region": "Left shoulder", "Structure of right shoulder region": "Right shoulder",
    "Structure of left knee region": "Left knee", "Structure of right knee region": "Right knee",
    "Right ventricular structure": "Right ventricle", "Left ventricular structure": "Left ventricle",
    "Left atrial structure": "Left atrium", "Right atrial structure": "Right atrium",
    "Entire right atrium": "Right atrium (entire)", "Entire left atrium": "Left atrium (entire)",
    "Cerebral white matter structure": "Cerebral white matter",
    "Periventricular white matter": "Periventricular white matter",
    "Tomography, Optical Coherence": "OCT", "X-Ray Computed Tomography": "CT",
    "Magnetic Resonance Imaging": "MRI", "Positron-Emission Tomography": "PET",
    "Anterior descending branch of left coronary artery": "LAD artery",
    "Structure of circumflex branch of left coronary artery": "Circumflex artery",
    "Intestines, Small": "Small intestine", "Aortic valve structure": "Aortic valve",
    "Structure of wisdom tooth": "Wisdom tooth", "Chest, abdomen, and pelvis": "Chest-abdomen-pelvis",
    "Tooth root structure": "Tooth root", "Thoracic spine structure": "Thoracic spine",
    "Common bile duct structure": "Common bile duct", "Bile duct structure": "Bile duct",
    "Right hip region structure": "Right hip", "Internal carotid artery structure": "Internal carotid artery",
    "Metastases to liver": "Liver metastases", "Structure of middle lobe of right lung": "Right middle lobe",
}

RX = re.compile(r"(C\d{7}) \((.*?)\) \[([\d.]+)(?:x, n=\d+)?\]")


def short(name):
    return SHORT.get(name, name)


def sentence_case(name):
    """'Mediastinal Emphysema' -> 'Mediastinal emphysema', keeps acronyms (CT, OCT, LAD)."""
    words = name.split()
    out = [words[0]] + [w if (w.isupper() or len(w) <= 3 and w.upper() == w) else w.lower() for w in words[1:]]
    return " ".join(out)


def pretty_model(run):
    """'tr_n32' -> 'TR head with 32 experts'."""
    m = re.match(r"(cp|tr)_n(\d+)", run)
    return f"{m.group(1).upper()} head with {m.group(2)} experts" if m else run


def load_reports(report_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(report_dir, "*.csv"))):
        with open(path) as f:
            lines = [l for l in f if not l.startswith("#")]
        rows = []
        for r in list(csv.reader(lines))[1:]:
            rows.append({
                "expert": r[0], "load": float(r[2]),
                "enriched": [(c, n, float(v)) for c, n, v in RX.findall(r[4])],
                "poly": float(r[5]) if re.match(r"^[0-9.]+$", r[5]) else None,
                "drops": [(c, n, float(v)) for c, n, v in RX.findall(r[7])],
            })
        runs[os.path.basename(path)[:-4]] = rows      # rows are sorted by load
    return runs


def build_grouper(taxonomy_path):
    if taxonomy_path:
        with open(taxonomy_path) as f:
            raw = json.load(f)
        cat = {}
        for item in raw.get("classifications", []):
            cui = str(item.get("cui", "")).strip().upper()
            c = str(item.get("category", "")).strip().lower()
            if not cui:
                continue
            if "modality" in c:
                cat[cui] = "Imaging Modality"
            elif "anatomy" in c:
                cat[cui] = "Anatomy"
            elif "finding" in c:
                cat[cui] = "Radiological Finding"
            elif "pathology" in c:
                cat[cui] = "Pathology"
            elif "device" in c:
                cat[cui] = "Device & Intervention"
            elif "clinical context" in c:
                cat[cui] = "Clinical Context"
        return (lambda c: cat.get(c, "Other")), TAXONOMY_ORDER + ["Other"]

    def fallback(c):
        for g, s in FALLBACK_GROUPS.items():
            if c in s:
                return g
        return "Fine anatomy"
    return fallback, FALLBACK_ORDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", required=True)
    ap.add_argument("--taxonomy", default=None, help="concept_classifications.json (optional)")
    ap.add_argument("--model", default="tr_n32", help="run name for the heatmap and table")
    ap.add_argument("--top-experts", type=int, default=10)
    ap.add_argument("--min-cell", type=float, default=0.10,
                    help="heatmap shows concepts with a drop >= this for at least one shown expert")
    ap.add_argument("--label-top", type=int, default=2, help="labelled points per group in Fig. 1")
    ap.add_argument("--out-dir", default="figures_mumoe")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = load_reports(os.path.join(args.ablation_dir, "reports"))
    with open(os.path.join(args.ablation_dir, "concepts_fixed.txt")) as f:
        fixed = [l.strip() for l in f if l.strip()]
    group_of, order = build_grouper(args.taxonomy)
    if args.model not in runs:
        raise SystemExit(f"--model {args.model} not found; available: {sorted(runs)}")

    names = {}
    for rows in runs.values():
        for r in rows:
            for c, n, _ in r["enriched"] + r["drops"]:
                names.setdefault(c, n)
    missing = [c for c in fixed if c not in names]   # concepts never affected or enriched
    for c in missing:
        names[c] = c

    # Figure 1
    best = collections.defaultdict(float)
    for rows in runs.values():
        for r in rows:
            for c, _, v in r["drops"]:
                best[c] = max(best[c], v)
    groups = [g for g in order if any(group_of(c) == g for c in fixed)]
    data = {g: sorted([(best.get(c, 0.0), c) for c in fixed if group_of(c) == g], reverse=True) for g in groups}

    def wrap(label, width=10):
        """Split long text over two lines so neighbouring labels never touch."""
        if len(label) <= width or " " not in label:
            return label
        words = label.split()
        best_split = min(range(1, len(words)),
                         key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
        return " ".join(words[:best_split]) + "\n" + " ".join(words[best_split:])

    top = max(v for vals in data.values() for v, _ in vals)
    ymax = max(0.1, top * 1.12)
    min_gap = 0.075 * (ymax + 0.02)          # minimum vertical distance between two labels

    rng = random.Random(0)
    fig, ax = plt.subplots(figsize=(max(6.2, 1.35 * len(groups) + 0.8), 3.8))
    for i, g in enumerate(groups):
        vals = data[g]
        xs = [i - 0.08 + rng.uniform(-0.14, 0.14) for _ in vals]
        ax.scatter(xs, [v for v, _ in vals], s=34, marker=CAT_MARKERS[i % 6], color=CAT_COLORS[i % 6],
                   edgecolor="white", linewidth=0.8, zorder=3)

        # labels for the largest drops, stacked in a column right of the dots
        lab = [(v, c, x) for (v, c), x in list(zip(vals, xs))[:args.label_top] if v >= 0.15]
        ys = [v for v, _, _ in lab]
        for k in range(1, len(ys)):                       # push down labels that would overlap
            ys[k] = min(ys[k], ys[k - 1] - min_gap)
        for (v, c, x), y in zip(lab, ys):
            ax.annotate(wrap(sentence_case(short(names[c])), 16), xy=(x, v), xytext=(i + 0.16, y),
                        textcoords="data", va="center", ha="left", fontsize=7, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5,
                                        shrinkA=0, shrinkB=3))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{wrap(g, 12)}\n($n={len(data[g])}$)" for g in groups], fontsize=8.5)
    ax.set_ylabel("Largest relative AP drop\n(any expert, any model)")
    ax.set_ylim(-0.02, ymax)
    ax.set_xlim(-0.5, len(groups) - 0.2)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"concept_dependence.{ext}"), dpi=200)
    plt.close(fig)

    # Figure 2
    rows = runs[args.model][:args.top_experts]
    mats = [{c: v for c, _, v in r["drops"]} for r in rows]
    shown = {c for m in mats for c, v in m.items() if v >= args.min_cell}
    # order concepts by the expert they depend on most -> diagonal blocks
    concepts = sorted(shown, key=lambda c: (max(range(len(mats)), key=lambda i: mats[i].get(c, 0)),
                                            -max(m.get(c, 0) for m in mats)))
    M = [[m.get(c, 0.0) for m in mats] for c in concepts]
    vmax = max(0.1, max(max(r) for r in M)) if M else 0.1

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(5.4, 0.9 + 0.24 * len(concepts)))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"E{r['expert']}" for r in rows], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels([sentence_case(short(names[c])) for c in concepts], fontsize=8)
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            if v >= args.min_cell:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 * vmax else INK)
    ax.set_xticks([x - 0.5 for x in range(1, len(rows))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(concepts))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, shrink=0.6)
    cb.set_label("Relative AP drop", fontsize=8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"expert_concept_heatmap.{ext}"), dpi=200)
    plt.close(fig)

    # Table
    def tex(s):
        return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

    body = []
    for r in rows:
        chosen = ", ".join(sentence_case(short(n)) for _, n, _ in r["enriched"][:3]) or "--"
        needed = ", ".join(f"{sentence_case(short(n))} ({v:.2f})" for _, n, v in r["drops"][:3] if v >= 0.05) or "--"
        p = f"{r['poly']:.2f}" if r["poly"] is not None else "--"
        body.append(f"E{r['expert']} & {r['load']:.1f} & {tex(chosen)} & {tex(needed)} & {p} \\tabularnewline")
    table = "\n".join([
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{The " + str(len(rows)) + " busiest experts of the " + pretty_model(args.model) +
        ". Load is the share of validation images routed to the expert. \\emph{Chosen for} lists the three "
        "most enriched concepts among those images, and \\emph{Needed for} the concepts whose AP drops most when "
        "the expert is removed (relative drop in parentheses, drops below 0.05 omitted). $p$ is the expert's "
        "polysemanticity.}",
        "\\label{tab:expert_profiles}",
        "\\small",
        "\\begin{tabular}{rr p{4.3cm} p{5.2cm} c}",
        "\\toprule",
        "Expert & Load (\\%) & Chosen for & Needed for & $p$ \\tabularnewline",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    with open(os.path.join(args.out_dir, "expert_profiles.tex"), "w") as f:
        f.write(table + "\n")

    print(f"Runs read: {sorted(runs)}")
    print(f"Concepts in fixed set: {len(fixed)} | grouping: {'taxonomy' if args.taxonomy else 'built-in'}")
    for g in groups:
        print(f"  {g:24s} n={len(data[g]):3d}  max drop={data[g][0][0]:.2f} ({short(names[data[g][0][1]])})")
    print(f"Heatmap ({args.model}): {len(rows)} experts x {len(concepts)} concepts")
    print(f"Wrote figures and table to {args.out_dir}/")


if __name__ == "__main__":
    main()
