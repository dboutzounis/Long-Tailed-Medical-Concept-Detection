"""Collect the per-run JSON summaries of analyze_experts.py into one table (Tables 5.14-5.15).

Usage:
    python summarize_ablations.py ablations/summaries/*.json --output ablations/summary_table.csv
"""
import argparse
import csv
import json

COLUMNS = [
    ("name", "run"),
    ("head_type", "head"),
    ("metric", "metric"),
    ("n_experts", "N"),
    ("decision_threshold", "thr"),
    ("max_coef_median", "median_max_coef"),
    ("uniform_coef", "uniform_coef"),
    ("pct_dead", "pct_dead_top1"),
    ("effective_experts", "effective_experts"),
    ("max_load_pct", "max_load_pct"),
    ("n_concepts_metric", "n_concepts"),
    ("n_measurable", "n_measurable"),
    ("n_ablated", "n_ablated"),
    ("mean_polysemanticity", "poly_mean"),
    ("mean_polysemanticity_above_median_load", "poly_mean_above_median_load"),
    ("mean_max_drop", "mean_max_drop"),
    ("corr_load_polysemanticity", "corr_load_poly"),
    ("used_feature_cache", "cache"),
]


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    rows = []
    for path in args.inputs:
        with open(path) as f:
            s = json.load(f)
        s["name"] = path.rsplit("/", 1)[-1].removesuffix(".json")
        rows.append(s)
    rows.sort(key=lambda s: (s.get("head_type", ""), s.get("n_experts", 0)))

    header = [h for _, h in COLUMNS]
    table = [[fmt(s.get(k, "")) for k, _ in COLUMNS] for s in rows]

    widths = [max(len(h), *(len(r[i]) for r in table)) for i, h in enumerate(header)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in table:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(table)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
