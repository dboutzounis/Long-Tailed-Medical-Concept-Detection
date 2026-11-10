"""Build the fixed concept set used for the polysemanticity metric.

A concept is kept if it has at least --min-support positive validation images and, in
every model, a baseline AP above --min-ap (or F1 above --min-f1 with --metric f1).
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metric", choices=["f1", "ap"], default="f1")
    p.add_argument("--min-f1", type=float, default=0.05)
    p.add_argument("--min-ap", type=float, default=0.10)
    p.add_argument("--min-support", type=int, default=30)
    args = p.parse_args()

    runs = [torch.load(path, weights_only=False) for path in args.inputs]
    cuis = runs[0]["cuis"]
    support = runs[0]["support"]
    key, thr = ("ap", args.min_ap) if args.metric == "ap" else ("f1", args.min_f1)
    for path, r in zip(args.inputs, runs):
        if r["cuis"] != cuis:
            raise SystemExit(f"Concept order differs in {path}")
        if not torch.equal(r["support"], support):
            raise SystemExit(f"Validation support differs in {path} (different val data?)")
        if key not in r:
            raise SystemExit(f"{path} has no '{key}' entry; rerun phase 1 with the new analyze_experts.py")

    keep = support >= args.min_support
    print(f"Concepts with >= {args.min_support} val positives: {int(keep.sum())}")
    print(f"Criterion: baseline {key.upper()} > {thr} in every model")
    for path, r in zip(args.inputs, runs):
        ok = (r[key] > thr) & (support >= args.min_support)
        print(f"  {path}: {int(ok.sum())} concepts pass")
        keep &= r[key] > thr

    kept = [c for c, k in zip(cuis, keep.tolist()) if k]
    with open(args.output, "w") as f:
        f.write("\n".join(kept) + "\n")
    print(f"Fixed concept set (intersection over {len(runs)} models): {len(kept)} concepts -> {args.output}")


if __name__ == "__main__":
    main()
