"""Count, for every parent-child edge, the training images annotated with both concepts.

Used to build the co-occurrence-restricted hierarchies (restrict_live.py).

Usage:
    python edge_cooccurrence.py --train_csv train_concepts.csv --hierarchy_csv umls_hierarchy.csv
"""

import argparse
import csv
from collections import defaultdict


def load_hierarchy(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_train_concepts(path: str):
    cui_to_samples = defaultdict(set)
    n_samples = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_samples += 1
            sample_id = row["ID"]
            cuis = {c.strip() for c in row["CUIs"].split(";") if c.strip()}
            for cui in cuis:
                cui_to_samples[cui].add(sample_id)
    return cui_to_samples, n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train_concepts.csv")
    parser.add_argument("--hierarchy_csv", default="umls_hierarchy.csv")
    parser.add_argument("--output_csv", default="edge_cooccurrence.csv")
    args = parser.parse_args()

    print(f"Loading hierarchy edges from {args.hierarchy_csv}...")
    edges = load_hierarchy(args.hierarchy_csv)
    print(f"  {len(edges)} edges loaded")

    print(f"\nLoading training samples from {args.train_csv}...")
    cui_to_samples, n_samples = load_train_concepts(args.train_csv)
    print(f"  {n_samples} samples, {len(cui_to_samples)} distinct CUIs seen")

    print("\nComputing edge co-occurrence (set intersection per edge)...")
    results = []
    edge_sample_sets = {}
    for e in edges:
        child, parent = e["child_cui"], e["parent_cui"]
        child_samples = cui_to_samples.get(child, set())
        parent_samples = cui_to_samples.get(parent, set())
        cooccur = child_samples & parent_samples
        edge_sample_sets[(child, parent)] = cooccur
        results.append({
            "child_cui": child, "child_name": e["child_name"],
            "parent_cui": parent, "parent_name": e["parent_name"],
            "source": e["source"],
            "child_sample_count": len(child_samples),
            "parent_sample_count": len(parent_samples),
            "cooccurrence_count": len(cooccur),
        })

    results.sort(key=lambda r: r["cooccurrence_count"], reverse=True)

    with open(args.output_csv, "w", newline="") as f:
        fieldnames = ["child_cui", "child_name", "parent_cui", "parent_name",
                      "source", "child_sample_count", "parent_sample_count",
                      "cooccurrence_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved per-edge co-occurrence -> {args.output_csv}")

    # aggregate stats
    live_edges = [r for r in results if r["cooccurrence_count"] > 0]
    dead_edges = [r for r in results if r["cooccurrence_count"] == 0]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total edges                                      : {len(edges)}")
    if edges:
        print(f"Live edges (co-occur in >=1 sample)              : {len(live_edges)} ({100*len(live_edges)/len(edges):.1f}%)")
        print(f"Dead edges (never co-occur in this dataset)      : {len(dead_edges)} ({100*len(dead_edges)/len(edges):.1f}%)")

    if live_edges:
        counts = sorted(r["cooccurrence_count"] for r in live_edges)
        n = len(counts)
        print("\nCo-occurrence count distribution (live edges only):")
        print(f"  min / median / max                             : {counts[0]} / {counts[n // 2]} / {counts[-1]}")
        print(f"  mean                                             : {sum(counts) / n:.2f}")

    print("\nTop 10 most frequently co-occurring edges:")
    for r in results[:10]:
        print(f"  {r['child_name']} + {r['parent_name']}: {r['cooccurrence_count']} samples  [{r['source']}]")

    # per-sample stats
    print("\nComputing per-sample satisfied-edge counts...")
    sample_satisfied_count = defaultdict(int)
    for cooccur in edge_sample_sets.values():
        for s in cooccur:
            sample_satisfied_count[s] += 1

    samples_with_any = len(sample_satisfied_count)
    samples_with_zero = n_samples - samples_with_any
    print(f"\nSamples with >=1 satisfied hierarchy edge        : {samples_with_any} ({100*samples_with_any / n_samples:.1f}% of {n_samples})")
    print(f"Samples with 0 satisfied hierarchy edges         : {samples_with_zero} ({100*samples_with_zero / n_samples:.1f}%)")
    if sample_satisfied_count:
        vals = sorted(sample_satisfied_count.values())
        m = len(vals)
        print(f"Among samples with >=1: min/median/max edges satisfied : {vals[0]} / {vals[m // 2]} / {vals[-1]}")


if __name__ == "__main__":
    main()
