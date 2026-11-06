"""Step 2: concatenate the API and RRF edge lists. Duplicates and conflicts are resolved in step 3.

Usage:
    python merge_relations.py --rrf_csv umls_hierarchy_rrf.csv --api_csv umls_hierarchy_api.csv
"""

import argparse
import csv


def load_edges(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rrf_csv", default="umls_hierarchy_rrf.csv")
    parser.add_argument("--api_csv", default="umls_hierarchy_api.csv")
    parser.add_argument("--output_csv", default="umls_hierarchy_merged.csv")
    args = parser.parse_args()

    rrf_edges = load_edges(args.rrf_csv)
    api_edges = load_edges(args.api_csv)
    print(f"Loaded {len(rrf_edges)} edges from {args.rrf_csv}")
    print(f"Loaded {len(api_edges)} edges from {args.api_csv}")

    merged = rrf_edges + api_edges

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["child_cui", "child_name", "parent_cui", "parent_name", "source"])
        writer.writeheader()
        writer.writerows(merged)

    print(f"\nWrote {len(merged)} raw (unresolved) rows -> {args.output_csv}")
    print("This file still contains duplicates and direction conflicts by design --")
    print("run build_descendant_matrix.py on it next to resolve them.")


if __name__ == "__main__":
    main()
