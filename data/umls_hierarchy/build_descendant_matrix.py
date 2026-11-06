"""Step 3: build the concept hierarchy and its descendant matrix (Figure 4.2).

Duplicate edges are collapsed, opposite directions asserted by different vocabularies are
resolved by source priority (SNOMED CT > FMA > NCI > MeSH > RadLex), and any remaining cycle is
broken. The transitive closure is stored as a sparse matrix D (C x C) in label-binarizer order,
with D[a, c] = 1 if c is a or one of its descendants.

Outputs: umls_hierarchy.csv (clean edges), umls_descendant_matrix.npz, umls_cui_index.json.

Usage:
    python build_descendant_matrix.py --hierarchy_csv umls_hierarchy_merged.csv --mlb_path mlb.pkl
"""

import argparse
import csv
import json
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy import sparse

# Highest priority first. See rationale in the module docstring above.
SOURCE_PRIORITY = ["SNOMEDCT_US", "FMA", "NCI", "MSH", "RADLEX"]

DIRECTION_TEST_ANCESTOR = "C0022417"  # Joints
DIRECTION_TEST_DESCENDANT = "C0042173"  # Uvula of palate


def source_rank(source: str) -> int:
    try:
        return SOURCE_PRIORITY.index(source)
    except ValueError:
        return len(SOURCE_PRIORITY)  # unknown sources sort last


def load_edges(hierarchy_csv: str) -> list[dict]:
    with open(hierarchy_csv, newline="") as f:
        return list(csv.DictReader(f))


def resolve_conflicts(edges: list[dict]) -> list[dict]:
    by_pair = defaultdict(list)
    for e in edges:
        by_pair[(e["child_cui"], e["parent_cui"])].append(e)

    seen_pairs = set()
    dropped_conflicts, kept = [], []
    collapsed_duplicates = 0

    for (child, parent), rows in by_pair.items():
        pair_key = tuple(sorted([child, parent]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        best_forward = min(rows, key=lambda e: source_rank(e["source"]))
        collapsed_duplicates += len(rows) - 1

        reverse = (parent, child)
        if reverse in by_pair:
            backward_rows = by_pair[reverse]
            best_backward = min(backward_rows, key=lambda e: source_rank(e["source"]))
            collapsed_duplicates += len(backward_rows) - 1

            if source_rank(best_forward["source"]) <= source_rank(best_backward["source"]):
                kept.append(best_forward)
                dropped_conflicts.append(best_backward)
            else:
                kept.append(best_backward)
                dropped_conflicts.append(best_forward)
        else:
            kept.append(best_forward)

    print(f"\nSame-direction duplicate rows collapsed: {collapsed_duplicates}")
    print(f"Reciprocal-direction conflicts resolved: {len(dropped_conflicts)}")
    for d in dropped_conflicts:
        print(f"  DROPPED: {d['child_cui']} ({d.get('child_name','')}) -> "
              f"{d['parent_cui']} ({d.get('parent_name','')})  [{d['source']}]")

    return kept


def build_graph(edges: list[dict]) -> tuple[nx.DiGraph, dict]:
    G = nx.DiGraph()
    edge_lookup = {}
    for e in edges:
        G.add_edge(e["child_cui"], e["parent_cui"])
        edge_lookup[(e["child_cui"], e["parent_cui"])] = e["source"]
    return G, edge_lookup


def break_remaining_cycles(G: nx.DiGraph, edge_lookup: dict) -> nx.DiGraph:
    iterations = 0
    while not nx.is_directed_acyclic_graph(G):
        iterations += 1
        if iterations > 50:
            raise RuntimeError("Could not break cycles after 50 iterations -- inspect manually.")
        cycle = nx.find_cycle(G)
        worst = max(cycle, key=lambda uv: source_rank(edge_lookup.get((uv[0], uv[1]), "")))
        print(f"  WARNING: additional cycle found, dropping {worst[0]} -> {worst[1]}")
        G.remove_edge(worst[0], worst[1])

    print(f"Additional (non-reciprocal) cycles broken: {iterations}")
    return G


def self_test_direction(G: nx.DiGraph):
    """Refuse to proceed if ancestor/descendant direction looks inverted."""
    if DIRECTION_TEST_ANCESTOR not in G or DIRECTION_TEST_DESCENDANT not in G:
        print("  (self-test chain not present in this graph run -- skipped)")
        return
    taxonomic_descendants_of_ancestor = nx.ancestors(G, DIRECTION_TEST_ANCESTOR)
    assert DIRECTION_TEST_DESCENDANT in taxonomic_descendants_of_ancestor, (
        "DIRECTION SELF-TEST FAILED: expected descendant not found via "
        "nx.ancestors(). Do NOT trust the output matrix -- direction logic "
        "is inverted somewhere upstream."
    )
    print("  Direction self-test passed (Joints -> ... -> Uvula chain confirmed).")


def taxonomic_descendants(G: nx.DiGraph, cui: str) -> set:
    """D_A for MCM: {A} union all more-specific concepts under A."""
    if cui not in G:
        return {cui}
    return nx.ancestors(G, cui) | {cui}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy_csv", default="umls_hierarchy_merged.csv")
    parser.add_argument("--mlb_path", required=True)
    parser.add_argument("--clean_csv_out", default="umls_hierarchy.csv")
    parser.add_argument("--matrix_out", default="umls_descendant_matrix.npz")
    parser.add_argument("--index_out", default="umls_cui_index.json")
    args = parser.parse_args()

    with open(args.mlb_path, "rb") as f:
        mlb = pickle.load(f)
    cui_order = list(mlb.classes_)
    print(f"Loaded {len(cui_order)} CUIs from mlb (defines matrix row/col order)")

    raw_edges = load_edges(args.hierarchy_csv)
    print(f"Loaded {len(raw_edges)} raw edges from {args.hierarchy_csv}")

    name_lookup = {}
    for e in raw_edges:
        name_lookup[e["child_cui"]] = e["child_name"]
        name_lookup[e["parent_cui"]] = e["parent_name"]

    edges = resolve_conflicts(raw_edges)

    G, edge_lookup = build_graph(edges)
    print("\nChecking for cycles...")
    G = break_remaining_cycles(G, edge_lookup)
    assert nx.is_directed_acyclic_graph(G), "Graph still has cycles after cleanup -- bug, stop and inspect."

    print("\nRunning direction self-test...")
    self_test_direction(G)

    with open(args.clean_csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["child_cui", "child_name", "parent_cui", "parent_name", "source"])
        writer.writeheader()
        for u, v in sorted(G.edges()):
            writer.writerow({
                "child_cui": u, "child_name": name_lookup.get(u, u),
                "parent_cui": v, "parent_name": name_lookup.get(v, v),
                "source": edge_lookup.get((u, v), ""),
            })
    print(f"\nSaved cleaned edges -> {args.clean_csv_out} ({G.number_of_edges()} edges)")

    N = len(cui_order)
    cui_to_idx = {cui: i for i, cui in enumerate(cui_order)}
    cui_set = set(cui_order)

    print(f"\nComputing descendant sets (D_A) for all {N} CUIs...")
    rows, cols, desc_sizes = [], [], []
    for cui in cui_order:
        i = cui_to_idx[cui]
        D_A = taxonomic_descendants(G, cui) & cui_set
        desc_sizes.append(len(D_A))
        for other in D_A:
            rows.append(i)
            cols.append(cui_to_idx[other])

    data = np.ones(len(rows), dtype=np.int8)
    D_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    sparse.save_npz(args.matrix_out, D_matrix)
    with open(args.index_out, "w") as f:
        json.dump(cui_order, f)

    print(f"Saved descendant matrix -> {args.matrix_out}  (shape {D_matrix.shape}, {D_matrix.nnz} nonzero entries)")
    print(f"Saved CUI index -> {args.index_out}")

    desc_sizes = np.array(desc_sizes)
    touched = np.array([cui in G for cui in cui_order])
    truly_isolated = int((~touched).sum())
    leaves_in_hierarchy = int((touched & (desc_sizes == 1)).sum())
    internal_nodes = int((desc_sizes > 1).sum())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total CUIs                                               : {N}")
    print(f"Truly isolated (zero edges -- no MCM benefit at all)     : {truly_isolated}")
    print(f"Leaves (have a parent, but no children of their own)     : {leaves_in_hierarchy}")
    print(f"Internal nodes (have >=1 descendant -- drive MCM's max)  : {internal_nodes}")
    print(f"  -> Touched by at least one edge (leaves + internal)    : {leaves_in_hierarchy + internal_nodes}")
    if internal_nodes > 0:
        print(f"Mean descendant set size (internal nodes only)           : {desc_sizes[desc_sizes > 1].mean():.2f}")
    print(f"Max descendant set size                                  : {desc_sizes.max()}")
    top5_idx = np.argsort(desc_sizes)[-5:][::-1]
    print("\nTop 5 CUIs by descendant-set size (broadest subtrees -- sanity check these):")
    for idx in top5_idx:
        cui = cui_order[idx]
        print(f"  {cui} ({name_lookup.get(cui, '?')}): {desc_sizes[idx]} descendants")


if __name__ == "__main__":
    main()
