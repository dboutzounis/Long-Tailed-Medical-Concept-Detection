"""Restricted hierarchy for MCLoss: keep only edges between two Anatomy concepts (Table 5.3).

Concepts left without edges keep only themselves in their descendant set, so MCLoss reduces
to BCE for them.

Usage:
    python restrict_anatomy.py --hierarchy_csv umls_hierarchy.csv --mlb_path mlb.pkl
"""

import argparse
import csv
import json

import networkx as nx
import numpy as np
from scipy import sparse

CHAIN_ANCESTOR = "C0022417"    # Joints
CHAIN_DESCENDANT = "C0042173"  # Uvula of palate


def load_cui_categories(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {c["cui"]: c["category"] for c in data["classifications"]}


def load_edges(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def filter_to_anatomy(edges: list[dict], cui_to_cat: dict) -> tuple[list[dict], int]:
    kept = []
    missing_cat = 0
    for e in edges:
        child_cat = cui_to_cat.get(e["child_cui"])
        parent_cat = cui_to_cat.get(e["parent_cui"])
        if child_cat is None or parent_cat is None:
            missing_cat += 1
            continue
        if child_cat == "Anatomy" and parent_cat == "Anatomy":
            kept.append(e)
    return kept, missing_cat


def taxonomic_descendants(G: nx.DiGraph, cui: str) -> set:
    """D_A: {A} union all more-specific concepts under A. Same direction
    convention as build_descendant_matrix.py -- nx.ancestors(G, A),
    NOT nx.descendants(G, A). See that script's docstring for why."""
    if cui not in G:
        return {cui}
    return nx.ancestors(G, cui) | {cui}


def self_test_direction(G: nx.DiGraph):
    if CHAIN_ANCESTOR not in G or CHAIN_DESCENDANT not in G:
        print("  (self-test chain not present after Anatomy filtering -- skipped)")
        return
    taxonomic_descendants_of_ancestor = nx.ancestors(G, CHAIN_ANCESTOR)
    assert CHAIN_DESCENDANT in taxonomic_descendants_of_ancestor, (
        "DIRECTION SELF-TEST FAILED after Anatomy filtering -- do NOT trust "
        "the output matrix. This would be surprising since filtering a "
        "verified-correct graph's edges can't change direction, so treat "
        "this as a sign something upstream changed unexpectedly."
    )
    print("  Direction self-test passed (Joints -> ... -> Uvula chain confirmed).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy_csv", default="umls_hierarchy.csv")
    parser.add_argument("--classifications_json", default="../taxonomy/concept_classifications.json")
    parser.add_argument("--mlb_path", required=True)
    parser.add_argument("--matrix_out", default="umls_descendant_matrix_anatomy.npz")
    parser.add_argument("--index_out", default="umls_cui_index_anatomy.json")
    args = parser.parse_args()

    import pickle
    with open(args.mlb_path, "rb") as f:
        mlb = pickle.load(f)
    cui_order = list(mlb.classes_)
    N = len(cui_order)
    print(f"Loaded {N} CUIs from mlb (defines matrix row/col order)")

    cui_to_cat = load_cui_categories(args.classifications_json)
    print(f"Loaded categories for {len(cui_to_cat)} CUIs")

    all_edges = load_edges(args.hierarchy_csv)
    print(f"Loaded {len(all_edges)} edges from {args.hierarchy_csv}")

    anatomy_edges, missing_cat = filter_to_anatomy(all_edges, cui_to_cat)
    if missing_cat:
        print(f"  WARNING: {missing_cat} edges skipped (CUI missing from classifications)")
    print(f"Kept {len(anatomy_edges)} / {len(all_edges)} edges "
          f"({100 * len(anatomy_edges) / len(all_edges):.1f}%) where BOTH endpoints are Anatomy")

    G = nx.DiGraph()
    name_lookup = {}
    for e in anatomy_edges:
        G.add_edge(e["child_cui"], e["parent_cui"])
        name_lookup[e["child_cui"]] = e["child_name"]
        name_lookup[e["parent_cui"]] = e["parent_name"]

    assert nx.is_directed_acyclic_graph(G), (
        "Anatomy-filtered subgraph has a cycle -- this should be impossible "
        "given umls_hierarchy.csv was already verified acyclic; stop and "
        "inspect rather than trusting the output."
    )
    print("\nRunning direction self-test...")
    self_test_direction(G)

    cui_to_idx = {cui: i for i, cui in enumerate(cui_order)}
    cui_set = set(cui_order)

    print(f"\nComputing descendant sets (D_A) for all {N} CUIs (Anatomy-restricted)...")
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
    print(f"Truly isolated (zero edges -- plain BCE, no MCM effect)  : {truly_isolated}")
    print(f"Leaves (have a parent, but no children of their own)     : {leaves_in_hierarchy}")
    print(f"Internal nodes (have >=1 descendant -- drive MCM's max)  : {internal_nodes}")
    print(f"  -> Touched by at least one edge (leaves + internal)    : {leaves_in_hierarchy + internal_nodes}")
    if internal_nodes > 0:
        print(f"Mean descendant set size (internal nodes only)           : {desc_sizes[desc_sizes > 1].mean():.2f}")
    print(f"Max descendant set size                                  : {desc_sizes.max()}")
    print("\nNote: every CUI outside this count now trains under plain BCE via MCLoss's "
          "own graceful-degradation path -- this is the entire mechanism of the test.")


if __name__ == "__main__":
    main()
