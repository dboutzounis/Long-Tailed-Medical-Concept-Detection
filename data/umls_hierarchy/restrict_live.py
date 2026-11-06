"""Restricted hierarchy for MCLoss: keep only edges whose two concepts co-occur in at least
--min_cooccurrence training images (Table 5.3, >= 1 and >= 5).

Usage:
    python restrict_live.py --cooccurrence_csv edge_cooccurrence.csv --mlb_path mlb.pkl --min_cooccurrence 1
    python restrict_live.py --cooccurrence_csv edge_cooccurrence.csv --mlb_path mlb.pkl --min_cooccurrence 5
"""

import argparse
import csv
import json
import pickle

import networkx as nx
import numpy as np
from scipy import sparse

CHAIN_ANCESTOR = "C0022417"    # Joints
CHAIN_DESCENDANT = "C0042173"  # Uvula of palate


def load_cooccurrence(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def filter_by_cooccurrence(edges: list[dict], min_cooccurrence: int) -> list[dict]:
    return [e for e in edges if int(e["cooccurrence_count"]) >= min_cooccurrence]


def taxonomic_descendants(G: nx.DiGraph, cui: str) -> set:
    """D_A: {A} union all more-specific concepts under A. Same direction
    convention as build_descendant_matrix.py --
    nx.ancestors(G, A), NOT nx.descendants(G, A)."""
    if cui not in G:
        return {cui}
    return nx.ancestors(G, cui) | {cui}


def self_test_direction(G: nx.DiGraph):
    if CHAIN_ANCESTOR not in G or CHAIN_DESCENDANT not in G:
        print("  (self-test chain not present after co-occurrence filtering -- skipped)")
        return
    descendants_of_ancestor = nx.ancestors(G, CHAIN_ANCESTOR)
    if CHAIN_DESCENDANT not in descendants_of_ancestor:
        print("  (self-test chain broken by filtering -- at least one link along it didn't "
              "meet --min_cooccurrence. Expected and not an error: the filter operates "
              "edge-by-edge, with no awareness of which multi-hop chains it might split.")
        return
    print("  Direction self-test passed (Joints -> ... -> Uvula chain confirmed).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cooccurrence_csv", default="edge_cooccurrence.csv")
    parser.add_argument("--mlb_path", required=True)
    parser.add_argument("--min_cooccurrence", type=int, default=1)
    parser.add_argument("--matrix_out", default=None,
                        help="default: umls_descendant_matrix_live.npz (min 1) or _live_<min>.npz")
    parser.add_argument("--index_out", default="umls_cui_index_live.json")
    args = parser.parse_args()
    if args.matrix_out is None:
        suffix = "" if args.min_cooccurrence == 1 else f"_{args.min_cooccurrence}"
        args.matrix_out = f"umls_descendant_matrix_live{suffix}.npz"

    with open(args.mlb_path, "rb") as f:
        mlb = pickle.load(f)
    cui_order = list(mlb.classes_)
    N = len(cui_order)
    print(f"Loaded {N} CUIs from mlb (defines matrix row/col order)")

    all_edges = load_cooccurrence(args.cooccurrence_csv)
    print(f"Loaded {len(all_edges)} edges from {args.cooccurrence_csv}")

    live_edges = filter_by_cooccurrence(all_edges, args.min_cooccurrence)
    print(f"Kept {len(live_edges)} / {len(all_edges)} edges "
          f"({100 * len(live_edges) / len(all_edges):.1f}%) with "
          f"cooccurrence_count >= {args.min_cooccurrence}")

    G = nx.DiGraph()
    name_lookup = {}
    for e in live_edges:
        G.add_edge(e["child_cui"], e["parent_cui"])
        name_lookup[e["child_cui"]] = e["child_name"]
        name_lookup[e["parent_cui"]] = e["parent_name"]

    assert nx.is_directed_acyclic_graph(G), (
        "Co-occurrence-filtered subgraph has a cycle -- this should be impossible "
        "given the source hierarchy was already verified acyclic; stop and "
        "inspect rather than trusting the output."
    )
    print("\nRunning direction self-test...")
    self_test_direction(G)

    cui_to_idx = {cui: i for i, cui in enumerate(cui_order)}
    cui_set = set(cui_order)

    print(f"\nComputing descendant sets (D_A) for all {N} CUIs (co-occurrence-restricted)...")
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


if __name__ == "__main__":
    main()
