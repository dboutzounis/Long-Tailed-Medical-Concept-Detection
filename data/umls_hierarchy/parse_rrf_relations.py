"""Step 1b: the same PAR/CHD relations from the local UMLS release files (MRREL.RRF, MRCONSO.RRF).

Source atoms are mapped to CUIs and only edges inside the label vocabulary are kept. The edge
direction is checked on a known chain (Uvula of palate -> ... -> Joints) before writing, and the
result is compared against the API edge list.

Output: umls_hierarchy_rrf.csv.

Usage:
    python parse_rrf_relations.py --mrrel_path MRREL.RRF --mrconso_path MRCONSO.RRF --mlb_path mlb.pkl
"""

import argparse
import csv
import pickle
from collections import defaultdict

TRUSTED_SOURCES = {"SNOMEDCT_US", "MSH", "NCI", "FMA", "RADLEX"}

GENERIC_TERMS = {"disease", "finding", "structure", "body structure",
                  "anatomical structure", "neoplasm", "procedure",
                  "disorder", "abnormality", "body region"}

SPECIFICITY_WORDS = {"left", "right", "upper", "lower", "anterior",
                      "posterior", "distal", "proximal", "medial", "lateral"}

# Known-good chain from the API-derived data, used as a calibration anchor.
CHAIN_ANCESTOR = "C0022417"   # Joints
CHAIN_DESCENDANT = "C0042173"  # Uvula of palate


def specificity_score(name: str) -> int:
    name_l = name.lower()
    score = name_l.count("structure of")
    score += name.count(",")
    score += sum(1 for w in SPECIFICITY_WORDS if w in name_l.split())
    score += max(0, len(name.split()) - 2)
    return score


def load_label_cuis(mlb_path: str) -> set:
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)
    return set(mlb.classes_)


def load_names(mrconso_path: str, cuis: set) -> dict:
    """Stream MRCONSO.RRF, keep first English name per label-set CUI."""
    names = {}
    with open(mrconso_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\n").split("|")
            cui, lat, str_ = fields[0], fields[1], fields[14]
            if cui not in cuis or cui in names or lat != "ENG":
                continue
            names[cui] = str_
    missing = cuis - names.keys()
    if missing:
        print(f"  WARNING: {len(missing)} label-set CUIs had no ENG name in MRCONSO.RRF")
    return names


def extract_raw_relations(mrrel_path: str, cuis: set) -> list[tuple]:
    """Stream MRREL.RRF, keep PAR/CHD rows between two label-set CUIs, trusted SAB."""
    raw = []
    with open(mrrel_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\n").split("|")
            cui1, rel, cui2, sab = fields[0], fields[3], fields[4], fields[10]
            if rel not in ("PAR", "CHD"):
                continue
            if sab not in TRUSTED_SOURCES:
                continue
            if cui1 == cui2 or cui1 not in cuis or cui2 not in cuis:
                continue
            raw.append((cui1, rel, cui2, sab))
    return raw


def build_hypothesis_edges(raw: list[tuple], flipped: bool) -> list[dict]:
    """Standard reading (flipped=False): REL=PAR -> CUI2 is parent of CUI1."""
    edges, seen = [], set()
    for cui1, rel, cui2, sab in raw:
        if not flipped:
            child, parent = (cui1, cui2) if rel == "PAR" else (cui2, cui1)
        else:
            child, parent = (cui2, cui1) if rel == "PAR" else (cui1, cui2)
        key = (child, parent)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"child_cui": child, "parent_cui": parent, "source": sab})
    return edges


def score_hypothesis(edges: list[dict], names: dict) -> dict:
    """Run the three calibration checks against one hypothesis's edge set."""
    generic_as_child = generic_as_parent = 0
    spec_child_narrower = spec_comparable = 0

    child_map = defaultdict(set)
    for e in edges:
        cn = names.get(e["child_cui"], "").lower()
        pn = names.get(e["parent_cui"], "").lower()
        if cn in GENERIC_TERMS:
            generic_as_child += 1
        if pn in GENERIC_TERMS:
            generic_as_parent += 1

        cs = specificity_score(names.get(e["child_cui"], ""))
        ps = specificity_score(names.get(e["parent_cui"], ""))
        if cs != ps:
            spec_comparable += 1
            if cs > ps:
                spec_child_narrower += 1

        child_map[e["parent_cui"]].add(e["child_cui"])

    chain_pass = None
    if CHAIN_ANCESTOR in child_map or any(CHAIN_ANCESTOR in c for c in child_map.values()):
        seen, frontier, found = set(), {CHAIN_ANCESTOR}, False
        for _ in range(20):
            nxt = set()
            for n in frontier:
                nxt |= child_map.get(n, set())
            nxt -= seen
            if CHAIN_DESCENDANT in nxt:
                found = True
                break
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        chain_pass = found

    return {
        "n_edges": len(edges),
        "generic_as_child": generic_as_child,
        "generic_as_parent": generic_as_parent,
        "spec_child_narrower_pct": (100 * spec_child_narrower / spec_comparable) if spec_comparable else None,
        "chain_pass": chain_pass,
    }


def hypothesis_points(score: dict) -> int:
    points = 0
    if score["generic_as_parent"] > score["generic_as_child"]:
        points += 1
    if score["spec_child_narrower_pct"] is not None and score["spec_child_narrower_pct"] > 50:
        points += 1
    if score["chain_pass"]:
        points += 1
    return points


def compare_hierarchies(rrf_edges: list[dict], api_edges: list[dict]):
    def undirected_map(edges):
        d = defaultdict(set)
        for e in edges:
            d[frozenset((e["child_cui"], e["parent_cui"]))].add((e["child_cui"], e["parent_cui"]))
        return d

    rrf_map = undirected_map(rrf_edges)
    api_map = undirected_map(api_edges)

    rrf_internal_conflicts = sum(1 for v in rrf_map.values() if len(v) > 1)
    api_internal_conflicts = sum(1 for v in api_map.values() if len(v) > 1)

    common = set(rrf_map) & set(api_map)
    agree, disagree, disagree_examples = 0, 0, []
    for pair in common:
        if rrf_map[pair] & api_map[pair]:
            agree += 1
        else:
            disagree += 1
            disagree_examples.append((next(iter(rrf_map[pair])), next(iter(api_map[pair]))))

    print(f"RRF-derived edges (undirected pairs)   : {len(rrf_map)}")
    print(f"API-derived edges (undirected pairs)   : {len(api_map)}")
    print(f"  RRF internal direction conflicts     : {rrf_internal_conflicts}")
    print(f"  API internal direction conflicts     : {api_internal_conflicts}")
    print(f"Pairs present in both sources          : {len(common)}")
    print(f"  Agree on direction                   : {agree}")
    print(f"  DISAGREE on direction                : {disagree}  <-- most important number")
    print(f"RRF-only pairs                          : {len(set(rrf_map) - set(api_map))}")
    print(f"API-only pairs                           : {len(set(api_map) - set(rrf_map))}")

    if disagree_examples:
        print("\nDirection disagreement examples (first 10):")
        for rrf_e, api_e in disagree_examples[:10]:
            print(f"  RRF says: {rrf_e[0]} child_of {rrf_e[1]}")
            print(f"  API says: {api_e[0]} child_of {api_e[1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrrel_path", required=True)
    parser.add_argument("--mrconso_path", required=True)
    parser.add_argument("--mlb_path", required=True)
    parser.add_argument("--compare_csv", default="umls_hierarchy_api.csv")
    parser.add_argument("--output_csv", default="umls_hierarchy_rrf.csv")
    args = parser.parse_args()

    cuis = load_label_cuis(args.mlb_path)
    print(f"Loaded {len(cuis)} label-set CUIs")

    print("\nStreaming MRCONSO.RRF for names (large file, may take a minute)...")
    names = load_names(args.mrconso_path, cuis)
    print(f"  Got names for {len(names)} / {len(cuis)} CUIs")

    print("\nStreaming MRREL.RRF for PAR/CHD relations (large file, may take a few minutes)...")
    raw = extract_raw_relations(args.mrrel_path, cuis)
    print(f"  Found {len(raw)} raw PAR/CHD rows between two label-set CUIs")

    print("\n" + "=" * 60)
    print("DIRECTION CALIBRATION -- testing both hypotheses")
    print("=" * 60)
    standard_edges = build_hypothesis_edges(raw, flipped=False)
    flipped_edges = build_hypothesis_edges(raw, flipped=True)
    standard_score = score_hypothesis(standard_edges, names)
    flipped_score = score_hypothesis(flipped_edges, names)

    for label, score in [("STANDARD (REL=PAR -> CUI2 is parent of CUI1)", standard_score),
                          ("FLIPPED  (REL=PAR -> CUI1 is parent of CUI2)", flipped_score)]:
        print(f"\n{label}")
        print(f"  edges                                    : {score['n_edges']}")
        print(f"  generic term as CHILD  (want ~0)         : {score['generic_as_child']}")
        print(f"  generic term as PARENT (want >> 0)       : {score['generic_as_parent']}")
        if score['spec_child_narrower_pct'] is not None:
            print(f"  child narrower than parent (want > 50%)  : {score['spec_child_narrower_pct']:.1f}%")
        print(f"  Joints -> ... -> Uvula chain check        : {score['chain_pass']}")

    standard_points = hypothesis_points(standard_score)
    flipped_points = hypothesis_points(flipped_score)
    print(f"\nSTANDARD hypothesis score: {standard_points}/3")
    print(f"FLIPPED hypothesis score : {flipped_points}/3")

    if standard_points == flipped_points:
        raise RuntimeError(
            "Calibration inconclusive -- both hypotheses scored equally. "
            "Do not trust either output automatically; inspect the raw "
            "relations manually before proceeding."
        )

    winner_edges = standard_edges if standard_points > flipped_points else flipped_edges
    winner_label = "STANDARD" if standard_points > flipped_points else "FLIPPED"
    print(f"\n==> Using {winner_label} direction (higher calibration score).")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["child_cui", "child_name", "parent_cui", "parent_name", "source"])
        writer.writeheader()
        for e in winner_edges:
            writer.writerow({
                "child_cui": e["child_cui"], "child_name": names.get(e["child_cui"], ""),
                "parent_cui": e["parent_cui"], "parent_name": names.get(e["parent_cui"], ""),
                "source": e["source"],
            })
    print(f"Saved {len(winner_edges)} edges -> {args.output_csv}")

    print("\n" + "=" * 60)
    print(f"COMPARISON vs {args.compare_csv}")
    print("=" * 60)
    try:
        with open(args.compare_csv, newline="") as f:
            api_edges = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"  {args.compare_csv} not found -- skipping comparison.")
        return

    compare_hierarchies(winner_edges, api_edges)


if __name__ == "__main__":
    main()
