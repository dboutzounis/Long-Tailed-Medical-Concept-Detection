"""Step 1a: PAR/CHD relations between the dataset's CUIs from the UMLS REST API (Section 4.2.3).

Queries /CUI/{cui}/relations restricted to SNOMED CT, MeSH, NCI, FMA and RadLex, maps
source-asserted identifiers back to CUIs and keeps only edges whose two ends are both in the
label vocabulary. Responses are cached in .umls_cache/ so interrupted runs resume.

Output: umls_hierarchy_api.csv (child_cui, child_name, parent_cui, parent_name, source).

Usage (needs a free UMLS API key from https://uts.nlm.nih.gov/uts/):
    python fetch_api_relations.py --api_key YOUR_KEY --train_csv train_concepts.csv --mlb_path mlb.pkl
"""

import argparse
import csv
import json
import os
import pickle
import time
from pathlib import Path

import requests
from tqdm import tqdm

TRAIN_CSV        = "train_concepts.csv"
MLB_PATH         = "mlb.pkl"
OUTPUT_HIERARCHY = "umls_hierarchy_api.csv"

CACHE_DIR = ".umls_cache"

BASE_URL   = "https://uts-ws.nlm.nih.gov/rest"
VERSION    = "current"
RATE_LIMIT = 0.1

# Trusted sources for clinical/radiology hierarchy, ranked by reliability.
SOURCES = "SNOMEDCT_US,MSH,NCI,FMA,RADLEX"


def load_label_cuis(train_csv: str, mlb_path: str) -> set:
    cuis = set()
    if os.path.exists(mlb_path):
        with open(mlb_path, "rb") as f:
            mlb = pickle.load(f)
        cuis.update(mlb.classes_)
        print(f"Loaded {len(mlb.classes_)} CUIs from mlb")
    if os.path.exists(train_csv):
        with open(train_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for cui in row["CUIs"].split(";"):
                    cui = cui.strip()
                    if cui.startswith("C"):
                        cuis.add(cui)
    return cuis


def cache_path(key: str) -> Path:
    Path(CACHE_DIR).mkdir(exist_ok=True)
    safe_key = key.replace("/", "_")
    return Path(CACHE_DIR) / f"{safe_key}.json"


def api_get(url: str, params: dict, api_key: str):
    """Shared GET wrapper: rate limiting, error handling, JSON parsing."""
    params = dict(params)
    params["apiKey"] = api_key
    time.sleep(RATE_LIMIT)
    try:
        r = requests.get(url, params=params, timeout=15)
    except requests.RequestException as e:
        print(f"  Request error for {url}: {e}")
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} for {url}")
        return None
    return r.json()


def get_relations(cui: str, api_key: str) -> list[dict]:
    """Fetch PAR/CHD-only relations for a CUI, filtered to trusted sources."""
    cp = cache_path(f"rel_{cui}")
    if cp.exists():
        with open(cp) as f:
            return json.load(f)

    all_results = []
    page = 1
    while True:
        url = f"{BASE_URL}/content/{VERSION}/CUI/{cui}/relations"
        params = {
            "sabs": SOURCES,
            "includeRelationLabels": "PAR,CHD",
            "pageNumber": page,
            "pageSize": 200,
        }
        data = api_get(url, params, api_key)
        if data is None:
            break
        result = data.get("result", [])
        if not result:
            break
        all_results.extend(result)
        if page >= data.get("pageCount", 1):
            break
        page += 1

    with open(cp, "w") as f:
        json.dump(all_results, f)
    return all_results


def get_cui_name(cui: str, api_key: str) -> str:
    cp = cache_path(f"name_{cui}")
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    url = f"{BASE_URL}/content/{VERSION}/CUI/{cui}"
    data = api_get(url, {}, api_key)
    name = data.get("result", {}).get("name", cui) if data else cui
    with open(cp, "w") as f:
        json.dump(name, f)
    return name


def resolve_source_atom_to_cuis(sab: str, code: str, api_key: str) -> list[str]:
    """Map a source-asserted identifier (e.g. SNOMEDCT_US/9468002) back to its owning UMLS CUI(s):"""
    cp = cache_path(f"resolve_{sab}_{code}")
    if cp.exists():
        with open(cp) as f:
            return json.load(f)

    url = f"{BASE_URL}/search/{VERSION}"
    params = {
        "string": code,
        "inputType": "sourceUi",
        "searchType": "exact",
        "sabs": sab,
    }
    data = api_get(url, params, api_key)
    cuis_found = []
    if data:
        results = data.get("result", {}).get("results", [])
        cuis_found = [r["ui"] for r in results if r.get("ui") and r["ui"] != "NONE"]

    with open(cp, "w") as f:
        json.dump(cuis_found, f)
    return cuis_found


def resolve_related_id(related_url: str, api_key: str) -> list[str]:
    """Turn a `relatedId` URL from /relations into a list of candidate CUIs."""
    if "/CUI/" in related_url:
        return [related_url.split("/CUI/")[-1].split("/")[0]]

    if "/source/" in related_url:
        tail = related_url.split("/source/")[-1]
        parts = tail.split("/")
        if len(parts) < 2:
            return []
        sab, code = parts[0], parts[1]
        return resolve_source_atom_to_cuis(sab, code, api_key)

    return []


def build_edges(cuis: set, api_key: str) -> list[dict]:
    print("\nFetching CUI names...")
    cui_names = {cui: get_cui_name(cui, api_key) for cui in tqdm(sorted(cuis))}

    print("\nFetching PAR/CHD relations for each CUI...")
    hierarchy_edges = []
    no_relations = 0
    resolved_via_search = 0
    unresolved = 0

    for cui in tqdm(sorted(cuis)):
        relations = get_relations(cui, api_key)
        if not relations:
            no_relations += 1
            continue

        for rel in relations:
            label = rel.get("relationLabel")
            if label not in ("PAR", "CHD"):
                continue

            source = rel.get("rootSource", "")
            related_url = rel.get("relatedId", "")
            was_direct_cui = "/CUI/" in related_url

            candidate_cuis = resolve_related_id(related_url, api_key)
            if not was_direct_cui:
                if candidate_cuis:
                    resolved_via_search += 1
                else:
                    unresolved += 1

            for other_cui in candidate_cuis:
                if other_cui == cui or other_cui not in cuis:
                    continue

                if label == "PAR":
                    child_cui, parent_cui = other_cui, cui
                else:  # CHD
                    child_cui, parent_cui = cui, other_cui

                hierarchy_edges.append({
                    "child_cui": child_cui,
                    "child_name": cui_names.get(child_cui, child_cui),
                    "parent_cui": parent_cui,
                    "parent_name": cui_names.get(parent_cui, rel.get("relatedIdName", "")),
                    "source": source,
                })

    def dedupe(edges):
        seen, out = set(), []
        for e in edges:
            k = (e["child_cui"], e["parent_cui"])
            if k not in seen:
                seen.add(k)
                out.append(e)
        return out

    hierarchy_edges = dedupe(hierarchy_edges)

    print("\nDone.")
    print(f"  Hierarchy (PAR/CHD) edges           : {len(hierarchy_edges)}")
    print(f"  CUIs with no PAR/CHD relations       : {no_relations}")
    print(f"  Source atoms resolved via search     : {resolved_via_search}")
    print(f"  Source atoms that failed to resolve  : {unresolved}")

    return hierarchy_edges


def save_csv(edges: list[dict], fields: list[str], output_csv: str):
    if not edges:
        print(f"No edges to save for {output_csv}")
        return
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(edges)
    print(f"Saved {len(edges)} edges → {output_csv}")


def print_summary(hierarchy_edges, cuis):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total CUIs in label set   : {len(cuis)}")

    h_cuis = {e["child_cui"] for e in hierarchy_edges} | {e["parent_cui"] for e in hierarchy_edges}

    print(f"\nHierarchy edges (PAR/CHD) : {len(hierarchy_edges)}")
    print(f"  CUIs covered            : {len(h_cuis)}")
    print(f"  CUIs fully isolated     : {len(cuis - h_cuis)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--train_csv", default=TRAIN_CSV)
    parser.add_argument("--mlb_path", default=MLB_PATH)
    args = parser.parse_args()

    cuis = load_label_cuis(args.train_csv, args.mlb_path)
    print(f"Label set: {len(cuis)} CUIs\n")

    hierarchy_edges = build_edges(cuis, args.api_key)

    save_csv(
        hierarchy_edges,
        ["child_cui", "child_name", "parent_cui", "parent_name", "source"],
        OUTPUT_HIERARCHY,
    )

    print_summary(hierarchy_edges, cuis)
