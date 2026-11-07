# Data preparation (Chapter 3)

## Splits and shards

The thesis uses the provided training and validation sets, with 7,005 training images held out as a test split by stratified re-splitting (Section 3.3): 90,359 train, 19,240 validation and 7,005 test images. `build_webdataset.py` expects these splits as `{train,valid,test}_concepts.csv` (columns `ID,CUIs`, CUIs separated by `;`), the matching `{train,valid,test}_captions.csv`, and an `images/` folder.

```bash
python build_webdataset.py --data-dir dev_concept --output-dir $WORK/imageclefmed2026_webdataset
cp dev_concept/cui_mapping.csv $WORK/imageclefmed2026_webdataset/
```

This writes 92 train, 20 validation and 8 test shards. Each sample is a JPEG and a JSON with its id, CUIs and caption.

## Label binarizer

```bash
python fit_label_binarizer.py --config ../cnn/configs/base.yaml
```

Fits a `MultiLabelBinarizer` on the training shards and saves it to `artifacts/mlb.pkl`. Every script loads this file, so a concept has the same index in every model (2,645 concepts appear in the training split).

## Exploratory data analysis

```bash
python eda.py --concepts dev_concept/concepts.csv --cui-mapping dev_concept/cui_mapping.csv --out-dir eda
```

Produces Tables 3.1-3.2 and Figures 3.2-3.3.

## Concept taxonomy (Section 3.5)

`taxonomy/` holds the taxonomy that groups the 2,646 CUIs into six ordered categories (Imaging Modality, Anatomy, Radiological Finding, Pathology, Device & Intervention, Clinical Context) and their subcategories. It was created with Claude Opus 4.8 and Gemini 3.1 Pro through the prompting process described in the thesis.

| File | Content |
|---|---|
| `concept_taxonomy.json` | categories and subcategories |
| `concept_classifications.json` | category and subcategory of every CUI |
| `concept_classification_counts.json` | number of concepts per category and subcategory |

## UMLS concept hierarchy (Section 4.2.3)

Builds the descendant matrices used by MCLoss. Run the steps in order from `umls_hierarchy/`. Step 1a needs a free [UMLS API key](https://uts.nlm.nih.gov/uts/) and step 1b the UMLS release files (`MRREL.RRF`, `MRCONSO.RRF`).

```bash
python fetch_api_relations.py --api_key KEY --train_csv train_concepts.csv --mlb_path mlb.pkl   # -> umls_hierarchy_api.csv
python parse_rrf_relations.py --mrrel_path MRREL.RRF --mrconso_path MRCONSO.RRF --mlb_path mlb.pkl   # -> umls_hierarchy_rrf.csv
python merge_relations.py                                                                    # -> umls_hierarchy_merged.csv
python build_descendant_matrix.py --mlb_path mlb.pkl        # -> umls_hierarchy.csv, umls_descendant_matrix.npz

# restricted hierarchies of Table 5.3
python restrict_anatomy.py --mlb_path mlb.pkl                                  # -> umls_descendant_matrix_anatomy.npz
python edge_cooccurrence.py --train_csv train_concepts.csv                     # -> edge_cooccurrence.csv
python restrict_live.py --mlb_path mlb.pkl --min_cooccurrence 1                # -> umls_descendant_matrix_live.npz
python restrict_live.py --mlb_path mlb.pkl --min_cooccurrence 5                # -> umls_descendant_matrix_live_5.npz
```

Copy the `.npz` files to `$WORK/Long-Tailed-Medical-Concept-Detection/artifacts/`, where the MCLoss configs expect them. The UMLS content itself is licensed by the NLM and is not included in this repository.
