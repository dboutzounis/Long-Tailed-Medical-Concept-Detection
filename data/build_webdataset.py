"""Pack the ImageCLEFmedical 2026 images and concepts into WebDataset shards (Section 3.6.2).

Each sample is a JPEG plus a JSON with its id, CUIs and caption. The number of shards per split
is rounded up to a multiple of the GPU count so every rank streams the same number of shards
(92 train, 20 val and 8 test shards for the thesis splits).

Expects in --data-dir: images/, {train,valid,test}_concepts.csv and {train,valid,test}_captions.csv.

Usage:
    python build_webdataset.py --data-dir dev_concept --output-dir imageclefmed2026_webdataset
"""
import argparse
import json
import math
import os

import pandas as pd
import webdataset as wds
from tqdm import tqdm


def create_balanced_shards(concept_csv, caption_csv, image_dir, output_pattern,
                           target_shard_size=1000, gpu_divisibility=4):
    df_concepts = pd.read_csv(concept_csv)
    df_captions = pd.read_csv(caption_csv)
    for df in (df_concepts, df_captions):
        df["ID"] = df["ID"].astype(str).str.replace(r"\.jpg$", "", regex=True)
    df = pd.merge(df_concepts, df_captions, on="ID", how="inner")
    total = len(df)
    if total == 0:
        return

    num_shards = int(math.ceil(total / target_shard_size / gpu_divisibility) * gpu_divisibility)
    per_shard, remainder = divmod(total, num_shards)
    capacities = [per_shard + (1 if i < remainder else 0) for i in range(num_shards)]
    print(f"{output_pattern}: {total} samples in {num_shards} shards")

    rows = df.iterrows()
    for shard_idx, capacity in enumerate(capacities):
        with wds.TarWriter(output_pattern % shard_idx) as sink:
            for _ in tqdm(range(capacity), desc=f"shard {shard_idx + 1}/{num_shards}"):
                _, row = next(rows)
                img_path = os.path.join(image_dir, f"{row['ID']}.jpg")
                if not os.path.exists(img_path):
                    continue
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                # the ';'-separated CUI string is stored as a single list entry
                cuis = [c.strip() for c in str(row["CUIs"]).split(",")] if pd.notna(row["CUIs"]) else []
                meta = {"id": row["ID"], "cuis": cuis,
                        "caption": str(row["Caption"]) if pd.notna(row["Caption"]) else ""}
                sink.write({"__key__": row["ID"], "jpg": image_bytes,
                            "json": json.dumps(meta, indent=2).encode("utf-8")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--output-dir", default="imageclefmed2026_webdataset")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--gpus", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for split, prefix in (("train", "train"), ("valid", "val"), ("test", "test")):
        create_balanced_shards(
            concept_csv=os.path.join(args.data_dir, f"{split}_concepts.csv"),
            caption_csv=os.path.join(args.data_dir, f"{split}_captions.csv"),
            image_dir=os.path.join(args.data_dir, "images"),
            output_pattern=os.path.join(args.output_dir, f"{prefix}-shards-%06d.tar"),
            target_shard_size=args.shard_size,
            gpu_divisibility=args.gpus,
        )


if __name__ == "__main__":
    main()
