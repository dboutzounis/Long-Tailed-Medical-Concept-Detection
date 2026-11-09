# Biomedical vision-language encoders (Sections 4.4, 5.5)

Four pretrained encoders, used with little or no task-specific training: BiomedCLIP, MedSigLIP and MedImageInsight (vision-language) and Rad-DINO (image only). The methods are ordered by how much supervision they use: zero-shot, label priors, nearest-neighbor retrieval, trained heads on frozen embeddings, and LoRA.

| Folder / file | Purpose |
|---|---|
| `common/` | encoder backends, shared helpers, the ordered data loader, `config.yaml` (encoder settings) and `base_config.yaml` (shards, label binarizer) |
| `extract_features.py` | caches the L2-normalized embeddings of every split for one encoder (Section 4.4.1) |
| `zero_shot/` | zero-shot classification and label priors (Table 5.6); text embeddings for Grad-ECLIP |
| `knn/` | nearest-neighbor retrieval and score-level fusion (Tables 5.7-5.8, 5.12-5.13) |
| `frozen_heads/` | ridge, linear, linear with prior bias, cosine and MLP heads (Tables 5.9, 5.11) |
| `lora/` | LoRA fine-tuning of the encoders and extraction of the adapted embeddings (Tables 5.10-5.11) |

The scripts read the cached embeddings from `artifacts/<encoder>_features/`. All caches share the same row order, which the fusion code checks by comparing label matrices.

## Running

Submit each job from the folder that contains it (create `logs/` first).

```bash
# 1. embeddings of the four encoders (from vlm/)
sbatch extract_features.sbatch

# 2. zero-shot and label priors (from vlm/zero_shot/), then the text embeddings used by Grad-ECLIP
for m in biomedclip medsiglip medimageinsight; do sbatch --export=ALL,MODEL=$m zeroshot.sbatch; done
python extract_text_embs.py --encoder biomedclip            # GPU node

# 3. retrieval: every single encoder and every fusion, selected on val, tested once (from vlm/knn/)
sbatch knn_fusion.sbatch

# 4. trained heads on frozen embeddings (from vlm/frozen_heads/)
sbatch run_frozen_heads.sbatch single
sbatch run_frozen_heads.sbatch concat

# 5. LoRA (from vlm/lora/); the head starts from the frozen linear head of step 4
for m in biomedclip medsiglip medimageinsight raddino; do sbatch run_lora_train.sbatch $m; done
for m in biomedclip medsiglip medimageinsight raddino; do sbatch run_lora_extract.sbatch $m; done

# 6. heads and retrieval on the LoRA-adapted embeddings
LORA=$WORK/Long-Tailed-Medical-Concept-Detection/artifacts/lora
(cd frozen_heads && sbatch --export=ALL,FEATURES_ROOT=$LORA,HEADS="ridge linear mlp" run_frozen_heads.sbatch single)
(cd frozen_heads && sbatch --export=ALL,FEATURES_ROOT=$LORA run_frozen_heads.sbatch concat)
(cd knn && sbatch --export=ALL,FEATURES=lora knn_fusion.sbatch)
```

`python frozen_heads/summarize_frozen_heads.py` prints the table of all heads and marks the MLP width with the best validation F1, the one reported in the thesis.
