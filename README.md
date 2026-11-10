# Long-Tailed Medical Concept Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=FFD43B" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.5">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/NVIDIA-A100-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA A100">
  <img src="https://img.shields.io/badge/HPC-Leonardo%20%7C%20SLURM-005A9C?style=for-the-badge&logo=linux&logoColor=white" alt="Leonardo, SLURM">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge" alt="MIT License"></a>
</p>

Code for the BSc thesis [**"A Representation Learning and Interpretability Approach for Long-Tailed Medical Concept Detection"**](https://pergamos.lib.uoa.gr/uoa/dl/object/5452504) by D. N. Boutzounis and S. Stavrou, Department of Informatics and Telecommunications, National and Kapodistrian University of Athens (2026), supervised by Assoc. Prof. Yannis Panagakis.

The task is the [ImageCLEFmedical 2026 Concept Detection](https://www.imageclef.org/) task: predict the set of UMLS concepts (CUIs) of a radiology image from an extended ROCOv2 collection. The concepts follow a heavy long-tailed distribution, and models are scored with sample-averaged F1.

Our competition system (2nd place, F1 0.5781) is in [dboutzounis/imageclef2026](https://github.com/dboutzounis/imageclef2026). This repository contains the thesis work that builds on it, in two parts:

- **Performance:** CNN baselines, loss functions (ASL, hierarchy-aware MCLoss), a 14-backbone comparison, and frozen or LoRA-adapted biomedical vision-language encoders (zero-shot, label priors, nearest-neighbor retrieval and fusion, trained heads).
- **Interpretability:** µMoE classification heads and expert ablation, layer-wise probing and a PCA ablation of BiomedCLIP, and Grad-ECLIP attribution maps with a faithfulness evaluation.

## Main results

Sample-averaged F1 on our held-out test split (7,005 images). These scores are not comparable to the competition leaderboard.

| Method                                                 | Test F1    |
| ------------------------------------------------------ | ---------- |
| EfficientNetV2-M, linear head, BCE (baseline)          | 0.5882     |
| EfficientNetV2-M, ASL                                  | 0.5928     |
| Best CNN backbone (ConvNeXt V2-Tiny, ASL)              | 0.5935     |
| Zero-shot with label priors (MedSigLIP, no training)   | 0.4675     |
| Nearest-neighbor retrieval, frozen MedImageInsight     | 0.5996     |
| Retrieval fusion of three frozen encoders              | 0.6012     |
| MLP on concatenated frozen embeddings (three encoders) | 0.6057     |
| LoRA-adapted MedImageInsight with a linear head        | 0.6073     |
| Linear head on concatenated LoRA-adapted embeddings    | **0.6085** |

## Repository structure

```
data/                 dataset preparation (Chapter 3)
  build_webdataset.py     pack images and concepts into WebDataset shards
  fit_label_binarizer.py  fixed concept order shared by every model
  eda.py                  exploratory data analysis (Section 3.4)
  taxonomy/               concept taxonomy and per-concept categories (Section 3.5)
  umls_hierarchy/         UMLS concept hierarchy for MCLoss (Section 4.2.3)
cnn/                  end-to-end CNNs (Sections 4.1-4.3, 5.2-5.4)
vlm/                  frozen and LoRA-adapted vision-language encoders (Sections 4.4, 5.5)
  common/                 encoder backends and shared helpers
  zero_shot/  knn/  frozen_heads/  lora/
interpretability/     (Sections 4.5-4.8, 5.6-5.9)
  mumoe/  layer_probing/  dim_ablation/  gradeclip/
scripts/              download of the pretrained CNN weights
```

Each top-level folder has its own README with the commands to run.

| Thesis section                              | Code                                                       |
| ------------------------------------------- | ---------------------------------------------------------- |
| 3.4 Exploratory data analysis               | `data/eda.py`                                              |
| 3.5 Concept taxonomy                        | `data/taxonomy/`                                           |
| 3.6 Label binarization and data pipeline    | `data/fit_label_binarizer.py`, `data/build_webdataset.py`  |
| 4.1 Single-model baseline                   | `cnn/` with `configs/base.yaml`                            |
| 4.2 Loss functions (ASL, MCLoss)            | `cnn/asl_loss.py`, `cnn/mcloss.py`, `data/umls_hierarchy/` |
| 4.3 Backbone exploration                    | `cnn/configs/backbones/`, `cnn/backbone_agreement.py`      |
| 4.4.1 Feature extraction                    | `vlm/extract_features.py`                                  |
| 4.4.2-4.4.3 Zero-shot, label priors         | `vlm/zero_shot/`                                           |
| 4.4.4 Nearest-neighbor retrieval and fusion | `vlm/knn/`                                                 |
| 4.4.5 Trained heads on frozen encoders      | `vlm/frozen_heads/`                                        |
| 4.4.6 LoRA adaptation                       | `vlm/lora/`                                                |
| 4.5 Interpretable heads with µMoE           | `interpretability/mumoe/`                                  |
| 4.6 Layer-wise probing                      | `interpretability/layer_probing/`                          |
| 4.7 Dimensionality ablation                 | `interpretability/dim_ablation/`                           |
| 4.8 Grad-ECLIP                              | `interpretability/gradeclip/`                              |

## Setup

The experiments ran on the Booster partition of Leonardo (CINECA) with SLURM, four A100 GPUs per node and Python 3.11. Compute nodes have no internet access, so all weights are downloaded beforehand.

```bash
cd $WORK
git clone https://github.com/dboutzounis/Long-Tailed-Medical-Concept-Detection.git
cd Long-Tailed-Medical-Concept-Detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Path conventions used by the configs and SLURM scripts:

| Path                                                                        | Content                                                                |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `$WORK/imageclefmed2026_webdataset/`                                        | WebDataset shards and `cui_mapping.csv`                                |
| `$WORK/Long-Tailed-Medical-Concept-Detection/artifacts/`                    | label binarizer, descendant matrices, cached embeddings, trained heads |
| `$WORK/Long-Tailed-Medical-Concept-Detection/runs/`                         | CNN and LoRA checkpoints and logs                                      |
| `$WORK/Long-Tailed-Medical-Concept-Detection/results/`                      | tables, predictions and figures                                        |
| `$WORK/Long-Tailed-Medical-Concept-Detection/model_cache/`                  | Hugging Face and torchvision weights                                   |
| `$WORK/Long-Tailed-Medical-Concept-Detection/third_party/MedImageInsights/` | community port of MedImageInsight                                      |

These folders are git-ignored. Replace `YOUR_ACCOUNT` in the `#SBATCH --account` lines with your own project account.

Pretrained weights (on a node with internet access):

```bash
python scripts/download_weights.py                    # the 14 CNN backbones
export HF_HOME=$WORK/Long-Tailed-Medical-Concept-Detection/model_cache
huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
huggingface-cli download google/medsiglip-448          # gated: accept the license on Hugging Face first
huggingface-cli download microsoft/rad-dino
git clone https://huggingface.co/lion-ai/MedImageInsights third_party/MedImageInsights
```

## Data

The ImageCLEFmedical 2026 Concept Detection dataset is not publicly available; it is distributed to the participants of the task. The thesis uses the provided training (97,364) and validation (19,240) images and holds out 7,005 training images as a test split by stratified re-splitting, which gives 90,359 / 19,240 / 7,005 images. See [`data/`](data/README.md).

## Reproducing the thesis

1. `data/`: build the shards, fit the label binarizer and, for MCLoss, build the UMLS hierarchy.
2. `cnn/`: train and evaluate the baseline, the ASL ablation, MCLoss and the backbone sweep.
3. `vlm/`: extract embeddings, then run zero-shot, retrieval, trained heads, and LoRA (followed by heads and retrieval on the adapted embeddings).
4. `interpretability/`: µMoE heads and ablation, layer-wise probing, the PCA ablation and Grad-ECLIP.

Every data-dependent choice (thresholds, loss hyperparameters, fusion weights, number of neighbors) is selected on the validation split, and the test split is evaluated once. All runs use seed 42.

## Acknowledgements

Third-party code used here: the µMoE layer of [Oldfield et al.](https://github.com/james-oldfield/muMoE) (vendored in `cnn/mumoe_layer.py`), and ports of [ASL](https://github.com/Alibaba-MIIL/ASL), [C-HMCNN / MCLoss](https://github.com/EGiunchiglia/C-HMCNN) and [Grad-ECLIP](https://github.com/Cyang-Zhao/Grad-Eclip). Encoders: [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), [MedSigLIP](https://huggingface.co/google/medsiglip-448), [MedImageInsight](https://huggingface.co/lion-ai/MedImageInsights) and [Rad-DINO](https://huggingface.co/microsoft/rad-dino).

## License

Released under the [MIT License](LICENSE). The vendored µMoE layer and the encoder weights keep their own licenses.
