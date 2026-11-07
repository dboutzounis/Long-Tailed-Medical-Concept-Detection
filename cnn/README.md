# End-to-end CNNs (Sections 4.1-4.3, 5.2-5.4)

An ImageNet-pretrained CNN, GeM pooling and a linear head (or a µMoE head, see `interpretability/mumoe/`), trained end to end on four GPUs with AdamW (lr 1e-4), ReduceLROnPlateau and early stopping on validation loss. A global threshold is selected on validation and applied once to the test split.

| File | Purpose |
|---|---|
| `train.py` | DDP training; keeps the checkpoint with the lowest validation loss |
| `evaluate.py` | threshold sweep on val, predictions and metrics on test |
| `models.py` | backbone (torchvision or timm), GeM pooling, linear / CP / TR µMoE head |
| `dataset.py` | WebDataset loader |
| `asl_loss.py`, `mcloss.py` | Asymmetric Loss and MCLoss; `mcloss.build_criterion` selects the loss from the config |
| `mumoe_layer.py` | µMoE layer, vendored unmodified from [muMoE](https://github.com/james-oldfield/muMoE) |
| `generate_configs.py` | writes every config below from `configs/base.yaml` |
| `collect_results.py` | gathers the eval logs into one table (Table 5.4) |
| `backbone_agreement.py` | agreement and error correlation between the 28 backbone models (Table 5.5, Figure 5.1) |

## Configs

| Config | Thesis |
|---|---|
| `configs/base.yaml` | single-model baseline: EfficientNetV2-M, BCE (Table 5.1) |
| `configs/asl_ablation/gpos{0,1,2}.yaml` | ASL ablation (Table 5.2); `gpos0` is the selected configuration |
| `configs/mcloss/{full,anatomy,cooc1,cooc5}.yaml` | MCLoss with the full and restricted hierarchies (Table 5.3) |
| `configs/backbones/<backbone>_{bce,asl}.yaml` | 14 backbones x 2 losses (Table 5.4) |

To change a setting for every run, edit `configs/base.yaml` and run `python generate_configs.py`.

## Running

From `cnn/`, after `python ../scripts/download_weights.py` on a node with internet access:

```bash
bash slurm/submit.sh train configs/base.yaml configs/asl_ablation/*.yaml configs/mcloss/*.yaml
bash slurm/submit.sh train configs/backbones/*.yaml

# when training has finished
bash slurm/submit.sh eval configs/base.yaml configs/asl_ablation/*.yaml configs/mcloss/*.yaml
bash slurm/submit.sh eval configs/backbones/*.yaml
python collect_results.py --logs-dir logs --out-csv results.csv --out-md results.md

python backbone_agreement.py \
    --runs-dir $WORK/Long-Tailed-Medical-Concept-Detection/runs/cnn/backbones \
    --gt-csv test_concepts.csv --out-dir agreement
```

Each eval job selects the threshold on val, then writes `test_predictions.csv` next to the run's checkpoints and prints the test metrics. Resubmitting a training job resumes from `last.pt`. MCLoss needs the descendant matrices from `data/umls_hierarchy/`.
