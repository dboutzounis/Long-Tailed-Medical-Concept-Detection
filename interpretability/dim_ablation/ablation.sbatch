#!/bin/bash
# PCA dimensionality ablation of the BiomedCLIP embedding (Section 4.7). Needs the layer
# features of layer_probing/. Submit from interpretability/dim_ablation/:
#   mkdir -p logs && sbatch ablation.sbatch
#SBATCH --job-name=dim_ablation
#SBATCH --time=01:00:00
#SBATCH --output=logs/dim_ablation_%j.out
#SBATCH --error=logs/dim_ablation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=YOUR_ACCOUNT

module load cuda/12.2
module load python/3.11.7
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$WORK/Long-Tailed-Medical-Concept-Detection/model_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ROOT="$WORK/Long-Tailed-Medical-Concept-Detection"
LAYER_DIR="$ROOT/artifacts/layer_features"
TAXONOMY="$ROOT/data/taxonomy/concept_classifications.json"
MLB_PATH="$ROOT/artifacts/mlb.pkl"
mkdir -p "$ROOT/results"

python pca_ablation.py --data-dir "$LAYER_DIR" --mlb-path "$MLB_PATH" --json-path "$TAXONOMY" \
  --output-prefix "$ROOT/results/dim_ablation" \
  --min-samples 0 --batch-size 2048 --max-epochs 100 --patience 5
