#!/bin/bash
# Cache the frozen embeddings of all four encoders. Submit from vlm/:
#   mkdir -p logs && sbatch extract_features.sbatch
#SBATCH --job-name=extract_features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

set -euo pipefail
module load cuda/12.2
module load python/3.11.7
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CACHE="$WORK/Long-Tailed-Medical-Concept-Detection/model_cache"
ART="$WORK/Long-Tailed-Medical-Concept-Detection/artifacts"
for enc in biomedclip medsiglip medimageinsight rad-dino; do
  python extract_features.py --encoder "$enc" --cache-dir "$CACHE" \
    --output-dir "$ART/${enc//-/}_features"
done
