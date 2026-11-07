#!/bin/bash
# Zero-shot classification for one encoder. Submit from vlm/zero_shot/:
#   mkdir -p logs && sbatch --export=ALL,MODEL=medsiglip zeroshot.sbatch
#SBATCH --job-name=zeroshot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=logs/zeroshot_%j.out
#SBATCH --error=logs/zeroshot_%j.err
set -euo pipefail
module load cuda/12.2
module load python/3.11.7
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$WORK/Long-Tailed-Medical-Concept-Detection/model_cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

[ -f "$WORK/Long-Tailed-Medical-Concept-Detection/results/zero_shot/train_stats.pt" ] || python train_stats.py
python zeroshot.py --model "${MODEL}"
