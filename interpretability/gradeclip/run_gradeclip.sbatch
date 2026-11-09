#!/bin/bash
# Grad-ECLIP maps, faithfulness curves and figures (Section 5.9). Submit from interpretability/gradeclip/:
#   mkdir -p logs && sbatch run_gradeclip.sbatch
#SBATCH --job-name=gradeclip
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gradeclip_%j.out
#SBATCH --error=logs/gradeclip_%j.err

set -euo pipefail
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python run_gradeclip.py --config config.yaml "$@"
python plot_gradeclip.py --out-dir "$WORK/Long-Tailed-Medical-Concept-Detection/results/gradeclip/biomedclip"
