#!/bin/bash
#SBATCH --job-name=lora_extract
#SBATCH --partition=boost_usr_prod
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lora_extract_%j.out
# Extract the LoRA-adapted embeddings of one encoder (LoRA merged into the weights). Submit from vlm/lora/:
#   sbatch run_lora_extract.sbatch medsiglip
# Arguments after the model name go to extract_lora_features.py.

set -euo pipefail
MODEL=${1:?usage: sbatch run_lora_extract.sbatch MODEL [extra args]}
shift
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

python extract_lora_features.py --config config.yaml --model "$MODEL" "$@"
