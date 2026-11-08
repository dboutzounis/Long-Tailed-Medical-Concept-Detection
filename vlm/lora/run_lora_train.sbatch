#!/bin/bash
#SBATCH --job-name=lora_train
#SBATCH --partition=boost_usr_prod
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lora_train_%j.out
# Train LoRA + linear head for one encoder on 4 GPUs. Submit from vlm/lora/:
#   mkdir -p logs && sbatch run_lora_train.sbatch medsiglip
# Arguments after the model name go to train_lora.py.

set -euo pipefail
MODEL=${1:?usage: sbatch run_lora_train.sbatch MODEL [extra args]}
shift
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=4

torchrun --standalone --nproc_per_node=4 train_lora.py --config config.yaml --model "$MODEL" "$@"
