#!/bin/bash
# Train one CNN config on 4 GPUs. Submit from cnn/:
#   CONFIG=configs/backbones/resnet50_asl.yaml sbatch --job-name=resnet50_asl slurm/train.sbatch
# Resubmitting the same command resumes from <checkpoints>/last.pt.
#SBATCH --job-name=cnn_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

module load cuda/12.2
module load python/3.11.7
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

CONFIG="${CONFIG:-configs/base.yaml}"
echo "Using config: ${CONFIG}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

# Compute nodes are offline: weights come from scripts/download_weights.py.
export HF_HUB_OFFLINE=1
export TORCH_HOME="$WORK/Long-Tailed-Medical-Concept-Detection/model_cache/torch"
export HF_HOME="$WORK/Long-Tailed-Medical-Concept-Detection/model_cache/hf"

PYTHONHASHSEED=42 torchrun \
  --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  train.py --config "${CONFIG}"
