#!/bin/bash
# Select the threshold on val, then evaluate test once and write the predictions. Submit from cnn/:
#   CONFIG=configs/backbones/resnet50_asl.yaml sbatch --job-name=eval_resnet50_asl slurm/evaluate.sbatch
# CKPT_PATH defaults to the last (lowest val loss) best checkpoint of the run.
#SBATCH --job-name=cnn_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err

module load cuda/12.2
module load python/3.11.7
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

export HF_HUB_OFFLINE=1
export TORCH_HOME="$WORK/Long-Tailed-Medical-Concept-Detection/model_cache/torch"
export HF_HOME="$WORK/Long-Tailed-Medical-Concept-Detection/model_cache/hf"

CONFIG="${CONFIG:-configs/base.yaml}"
CKPT_DIR=$(python -c "import os, yaml; print(os.path.expandvars(yaml.safe_load(open('$CONFIG'))['paths']['checkpoints']))")
CKPT_PATH="${CKPT_PATH:-$(ls "$CKPT_DIR"/*__loss_*.pt | sort | tail -n 1)}"
echo "Using config: ${CONFIG}"
echo "Using checkpoint: ${CKPT_PATH}"

VAL_OUT=$(python evaluate.py --config "$CONFIG" --checkpoint "$CKPT_PATH" --split val)
echo "$VAL_OUT"
BEST_THRESHOLD=$(echo "$VAL_OUT" | awk '/Best threshold:/ {print $4}')
if [ -z "$BEST_THRESHOLD" ]; then
  echo "ERROR: could not parse the best threshold from the val output" >&2
  exit 1
fi

python evaluate.py --config "$CONFIG" --checkpoint "$CKPT_PATH" --split test \
  --threshold "$BEST_THRESHOLD" \
  --output "$(dirname "${CKPT_DIR%/}")/test_predictions.csv"
