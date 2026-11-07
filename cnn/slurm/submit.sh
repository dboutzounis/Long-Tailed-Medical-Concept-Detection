#!/bin/bash
# Submit train or eval jobs for a set of configs. Run from cnn/:
#   bash slurm/submit.sh train configs/backbones/*.yaml
#   bash slurm/submit.sh eval  configs/backbones/*.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: submit.sh train|eval CONFIG...}"
shift
mkdir -p logs

for config in "$@"; do
  name=$(basename "$config" .yaml)
  case "$MODE" in
    train) CONFIG="$config" sbatch --job-name="$name" slurm/train.sbatch ;;
    eval)  CONFIG="$config" sbatch --job-name="$name" slurm/evaluate.sbatch ;;
    *) echo "unknown mode: $MODE" >&2; exit 1 ;;
  esac
done
