#!/bin/bash
# Train and then evaluate the 8 µMoE heads with the CNN pipeline (Section 4.5.1).
#   bash slurm/train_all.sh          # submit training
#   bash slurm/train_all.sh eval     # after training: val threshold + test F1 (Table 5.14)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/../../cnn"
bash slurm/submit.sh "${1:-train}" "$HERE"/configs/*.yaml
