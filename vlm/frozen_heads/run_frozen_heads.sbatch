#!/bin/bash
# Heads on frozen (or LoRA-adapted) embeddings. Submit from vlm/frozen_heads/:
#   mkdir -p logs
#   sbatch run_frozen_heads.sbatch single   # ridge, linear, linear_prior, cosine, MLP x3 per encoder
#   sbatch run_frozen_heads.sbatch concat   # ridge, linear, MLP x3 on concat3 / concat4
# LoRA-adapted features (Table 5.11): ridge, linear and MLP on the LoRA feature folder:
#   sbatch --export=ALL,FEATURES_ROOT=$WORK/Long-Tailed-Medical-Concept-Detection/artifacts/lora,HEADS="ridge linear mlp" run_frozen_heads.sbatch single
#SBATCH --job-name=frozen_heads
#SBATCH --partition=boost_usr_prod
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/frozen_heads_%x_%j.out

set -uo pipefail
PHASE=${1:?usage: sbatch run_frozen_heads.sbatch single|concat}
FEATURES_ROOT="${FEATURES_ROOT:-$WORK/Long-Tailed-Medical-Concept-Detection/artifacts}"
OUT_ROOT="$FEATURES_ROOT/frozen_heads"
HEADS="${HEADS:-ridge linear linear_prior cosine mlp}"
TAU_INIT=10
MLP_WIDTHS=(512 1024 2048)

source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=8  # fixed thread count keeps the ridge solve reproducible
export MKL_NUM_THREADS=8

FAILED=()
run() {
  echo ">>> python train_frozen_heads.py $*"
  python train_frozen_heads.py --features-root "$FEATURES_ROOT" --out-root "$OUT_ROOT" "$@" || FAILED+=("$*")
}

case "$PHASE" in
  single)
    for enc in biomedclip medsiglip medimageinsight raddino; do
      for head in $HEADS; do
        case "$head" in
          cosine) run --encoder "$enc" --head cosine --tau-init "$TAU_INIT" ;;
          mlp) for h in "${MLP_WIDTHS[@]}"; do run --encoder "$enc" --head mlp --hidden "$h"; done ;;
          *) run --encoder "$enc" --head "$head" ;;
        esac
      done
    done
    ;;
  concat)
    for grp in concat3 concat4; do
      run --encoder "$grp" --head ridge
      run --encoder "$grp" --head linear
      for h in "${MLP_WIDTHS[@]}"; do
        run --encoder "$grp" --head mlp --hidden "$h"
      done
    done
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 1 ;;
esac

python summarize_frozen_heads.py --out-root "$OUT_ROOT"
if (( ${#FAILED[@]} )); then
  printf 'FAILED: %s\n' "${FAILED[@]}"
  exit 1
fi
