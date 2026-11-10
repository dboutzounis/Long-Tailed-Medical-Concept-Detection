#!/bin/bash
# Expert analysis for every µMoE run in slurm/runs.tsv (Section 5.6). Submit from interpretability/mumoe/:
#   mkdir -p logs && sbatch slurm/run_all_ablations.sbatch
# The thresholds in runs.tsv are the ones selected on val by cnn/slurm/evaluate.sbatch.
# Each run uses its last (lowest val loss) best checkpoint.
#SBATCH --job-name=mumoe_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load cuda/12.2
module load python/3.11.7
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export TORCH_HOME="$WORK/Long-Tailed-Medical-Concept-Detection/model_cache/torch"

MANIFEST="${MANIFEST:-slurm/runs.tsv}"
OUT_DIR="${OUT_DIR:-$WORK/Long-Tailed-Medical-Concept-Detection/results/mumoe_ablation}"
METRIC="${METRIC:-ap}"   # ap (thesis) or f1
CUI_MAPPING="$WORK/imageclefmed2026_webdataset/cui_mapping.csv"
mkdir -p "$OUT_DIR"/{base_scores,reports,summaries}

checkpoint() {
  local dir
  dir=$(python -c "import os, yaml; print(os.path.expandvars(yaml.safe_load(open('$1'))['paths']['checkpoints']))")
  ls "$dir"/*__loss_*.pt | sort | tail -n 1
}

echo "=== 1/4 baseline per-concept scores ==="
while IFS=$'\t' read -r NAME CONFIG THR <&3; do
  [[ "$NAME" == name ]] && continue
  python analyze_experts.py --config "$CONFIG" --checkpoint "$(checkpoint "$CONFIG")" \
    --decision-threshold "$THR" --dump-base-f1 "$OUT_DIR/base_scores/$NAME.pt"
done 3< "$MANIFEST"

echo "=== 2/4 fixed concept set ==="
python build_concept_list.py --inputs "$OUT_DIR"/base_scores/*.pt \
  --output "$OUT_DIR/concepts_fixed.txt" \
  --metric "$METRIC" --min-f1 0.05 --min-ap 0.10 --min-support 30

echo "=== 3/4 routing statistics and full ablation ==="
while IFS=$'\t' read -r NAME CONFIG THR <&3; do
  [[ "$NAME" == name ]] && continue
  python analyze_experts.py --config "$CONFIG" --checkpoint "$(checkpoint "$CONFIG")" \
    --decision-threshold "$THR" \
    --concept-list "$OUT_DIR/concepts_fixed.txt" --metric "$METRIC" \
    --cui-mapping "$CUI_MAPPING" --load-mode top1 --min-enrich-count 10 --ablate-top-k -1 \
    --output "$OUT_DIR/reports/$NAME.csv" --summary-json "$OUT_DIR/summaries/$NAME.json"
done 3< "$MANIFEST"

echo "=== 4/4 summary table ==="
python summarize_ablations.py "$OUT_DIR"/summaries/*.json --output "$OUT_DIR/summary_table.csv"
