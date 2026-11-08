#!/bin/bash
# Cache the CLS token of every BiomedCLIP block, then train the 72 layer x category probes
# (Section 4.6). Submit from interpretability/layer_probing/:
#   mkdir -p logs && sbatch probe_hierarchy.sbatch
#SBATCH --job-name=probe_hierarchy
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe_hierarchy_%j.out
#SBATCH --error=logs/probe_hierarchy_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=YOUR_ACCOUNT

module load cuda/12.2
module load python/3.11.7
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$WORK/Long-Tailed-Medical-Concept-Detection/model_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ROOT="$WORK/Long-Tailed-Medical-Concept-Detection"
LAYER_DIR="$ROOT/artifacts/layer_features"
TAXONOMY="$ROOT/data/taxonomy/concept_classifications.json"
MLB_PATH="$ROOT/artifacts/mlb.pkl"
mkdir -p "$ROOT/results"

[ -f "$LAYER_DIR/test_layer_11_embs.pt" ] || python extract_layers.py \
  --config ../../vlm/common/base_config.yaml --cache-dir "$HF_HOME" --output-dir "$LAYER_DIR"

python probe_hierarchy.py --data-dir "$LAYER_DIR" --mlb-path "$MLB_PATH" \
  --json-path "$TAXONOMY" --output-csv "$ROOT/results/hierarchy_layer_f1.csv"
python plot_probe_performance.py --csv-path "$ROOT/results/hierarchy_layer_f1.csv" \
  --output-image "$ROOT/results/probing_f1_layers.png"
