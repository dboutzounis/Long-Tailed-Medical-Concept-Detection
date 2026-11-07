#!/bin/bash
# KNN retrieval for every single encoder and every 2-, 3- and 4-encoder fusion
# (Tables 5.7-5.8 on frozen features, Tables 5.12-5.13 on LoRA-adapted features).
# Phase 1 selects K, rho and the fusion weights on val; phase 2 applies them once on test.
# Submit from vlm/knn/:
#   mkdir -p logs && sbatch knn_fusion.sbatch
#   sbatch --export=ALL,FEATURES=lora knn_fusion.sbatch    # LoRA-adapted features
#   sbatch --export=ALL,RUN_VAL=0 knn_fusion.sbatch        # test phase only
#SBATCH --job-name=knn_fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=logs/knn_fusion_%j.out
#SBATCH --error=logs/knn_fusion_%j.err

set -u
module load cuda/12.2
module load python/3.11.7
source "$WORK/Long-Tailed-Medical-Concept-Detection/venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

FEATURES="${FEATURES:-frozen}"   # frozen | lora
ART="$WORK/Long-Tailed-Medical-Concept-Detection/artifacts"
MLB_PATH="$ART/mlb.pkl"
FEAT_ROOT="$ART"; [ "$FEATURES" = "lora" ] && FEAT_ROOT="$ART/lora"
RES_DIR="$WORK/Long-Tailed-Medical-Concept-Detection/results/knn_$FEATURES"
mkdir -p "$RES_DIR/predictions"
VAL_RESULTS="$RES_DIR/val_best.tsv"
TEST_RESULTS="$RES_DIR/test_results.tsv"
RUN_VAL="${RUN_VAL:-1}"
RUN_TEST="${RUN_TEST:-1}"

RATIO_GRID="0.20 0.25 0.30 0.35 0.40 0.45 0.50"
K_GRID="1 3 5 10 20 50 100 200"

NAMES=(medimageinsight medsiglip biomedclip raddino)
DIRS=()
for n in "${NAMES[@]}"; do DIRS+=("$FEAT_ROOT/${n}_features"); done
N=${#NAMES[@]}
BEST_K=(0 0 0 0)  # filled by the single-encoder sweeps, reused for fusion

# Phase 1: hyperparameter sweeps on val
if [ "$RUN_VAL" = "1" ]; then
  echo "############ PHASE 1: VALIDATION SWEEPS ############"
  printf "subset\tencoders\tks\tweights\tratio\tval_f1\n" > "$VAL_RESULTS"

  # single encoders: joint K x ratio sweep, giving each encoder its own K
  for ((i=0; i<N; i++)); do
    name="${NAMES[$i]}"; dir="${DIRS[$i]}"
    echo ""; echo "=== [val][single] $name ==="
    if [ ! -f "$dir/val_embs.pt" ]; then
      echo "  SKIP: $dir/val_embs.pt not found"; continue
    fi
    log="$RES_DIR/val_single_${name}.log"
    python knn_retrieve.py --data-dir "$dir" --split val \
      --k-grid $K_GRID --ratio-grid $RATIO_GRID --diagnose 2>&1 | tee "$log"

    k=$(grep -oP 'BEST cell: K=\K[0-9]+' "$log" | tail -1)
    r=$(grep -oP 'BEST cell: K=[0-9]+\s+ratio=\K[0-9.]+' "$log" | tail -1)
    f=$(grep -oP 'BEST cell:.*F1=\K[0-9.]+' "$log" | tail -1)
    if [ -n "${k:-}" ]; then
      BEST_K[$i]=$k
      printf "single\t%s\t%s\t1.00\t%s\t%s\n" "$name" "$k" "$r" "$f" >> "$VAL_RESULTS"
      echo "  -> best K=$k ratio=$r F1=$f"
    else
      echo "  WARNING: could not parse a winner for $name"
    fi
  done

  echo ""; echo "per-encoder best K: ${BEST_K[*]}"

  # every subset of 2, 3 and 4 encoders (bitmask over NAMES), first member is primary
  for ((want=2; want<=N; want++)); do
   for ((mask=1; mask<(1<<N); mask++)); do
    idxs=(); for ((i=0; i<N; i++)); do
      (( (mask>>i) & 1 )) && idxs+=($i)
    done
    size=${#idxs[@]}
    [ "$size" -ne "$want" ] && continue

    primary=${idxs[0]}
    pdir="${DIRS[$primary]}"
    tag="${NAMES[$primary]}"
    fuse_dirs=""; ks="${BEST_K[$primary]}"
    for ((j=1; j<size; j++)); do
      m=${idxs[$j]}
      fuse_dirs="$fuse_dirs ${DIRS[$m]}"
      ks="$ks ${BEST_K[$m]}"
      tag="${tag}+${NAMES[$m]}"
    done

    # coarser weight grid for 4 encoders (286 combinations at step 0.1)
    step=0.1; [ "$size" -ge 4 ] && step=0.2

    echo ""; echo "=== [val][${size}-way] $tag  (K=$ks, step=$step) ==="
    if [[ "$ks" =~ (^| )0( |$) ]]; then
      echo "  SKIP: a member has no best K (its single run failed)"; continue
    fi
    log="$RES_DIR/val_fuse_${tag}.log"
    python knn_retrieve.py --data-dir "$pdir" --fuse-dir $fuse_dirs \
      --split val --fuse-k $ks \
      --fuse-weight-step $step --ratio-grid $RATIO_GRID 2>&1 | tee "$log"

    w=$(grep -oP '^BEST: w=\[\K[^]]+' "$log" | tail -1)
    r=$(grep -oP '^BEST: w=\[[^]]+\] ratio=\K[0-9.]+' "$log" | tail -1)
    f=$(grep -oP '^BEST: .*F1=\K[0-9.]+' "$log" | tail -1)
    if [ -n "${w:-}" ]; then
      printf "%d-way\t%s\t%s\t%s\t%s\t%s\n" "$size" "$tag" "$ks" "$w" "$r" "$f" \
        >> "$VAL_RESULTS"
      echo "  -> best w=[$w] ratio=$r F1=$f"
    else
      echo "  WARNING: could not parse a winner for $tag"
    fi
   done
  done

  echo ""; echo "=== validation summary ==="
  column -t -s $'\t' "$VAL_RESULTS" || cat "$VAL_RESULTS"
fi

# Phase 2: test with the val-selected hyperparameters
if [ "$RUN_TEST" = "1" ]; then
  echo ""; echo "############ PHASE 2: TEST ############"
  if [ ! -f "$VAL_RESULTS" ]; then
    echo "ERROR: $VAL_RESULTS missing - run phase 1 first."; exit 1
  fi
  printf "subset\tencoders\tks\tweights\tratio\tval_f1\ttest_line\n" > "$TEST_RESULTS"

  dir_of() {
    for ((i=0; i<N; i++)); do
      [ "${NAMES[$i]}" = "$1" ] && { echo "${DIRS[$i]}"; return; }
    done
  }

  tail -n +2 "$VAL_RESULTS" | while IFS=$'\t' read -r subset encs ks weights ratio vf1; do
    [ -z "${subset:-}" ] && continue
    echo ""; echo "=== [test] $subset  $encs  (K=$ks, w=[$weights], ratio=$ratio) ==="
    log="$RES_DIR/test_${encs}.log"
    out="$RES_DIR/predictions/${encs//+/_}.csv"

    if [ "$subset" = "single" ]; then
      python knn_retrieve.py --data-dir "$(dir_of "$encs")" --split test \
        --k "$ks" --ratio "$ratio" --diagnose \
        --mlb-path "$MLB_PATH" --output "$out" 2>&1 | tee "$log"
    else
      IFS='+' read -ra members <<< "$encs"
      pdir="$(dir_of "${members[0]}")"
      fuse_dirs=""
      for ((j=1; j<${#members[@]}; j++)); do
        fuse_dirs="$fuse_dirs $(dir_of "${members[$j]}")"
      done
      python knn_retrieve.py --data-dir "$pdir" --fuse-dir $fuse_dirs \
        --split test --fuse-k $ks \
        --fuse-weights $weights --ratio-grid "$ratio" \
        --mlb-path "$MLB_PATH" --output "$out" 2>&1 | tee "$log"
    fi

    tline=$(grep -oP '^\[(test|fusion).*' "$log" | tail -1)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$subset" "$encs" "$ks" "$weights" "$ratio" "$vf1" "${tline:-NA}" \
      >> "$TEST_RESULTS"
  done

  echo ""; echo "=== test summary (val-selected configs) ==="
  column -t -s $'\t' "$TEST_RESULTS" || cat "$TEST_RESULTS"
  echo "tables:      $VAL_RESULTS  |  $TEST_RESULTS"
fi

echo ""; echo "== done =="
