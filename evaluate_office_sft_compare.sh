#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec}"
CONDA_ENV="${CONDA_ENV:-/mnt/bms_afs/miniconda3/envs/minionerec312}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/hf_cache}"

export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export CATEGORY="${CATEGORY:-Office_Products}"
export TEST_FILE="${TEST_FILE:-$REPO_ROOT/output/office_rqkmeans_plus/data/test/Office_Products_5_2016-10-2018-11.csv}"
export INFO_FILE="${INFO_FILE:-$REPO_ROOT/output/office_rqkmeans_plus/data/info/Office_Products_5_2016-10-2018-11.txt}"
export RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/office_rqkmeans_plus/sft}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/office_rqkmeans_plus/eval}"
export CUDA_IDS="${CUDA_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_BEAMS="${NUM_BEAMS:-10}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

source /mnt/bms_afs/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT" "$RESULTS_ROOT"

MODES=(
  original_full
  original_freeze_new_tokens
  new_lora
)

for mode in "${MODES[@]}"; do
  case "$mode" in
    original_full|original_freeze_new_tokens)
      export EXP_NAME="$REPO_ROOT/output/office_rqkmeans_plus/sft/$mode/final_checkpoint"
      ;;
    new_lora)
      export EXP_NAME="$REPO_ROOT/output/office_rqkmeans_plus/sft/$mode/final_checkpoint"
      ;;
  esac

  export TEMP_DIR="$OUTPUT_ROOT/temp_eval_$mode"
  export OUTPUT_DIR="$RESULTS_ROOT/$mode"

  rm -rf "$TEMP_DIR"
  mkdir -p "$TEMP_DIR" "$OUTPUT_DIR"

  echo "[$(date)] start eval mode=$mode host=$(hostname)"
  echo "model=$EXP_NAME"
  echo "test=$TEST_FILE"
  echo "info=$INFO_FILE"
  echo "output=$OUTPUT_DIR"
  echo "cuda=$CUDA_IDS batch=$BATCH_SIZE beams=$NUM_BEAMS max_new_tokens=$MAX_NEW_TOKENS"

  bash "$REPO_ROOT/evaluate.sh" "$mode"

  echo "[$(date)] done eval mode=$mode"
done

echo "[$(date)] all Office_Products rqkmeans_plus SFT evaluations finished"
