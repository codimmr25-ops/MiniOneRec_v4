#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:-E1}"

export HF_HOME="${HF_HOME:-/root/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/hf_cache/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.cache/pip}"

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/MiniOneRec_v2}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$REPO_ROOT/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/research}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/research}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

case "$EXP_ID" in
  E1|e1)
    EXP_NAME="${EXP_NAME:-$OUTPUT_ROOT/E1_sft/final_checkpoint}"
    ;;
  E2|e2)
    EXP_NAME="${EXP_NAME:-$OUTPUT_ROOT/E2_dpo/final_checkpoint}"
    ;;
  E3|e3)
    EXP_NAME="${EXP_NAME:-$OUTPUT_ROOT/E3_grpo/final_checkpoint}"
    ;;
  E4|e4)
    EXP_NAME="${EXP_NAME:-$OUTPUT_ROOT/E4_dpo_grpo/final_checkpoint}"
    ;;
  *)
    EXP_NAME="${EXP_NAME:-$EXP_ID}"
    ;;
esac

TEST_FILE="${TEST_FILE:-$BUNDLE_ROOT/converted/rqvae/test/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
INFO_FILE="${INFO_FILE:-$BUNDLE_ROOT/converted/rqvae/info/Industrial_and_Scientific_5_2016-10-2018-11.txt}"
CUDA_IDS="${CUDA_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_BEAMS="${NUM_BEAMS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TEMP_DIR="${TEMP_DIR:-$OUTPUT_ROOT/temp_eval_${EXP_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_ROOT/${EXP_ID}}"

mkdir -p "$TEMP_DIR" "$OUTPUT_DIR"
cd "$REPO_ROOT"

python "$REPO_ROOT/split.py" \
  --input_path "$TEST_FILE" \
  --output_path "$TEMP_DIR" \
  --cuda_list "$CUDA_IDS"

IFS=',' read -r -a gpu_ids <<< "$CUDA_IDS"
for gpu_id in "${gpu_ids[@]}"; do
  if [[ -f "$TEMP_DIR/${gpu_id}.csv" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" python -u "$REPO_ROOT/evaluate.py" \
      --base_model "$EXP_NAME" \
      --info_file "$INFO_FILE" \
      --category "$CATEGORY" \
      --test_data_path "$TEMP_DIR/${gpu_id}.csv" \
      --result_json_data "$TEMP_DIR/${gpu_id}.json" \
      --batch_size "$BATCH_SIZE" \
      --num_beams "$NUM_BEAMS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --length_penalty 0.0 &
  fi
done
wait

actual_cuda_list=$(ls "$TEMP_DIR"/*.json 2>/dev/null | sed 's#^.*/##' | sed 's/\.json$//' | tr '\n' ',' | sed 's/,$//')
if [[ -z "$actual_cuda_list" ]]; then
  echo "No evaluation result shards were generated." >&2
  exit 1
fi

python "$REPO_ROOT/merge.py" \
  --input_path "$TEMP_DIR" \
  --output_path "$OUTPUT_DIR/final_result_${CATEGORY}.json" \
  --cuda_list "$actual_cuda_list"

python "$REPO_ROOT/calc.py" \
  --path "$OUTPUT_DIR/final_result_${CATEGORY}.json" \
  --item_path "$INFO_FILE"
