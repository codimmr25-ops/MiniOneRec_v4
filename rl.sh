#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/hf_cache/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.cache/pip}"

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/MiniOneRec_v2}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$REPO_ROOT/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/research}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

TRAIN_FILE="${TRAIN_FILE:-$BUNDLE_ROOT/converted/rqvae/train/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
EVAL_FILE="${EVAL_FILE:-$BUNDLE_ROOT/converted/rqvae/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
INFO_FILE="${INFO_FILE:-$BUNDLE_ROOT/converted/rqvae/info/Industrial_and_Scientific_5_2016-10-2018-11.txt}"
SID_INDEX_PATH="${SID_INDEX_PATH:-$BUNDLE_ROOT/sid_variants/rqvae/Industrial_and_Scientific.index.json}"
ITEM_META_PATH="${ITEM_META_PATH:-$BUNDLE_ROOT/shared/Industrial_and_Scientific.item.json}"

NPROC="${NPROC:-1}"
CUDA_IDS="${CUDA_IDS:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29503}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
BETA="${BETA:-1e-3}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${MODE}_grpo}"

if [[ "$MODE" == "smoke" ]]; then
  MODEL_PATH="${MODEL_PATH:-$OUTPUT_ROOT/sft_smoke/final_checkpoint}"
  OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/grpo_smoke}"
elif [[ "$MODE" == "e3" ]]; then
  MODEL_PATH="${MODEL_PATH:-$OUTPUT_ROOT/E1_sft/final_checkpoint}"
  OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/E3_grpo}"
elif [[ "$MODE" == "e4" ]]; then
  MODEL_PATH="${MODEL_PATH:-$OUTPUT_ROOT/E2_dpo/final_checkpoint}"
  OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/E4_dpo_grpo}"
else
  echo "Unknown mode '$MODE'. Use: smoke | e3 | e4" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="$CUDA_IDS" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" accelerate launch \
  --config_file "$REPO_ROOT/config/zero2_opt.yaml" \
  --num_processes "$NPROC" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  "$REPO_ROOT/rl.py" \
  --model_path "$MODEL_PATH" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --train_file "$TRAIN_FILE" \
  --eval_file "$EVAL_FILE" \
  --info_file "$INFO_FILE" \
  --category "$CATEGORY" \
  --sample_train False \
  --eval_step 0.0999 \
  --reward_type ranking \
  --num_generations "$NUM_GENERATIONS" \
  --mask_all_zero False \
  --dynamic_sampling False \
  --sync_ref_model True \
  --beam_search True \
  --test_during_training False \
  --temperature 1.0 \
  --learning_rate "$LEARNING_RATE" \
  --add_gt False \
  --beta "$BETA" \
  --dapo False \
  --output_dir "$OUTPUT_DIR" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --sid_index_path "$SID_INDEX_PATH" \
  --item_meta_path "$ITEM_META_PATH"
