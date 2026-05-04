#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/hf_cache/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.cache/pip}"

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/MiniOneRec_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/research}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
BETA="${BETA:-0.1}"
LOSS_TYPE="${LOSS_TYPE:-sigmoid}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${MODE}_dpo}"

if [[ "$MODE" == "smoke" ]]; then
  MODEL_PATH="${MODEL_PATH:-$OUTPUT_ROOT/sft_smoke/final_checkpoint}"
  OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/dpo_smoke}"
  TRAIN_JSONL="${TRAIN_JSONL:-$OUTPUT_ROOT/dpo_pairs_smoke.jsonl}"
  EVAL_JSONL="${EVAL_JSONL:-$OUTPUT_ROOT/dpo_pairs_smoke.jsonl}"
elif [[ "$MODE" == "e2" || "$MODE" == "full" ]]; then
  MODEL_PATH="${MODEL_PATH:-$OUTPUT_ROOT/E1_sft/final_checkpoint}"
  OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/E2_dpo}"
  TRAIN_JSONL="${TRAIN_JSONL:-$OUTPUT_ROOT/E2_pairs_train.jsonl}"
  EVAL_JSONL="${EVAL_JSONL:-$OUTPUT_ROOT/E2_pairs_valid.jsonl}"
else
  echo "Unknown mode '$MODE'. Use: smoke | e2 | full" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

python "$REPO_ROOT/dpo.py" \
  --model_path "$MODEL_PATH" \
  --train_jsonl "$TRAIN_JSONL" \
  --eval_jsonl "$EVAL_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --beta "$BETA" \
  --loss_type "$LOSS_TYPE" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "$WANDB_RUN_NAME"
