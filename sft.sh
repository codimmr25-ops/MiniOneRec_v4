#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/hf_cache/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.cache/pip}"

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/MiniOneRec_v2}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$REPO_ROOT/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411}"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen2.5-3B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/research}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

TRAIN_FILE="${TRAIN_FILE:-$BUNDLE_ROOT/converted/rqvae/train/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
EVAL_FILE="${EVAL_FILE:-$BUNDLE_ROOT/converted/rqvae/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
SID_INDEX_PATH="${SID_INDEX_PATH:-$BUNDLE_ROOT/sid_variants/rqvae/Industrial_and_Scientific.index.json}"
ITEM_META_PATH="${ITEM_META_PATH:-$BUNDLE_ROOT/shared/Industrial_and_Scientific.item.json}"

NPROC="${NPROC:-1}"
CUDA_IDS="${CUDA_IDS:-0}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${MODE}_sft}"

if [[ "$MODE" == "smoke" ]]; then
  SAMPLE="${SAMPLE:-128}"
  NUM_EPOCHS="${SMOKE_NUM_EPOCHS:-1}"
  BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
  OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/sft_smoke}"
elif [[ "$MODE" == "full" || "$MODE" == "e1" ]]; then
  SAMPLE="${SAMPLE:--1}"
  OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/E1_sft}"
else
  echo "Unknown mode '$MODE'. Use: smoke | full | e1" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

cmd=(
  "$REPO_ROOT/sft.py"
  --base_model "$MODEL_PATH"
  --batch_size "$BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --num_epochs "$NUM_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --sample "$SAMPLE"
  --train_file "$TRAIN_FILE"
  --eval_file "$EVAL_FILE"
  --output_dir "$OUTPUT_DIR"
  --wandb_project "$WANDB_PROJECT"
  --wandb_run_name "$WANDB_RUN_NAME"
  --category "$CATEGORY"
  --train_from_scratch False
  --seed 42
  --sid_index_path "$SID_INDEX_PATH"
  --item_meta_path "$ITEM_META_PATH"
  --freeze_LLM False
  --use_lora True
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --target_modules "$TARGET_MODULES"
  --gradient_checkpointing True
  --train_new_token_embeddings True
)

if [[ "$NPROC" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_IDS" python "${cmd[@]}"
else
  CUDA_VISIBLE_DEVICES="$CUDA_IDS" torchrun --nproc_per_node "$NPROC" "${cmd[@]}"
fi
