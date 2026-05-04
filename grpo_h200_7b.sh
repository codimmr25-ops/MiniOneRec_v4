#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"

REPO_ROOT="${REPO_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec}"
CONDA_ENV="${CONDA_ENV:-/mnt/bms_afs/miniconda3/envs/minionerec312}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/hf_cache}"

export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export BUNDLE_ROOT="${BUNDLE_ROOT:-$REPO_ROOT/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411}"
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/output/h200_7b/E1_sft_rqvae/final_checkpoint}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/h200_7b}"
export CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

export TRAIN_FILE="${TRAIN_FILE:-$BUNDLE_ROOT/converted/rqvae/train/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
export EVAL_FILE="${EVAL_FILE:-$BUNDLE_ROOT/converted/rqvae/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
export INFO_FILE="${INFO_FILE:-$BUNDLE_ROOT/converted/rqvae/info/Industrial_and_Scientific_5_2016-10-2018-11.txt}"
export SID_INDEX_PATH="${SID_INDEX_PATH:-$BUNDLE_ROOT/sid_variants/rqvae/Industrial_and_Scientific.index.json}"
export ITEM_META_PATH="${ITEM_META_PATH:-$BUNDLE_ROOT/shared/Industrial_and_Scientific.item.json}"

export CUDA_IDS="${CUDA_IDS:-0,1,2,3}"
export NPROC="${NPROC:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29513}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export BETA="${BETA:-1e-3}"
export REWARD_TYPE="${REWARD_TYPE:-ranking}"
export OPTIM="${OPTIM:-paged_adamw_32bit}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ "$MODE" == "smoke" ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/E3_grpo_rqvae_smoke}"
  export SAMPLE="${SAMPLE:-64}"
  export SEQ_SAMPLE="${SEQ_SAMPLE:-64}"
  export MAX_STEPS="${MAX_STEPS:-3}"
  export EVAL_STEP="${EVAL_STEP:-0.5}"
elif [[ "$MODE" == "full" || "$MODE" == "e3" ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/E3_grpo_rqvae}"
  export SAMPLE="${SAMPLE:--1}"
  export SEQ_SAMPLE="${SEQ_SAMPLE:-10000}"
  export MAX_STEPS="${MAX_STEPS:--1}"
  export EVAL_STEP="${EVAL_STEP:-0.0999}"
else
  echo "Unknown mode '$MODE'. Use: smoke | full | e3" >&2
  exit 2
fi

export WANDB_RUN_NAME="${WANDB_RUN_NAME:-h200_7b_${MODE}_grpo_rqvae}"

source /mnt/bms_afs/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "host=$(hostname)"
echo "repo=$REPO_ROOT"
echo "model=$MODEL_PATH"
echo "output=$OUTPUT_DIR"
echo "cuda=$CUDA_IDS nproc=$NPROC mode=$MODE"

CUDA_VISIBLE_DEVICES="$CUDA_IDS" accelerate launch \
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
  --sample "$SAMPLE" \
  --seq_sample "$SEQ_SAMPLE" \
  --max_steps "$MAX_STEPS" \
  --eval_step "$EVAL_STEP" \
  --reward_type "$REWARD_TYPE" \
  --num_generations "$NUM_GENERATIONS" \
  --mask_all_zero False \
  --dynamic_sampling False \
  --sync_ref_model False \
  --beam_search True \
  --test_during_training False \
  --temperature 1.0 \
  --learning_rate "$LEARNING_RATE" \
  --add_gt False \
  --beta "$BETA" \
  --dapo False \
  --optim "$OPTIM" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --sid_index_path "$SID_INDEX_PATH" \
  --item_meta_path "$ITEM_META_PATH" \
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
