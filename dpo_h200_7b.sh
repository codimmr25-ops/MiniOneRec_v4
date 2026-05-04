#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec}"
CONDA_ENV="${CONDA_ENV:-/mnt/bms_afs/miniconda3/envs/minionerec312}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/hf_cache}"

export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/h200_7b}"
export MODEL_PATH="${MODEL_PATH:-$OUTPUT_ROOT/E1_sft_rqvae/final_checkpoint}"
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/E2_dpo_rqvae}"
export TRAIN_JSONL="${TRAIN_JSONL:-$OUTPUT_ROOT/E2_pairs_train.jsonl}"
export EVAL_JSONL="${EVAL_JSONL:-$OUTPUT_ROOT/E2_pairs_valid.jsonl}"

export CUDA_IDS="${CUDA_IDS:-0,1,2,3}"
export NPROC="${NPROC:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29523}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export BETA="${BETA:-0.1}"
export LOSS_TYPE="${LOSS_TYPE:-sigmoid}"
export LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export MAX_LENGTH="${MAX_LENGTH:-544}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export USE_LORA="${USE_LORA:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-h200_7b_dpo_rqvae}"

source /mnt/bms_afs/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "host=$(hostname)"
echo "repo=$REPO_ROOT"
echo "model=$MODEL_PATH"
echo "train_jsonl=$TRAIN_JSONL"
echo "eval_jsonl=$EVAL_JSONL"
echo "output=$OUTPUT_DIR"
echo "cuda=$CUDA_IDS nproc=$NPROC"
echo "beta=$BETA loss_type=$LOSS_TYPE label_smoothing=$LABEL_SMOOTHING"

CUDA_VISIBLE_DEVICES="$CUDA_IDS" accelerate launch \
  --config_file "$REPO_ROOT/config/zero2_opt.yaml" \
  --num_processes "$NPROC" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  "$REPO_ROOT/dpo.py" \
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
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_length "$MAX_LENGTH" \
  --logging_steps "$LOGGING_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --use_lora "$USE_LORA" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "$WANDB_RUN_NAME"
