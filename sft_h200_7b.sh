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

export MODEL_PATH="${MODEL_PATH:-$HF_CACHE_ROOT/Qwen2.5-7B}"
export BUNDLE_ROOT="${BUNDLE_ROOT:-$REPO_ROOT/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/h200_7b}"
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/E1_sft_rqvae}"
export CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

export TRAIN_FILE="${TRAIN_FILE:-$BUNDLE_ROOT/converted/rqvae/train/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
export EVAL_FILE="${EVAL_FILE:-$BUNDLE_ROOT/converted/rqvae/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
export SID_INDEX_PATH="${SID_INDEX_PATH:-$BUNDLE_ROOT/sid_variants/rqvae/Industrial_and_Scientific.index.json}"
export ITEM_META_PATH="${ITEM_META_PATH:-$BUNDLE_ROOT/shared/Industrial_and_Scientific.item.json}"

export CUDA_IDS="${CUDA_IDS:-0,1,2,3}"
export NPROC="${NPROC:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export SAMPLE="${SAMPLE:--1}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-h200_7b_sft_rqvae}"

source /mnt/bms_afs/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "host=$(hostname)"
echo "repo=$REPO_ROOT"
echo "model=$MODEL_PATH"
echo "output=$OUTPUT_DIR"
echo "cuda=$CUDA_IDS nproc=$NPROC"

CUDA_VISIBLE_DEVICES="$CUDA_IDS" torchrun --nproc_per_node "$NPROC" "$REPO_ROOT/sft.py" \
  --base_model "$MODEL_PATH" \
  --batch_size "$BATCH_SIZE" \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --sample "$SAMPLE" \
  --train_file "$TRAIN_FILE" \
  --eval_file "$EVAL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --category "$CATEGORY" \
  --train_from_scratch False \
  --seed 42 \
  --sid_index_path "$SID_INDEX_PATH" \
  --item_meta_path "$ITEM_META_PATH" \
  --freeze_LLM False \
  --use_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
  --gradient_checkpointing True \
  --train_new_token_embeddings True
