#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-new_lora}"

REPO_ROOT="${REPO_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec}"
CONDA_ENV="${CONDA_ENV:-/mnt/bms_afs/miniconda3/envs/minionerec312}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/hf_cache}"

export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_PATH="${MODEL_PATH:-$HF_CACHE_ROOT/Qwen2.5-7B}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/output/office_rqkmeans_plus/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/office_rqkmeans_plus/sft}"
CATEGORY="${CATEGORY:-Office_Products}"

TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/train/Office_Products_5_2016-10-2018-11.csv}"
EVAL_FILE="${EVAL_FILE:-$DATA_ROOT/valid/Office_Products_5_2016-10-2018-11.csv}"
SID_INDEX_PATH="${SID_INDEX_PATH:-$REPO_ROOT/output/sid_office/rqkmeans_plus/Office_Products.index.json}"
ITEM_META_PATH="${ITEM_META_PATH:-$REPO_ROOT/data/Amazon/index/Office_Products.item.json}"

CUDA_IDS="${CUDA_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
NPROC="${NPROC:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
SAMPLE="${SAMPLE:--1}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
EVAL_STEPS_RATIO="${EVAL_STEPS_RATIO:-0.10}"
SAVE_STEPS_RATIO="${SAVE_STEPS_RATIO:-0.10}"

USE_LORA=False
FREEZE_LLM=False
GRADIENT_CHECKPOINTING=True
TRAIN_NEW_TOKEN_EMBEDDINGS=True
OUTPUT_DIR="$OUTPUT_ROOT/$MODE"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-office_rqkmeans_plus_$MODE}"

case "$MODE" in
  original_full)
    USE_LORA=False
    FREEZE_LLM=False
    GRADIENT_CHECKPOINTING=True
    TRAIN_NEW_TOKEN_EMBEDDINGS=False
    ;;
  original_freeze_new_tokens)
    USE_LORA=False
    FREEZE_LLM=True
    GRADIENT_CHECKPOINTING=False
    TRAIN_NEW_TOKEN_EMBEDDINGS=True
    ;;
  new_lora)
    USE_LORA=True
    FREEZE_LLM=False
    GRADIENT_CHECKPOINTING=True
    TRAIN_NEW_TOKEN_EMBEDDINGS=True
    ;;
  *)
    echo "Unknown MODE '$MODE'. Use: original_full | original_freeze_new_tokens | new_lora" >&2
    exit 2
    ;;
esac

source /mnt/bms_afs/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "host=$(hostname)"
echo "mode=$MODE"
echo "repo=$REPO_ROOT"
echo "model=$MODEL_PATH"
echo "train=$TRAIN_FILE"
echo "eval=$EVAL_FILE"
echo "sid_index=$SID_INDEX_PATH"
echo "item_meta=$ITEM_META_PATH"
echo "output=$OUTPUT_DIR"
echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "cuda=$CUDA_IDS nproc=$NPROC batch=$BATCH_SIZE micro=$MICRO_BATCH_SIZE epochs=$NUM_EPOCHS lr=$LEARNING_RATE sample=$SAMPLE eval_ratio=$EVAL_STEPS_RATIO save_ratio=$SAVE_STEPS_RATIO"
echo "use_lora=$USE_LORA freeze_LLM=$FREEZE_LLM gradient_checkpointing=$GRADIENT_CHECKPOINTING"

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
  --freeze_LLM "$FREEZE_LLM" \
  --use_lora "$USE_LORA" \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --train_new_token_embeddings "$TRAIN_NEW_TOKEN_EMBEDDINGS" \
  --eval_steps_ratio "$EVAL_STEPS_RATIO" \
  --save_steps_ratio "$SAVE_STEPS_RATIO"
