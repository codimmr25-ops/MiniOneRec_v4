#!/usr/bin/env bash
#SBATCH -J office-grpo-v1
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH -o /mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/grpo/slurm-office-v1-%j.out
#SBATCH -e /mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/grpo/slurm-office-v1-%j.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec}"
export CONDA_ENV="${CONDA_ENV:-/mnt/bms_afs/miniconda3/envs/minionerec312}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/hf_cache}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export CUDA_IDS="${CUDA_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export NPROC="${NPROC:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29615}"
export MODE="${MODE:-full}"
export CATEGORY="${CATEGORY:-Office_Products}"
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/output/office_rqkmeans_plus/sft/new_lora/final_checkpoint}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/office_rqkmeans_plus/grpo}"
export TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/output/office_rqkmeans_plus/data/train/Office_Products_5_2016-10-2018-11.csv}"
export EVAL_FILE="${EVAL_FILE:-$REPO_ROOT/output/office_rqkmeans_plus/data/valid/Office_Products_5_2016-10-2018-11.csv}"
export INFO_FILE="${INFO_FILE:-$REPO_ROOT/output/office_rqkmeans_plus/data/info/Office_Products_5_2016-10-2018-11.txt}"
export SID_INDEX_PATH="${SID_INDEX_PATH:-$REPO_ROOT/output/sid_office/rqkmeans_plus/Office_Products.index.json}"
export ITEM_META_PATH="${ITEM_META_PATH:-$REPO_ROOT/data/Amazon/index/Office_Products.item.json}"

if [[ "$MODE" == "smoke" ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/office_v1_smoke}"
  export SAMPLE="${SAMPLE:-64}"
  export MAX_STEPS="${MAX_STEPS:-3}"
elif [[ "$MODE" == "trend" ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/office_v1_trend}"
  export SAMPLE="${SAMPLE:--1}"
  export MAX_STEPS="${MAX_STEPS:-175}"
elif [[ "$MODE" == "full" ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/office_v1_full}"
  export SAMPLE="${SAMPLE:--1}"
  export MAX_STEPS="${MAX_STEPS:--1}"
else
  echo "Unknown MODE '$MODE'. Use: smoke | trend | full" >&2
  exit 2
fi

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export LEARNING_RATE="${LEARNING_RATE:-5e-7}"
export BETA="${BETA:-0.05}"
export REWARD_TYPE="${REWARD_TYPE:-rule}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-32}"
export MASK_ALL_ZERO="${MASK_ALL_ZERO:-True}"
export ADD_GT="${ADD_GT:-False}"
export ONLY_SID_DATASET="${ONLY_SID_DATASET:-True}"
export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-MiniOneRec-Office-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export LOGGING_DIR="${LOGGING_DIR:-$OUTPUT_DIR/runs}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT-}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "[$(date)] slurm_job_id=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "[$(date)] mode=$MODE output=$OUTPUT_DIR cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} cuda_ids=$CUDA_IDS resume=$RESUME_FROM_CHECKPOINT"
bash "$REPO_ROOT/grpo_office_v1.sh" "$MODE"
