#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec}"
export CONDA_ENV="${CONDA_ENV:-/mnt/bms_afs/miniconda3/envs/minionerec312}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/hf_cache}"

export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export BUNDLE_ROOT="${BUNDLE_ROOT:-$REPO_ROOT/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/h200_7b}"
export RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/h200_7b}"
export CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

export EXP_NAME="${EXP_NAME:-$OUTPUT_ROOT/E2_dpo_rqvae/final_checkpoint}"
export CUDA_IDS="${CUDA_IDS:-0,1,2,3}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_BEAMS="${NUM_BEAMS:-10}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
export TEMP_DIR="${TEMP_DIR:-$OUTPUT_ROOT/temp_eval_E2_dpo_rqvae_final}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_ROOT/E2_dpo_rqvae_final}"

source /mnt/bms_afs/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$REPO_ROOT"

echo "host=$(hostname)"
echo "model=$EXP_NAME"
echo "output=$OUTPUT_DIR"
echo "cuda=$CUDA_IDS batch=$BATCH_SIZE beams=$NUM_BEAMS"

bash "$REPO_ROOT/evaluate.sh" E2_dpo_rqvae_final
