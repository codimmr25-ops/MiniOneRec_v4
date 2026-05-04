#!/usr/bin/env bash
#SBATCH -J office-sft-rqplus
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH -o /mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/sft/slurm-%j.out
#SBATCH -e /mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/sft/slurm-%j.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export CUDA_IDS="${CUDA_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export NPROC="${NPROC:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export SAMPLE="${SAMPLE:--1}"
export EVAL_STEPS_RATIO="${EVAL_STEPS_RATIO:-0.10}"
export SAVE_STEPS_RATIO="${SAVE_STEPS_RATIO:-0.10}"

mkdir -p "$REPO_ROOT/output/office_rqkmeans_plus/sft"
cd "$REPO_ROOT"

echo "[$(date)] slurm_job_id=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "[$(date)] slurm_job_gpus=${SLURM_JOB_GPUS:-unset} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} cuda_ids=$CUDA_IDS"
echo "[$(date)] start Office_Products rqkmeans_plus SFT comparison"
bash "$REPO_ROOT/run_office_sft_compare.sh"
echo "[$(date)] done Office_Products rqkmeans_plus SFT comparison"
