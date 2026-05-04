#!/usr/bin/env bash
#SBATCH -J office-sft-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH -o /mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/eval/slurm-%j.out
#SBATCH -e /mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/eval/slurm-%j.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec}"
export CUDA_IDS="${CUDA_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_BEAMS="${NUM_BEAMS:-10}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

mkdir -p "$REPO_ROOT/output/office_rqkmeans_plus/eval"
cd "$REPO_ROOT"

echo "[$(date)] slurm_job_id=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "[$(date)] slurm_job_gpus=${SLURM_JOB_GPUS:-unset} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} cuda_ids=$CUDA_IDS"
bash "$REPO_ROOT/evaluate_office_sft_compare.sh"
