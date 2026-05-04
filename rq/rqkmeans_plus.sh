#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="${DATASET:-Office_Products}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/Amazon/index/${DATASET}.emb-qwen-td.npy}"
PRETRAINED_CODEBOOK_PATH="${PRETRAINED_CODEBOOK_PATH:-$REPO_ROOT/output/sid_office/rqkmeans_constrained/${DATASET}.codebooks_constrained.npz}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/output/sid_office/rqkmeans_plus}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
E_DIM="${E_DIM:-2560}"
DEVICE="${DEVICE:-cuda:0}"

cd "$REPO_ROOT/rq"

python rqkmeans_plus.py \
  --data_path "$DATA_PATH" \
  --pretrained_codebook_path "$PRETRAINED_CODEBOOK_PATH" \
  --num_emb_list 256 256 256 \
  --e_dim "$E_DIM" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --ckpt_dir "$CKPT_DIR" \
  --device "$DEVICE"
