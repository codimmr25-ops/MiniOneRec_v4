#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="${DATASET:-Office_Products}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/Amazon/index/${DATASET}.emb-qwen-td.npy}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/output/sid_office/rqvae}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-20480}"
DEVICE="${DEVICE:-cuda:0}"

cd "$REPO_ROOT/rq"

python rqvae.py \
      --data_path "$DATA_PATH" \
      --ckpt_dir "$CKPT_DIR" \
      --lr "$LR" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --device "$DEVICE"
