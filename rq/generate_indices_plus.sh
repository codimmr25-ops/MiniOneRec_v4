#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="${DATASET:-Office_Products}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/Amazon/index/${DATASET}.emb-qwen-td.npy}"
CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to an RQ-KMeans+ best_collision_model.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/output/sid_office/rqkmeans_plus/${DATASET}.index.json}"
E_DIM="${E_DIM:-2560}"
DEVICE="${DEVICE:-cuda:0}"

cd "$REPO_ROOT/rq"

python generate_indices_plus.py \
  --data_path "$DATA_PATH" \
  --ckpt_path "$CKPT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_emb_list 256 256 256 \
  --e_dim "$E_DIM" \
  --device "$DEVICE"
