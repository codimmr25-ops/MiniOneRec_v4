#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="${DATASET:-Office_Products}"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/Amazon/index/${DATASET}.emb-qwen-td.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/sid_office/rqopq}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$OUTPUT_DIR/${DATASET}.rqopq}"

NUM_LEVELS="${NUM_LEVELS:-3}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
OPQ_M="${OPQ_M:-auto}"
OPQ_NITER="${OPQ_NITER:-8}"
OPQ_NITER_PQ="${OPQ_NITER_PQ:-4}"
MAX_BEAM_SIZE="${MAX_BEAM_SIZE:-1}"
TRAIN_SAMPLE="${TRAIN_SAMPLE:--1}"
SEED="${SEED:-42}"

cd "$REPO_ROOT"

python "$REPO_ROOT/rq/rqopq.py" \
  --dataset "$DATASET" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "$OUTPUT_PREFIX" \
  --num_levels "$NUM_LEVELS" \
  --codebook_size "$CODEBOOK_SIZE" \
  --opq_m "$OPQ_M" \
  --opq_niter "$OPQ_NITER" \
  --opq_niter_pq "$OPQ_NITER_PQ" \
  --max_beam_size "$MAX_BEAM_SIZE" \
  --train_sample "$TRAIN_SAMPLE" \
  --seed "$SEED"
