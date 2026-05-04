#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-/mnt/bms_afs/miniconda3/envs/minionerec312}"
DATASET="${DATASET:-Office_Products}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/Amazon/index}"
DATA_PATH="${DATA_PATH:-$DATA_DIR/${DATASET}.emb-qwen-td.npy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/sid_office}"
DEVICE="${DEVICE:-cuda:0}"

RQFAISS_OUT="$OUTPUT_ROOT/rqkmeans_faiss"
CONSTRAINED_OUT="$OUTPUT_ROOT/rqkmeans_constrained"
RQOPQ_OUT="$OUTPUT_ROOT/rqopq"
RQVAE_OUT="$OUTPUT_ROOT/rqvae"
RQPLUS_OUT="$OUTPUT_ROOT/rqkmeans_plus"
COMPARE_OUT="$OUTPUT_ROOT/compare"

source /mnt/bms_afs/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

mkdir -p "$RQFAISS_OUT" "$CONSTRAINED_OUT" "$RQOPQ_OUT" "$RQVAE_OUT" "$RQPLUS_OUT" "$COMPARE_OUT"

cd "$REPO_ROOT"

echo "host=$(hostname)"
echo "repo=$REPO_ROOT"
echo "conda=$CONDA_ENV"
echo "dataset=$DATASET"
echo "data=$DATA_PATH"
echo "output=$OUTPUT_ROOT"
echo "device=$DEVICE"

test -f "$DATA_PATH"

echo "[1/5] RQ-KMeans FAISS"
python "$REPO_ROOT/rq/rqkmeans_faiss.py" \
  --dataset "$DATASET" \
  --data_path "$DATA_PATH" \
  --output_root "$RQFAISS_OUT"

echo "[2/5] RQ-KMeans constrained"
bash "$REPO_ROOT/rq/rqkmeans_constrained.sh" \
  --dataset "$DATASET" \
  --root "$DATA_DIR" \
  --output_dir "$CONSTRAINED_OUT"

echo "[3/5] RQ-OPQ"
DATASET="$DATASET" \
DATA_PATH="$DATA_PATH" \
OUTPUT_DIR="$RQOPQ_OUT" \
DEVICE="$DEVICE" \
bash "$REPO_ROOT/rq/rqopq.sh"

echo "[4/5] RQ-VAE"
DATASET="$DATASET" \
DATA_PATH="$DATA_PATH" \
CKPT_DIR="$RQVAE_OUT" \
DEVICE="$DEVICE" \
bash "$REPO_ROOT/rq/rqvae.sh"

RQVAE_CKPT="$(find "$RQVAE_OUT" -name best_collision_model.pth | sort | tail -n 1)"
if [[ -z "$RQVAE_CKPT" ]]; then
  echo "No RQ-VAE best_collision_model.pth found under $RQVAE_OUT" >&2
  exit 1
fi
cd "$REPO_ROOT/rq"
python generate_indices.py \
  --ckpt_path "$RQVAE_CKPT" \
  --data_path "$DATA_PATH" \
  --output_path "$RQVAE_OUT/${DATASET}.index.json" \
  --device "$DEVICE"
cd "$REPO_ROOT"

echo "[5/5] RQ-KMeans+"
DATASET="$DATASET" \
DATA_PATH="$DATA_PATH" \
PRETRAINED_CODEBOOK_PATH="$CONSTRAINED_OUT/${DATASET}.codebooks_constrained.npz" \
CKPT_DIR="$RQPLUS_OUT" \
DEVICE="$DEVICE" \
bash "$REPO_ROOT/rq/rqkmeans_plus.sh"

RQPLUS_CKPT="$(find "$RQPLUS_OUT" -name best_collision_model.pth | sort | tail -n 1)"
if [[ -z "$RQPLUS_CKPT" ]]; then
  echo "No RQ-KMeans+ best_collision_model.pth found under $RQPLUS_OUT" >&2
  exit 1
fi
CKPT_PATH="$RQPLUS_CKPT" \
DATA_PATH="$DATA_PATH" \
OUTPUT_PATH="$RQPLUS_OUT/${DATASET}.index.json" \
DEVICE="$DEVICE" \
bash "$REPO_ROOT/rq/generate_indices_plus.sh"

echo "[compare] SID collision summary"
python "$REPO_ROOT/rq/compare_sid_indices.py" \
  --index "rqkmeans_faiss=$RQFAISS_OUT/$DATASET/${DATASET}.faiss-rq.index.json" \
  --index "rqkmeans_constrained=$CONSTRAINED_OUT/${DATASET}.index.json" \
  --index "rqopq=$RQOPQ_OUT/${DATASET}.rqopq.index.json" \
  --index "rqvae=$RQVAE_OUT/${DATASET}.index.json" \
  --index "rqkmeans_plus=$RQPLUS_OUT/${DATASET}.index.json" \
  --output_csv "$COMPARE_OUT/${DATASET}_sid_compare.csv" \
  --output_json "$COMPARE_OUT/${DATASET}_sid_compare.json"

echo "Done. Compare files:"
echo "  $COMPARE_OUT/${DATASET}_sid_compare.csv"
echo "  $COMPARE_OUT/${DATASET}_sid_compare.json"
