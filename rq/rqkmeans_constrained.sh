#!/bin/bash
#
# RQ-KMeans Constrained Training Script
#

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Default parameters
DATASET="Office_Products"
ROOT="$REPO_ROOT/data/Amazon/index"
OUTPUT_DIR="$REPO_ROOT/output/sid_office/rqkmeans_constrained"
K=256
L=3
MAX_ITER=100
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --l) L="$2"; shift 2 ;;
        --max_iter) MAX_ITER="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "Dataset: $DATASET"
echo "Root: $ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "K=$K, L=$L"

cd "$REPO_ROOT/rq"

python rqkmeans_constrained.py \
    --dataset "$DATASET" \
    --root "$ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --k "$K" \
    --l "$L" \
    --max_iter "$MAX_ITER" \
    --seed "$SEED" \
    --verbose
