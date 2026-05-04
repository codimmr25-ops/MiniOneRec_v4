#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec}"
CONDA_ENV="${CONDA_ENV:-/mnt/bms_afs/miniconda3/envs/minionerec312}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/hf_cache}"

export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export BUNDLE_ROOT="${BUNDLE_ROOT:-$REPO_ROOT/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/h200_7b}"
export MODEL_PATH="${MODEL_PATH:-$OUTPUT_ROOT/E1_sft_rqvae/final_checkpoint}"
export CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

export TRAIN_FILE="${TRAIN_FILE:-$BUNDLE_ROOT/converted/rqvae/train/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
export EVAL_FILE="${EVAL_FILE:-$BUNDLE_ROOT/converted/rqvae/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
export INFO_FILE="${INFO_FILE:-$BUNDLE_ROOT/converted/rqvae/info/Industrial_and_Scientific_5_2016-10-2018-11.txt}"
export SID_INDEX_PATH="${SID_INDEX_PATH:-$BUNDLE_ROOT/sid_variants/rqvae/Industrial_and_Scientific.index.json}"
export ITEM_META_PATH="${ITEM_META_PATH:-$BUNDLE_ROOT/shared/Industrial_and_Scientific.item.json}"

export PROVIDER="${PROVIDER:-minimax}"
export LLM_MODEL="${LLM_MODEL:-MiniMax-M2.5}"
export API_KEY_ENV="${API_KEY_ENV:-MINIMAX_API_KEY}"
export BASE_URL="${BASE_URL:-}"
export PREFERENCE_BATCH_SIZE="${PREFERENCE_BATCH_SIZE:-4}"
export PREFERENCE_MAX_TOKENS="${PREFERENCE_MAX_TOKENS:-128}"
export GENERATION_CUDA_IDS="${GENERATION_CUDA_IDS:-0}"
export GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
export NUM_BEAMS="${NUM_BEAMS:-8}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
export NUM_NEGATIVES_PER_POSITIVE="${NUM_NEGATIVES_PER_POSITIVE:-1}"
export SEED="${SEED:-2026}"

export TRAIN_PREFERENCE_JSONL="${TRAIN_PREFERENCE_JSONL:-$OUTPUT_ROOT/preferences_train_history_only.jsonl}"
export EVAL_PREFERENCE_JSONL="${EVAL_PREFERENCE_JSONL:-$OUTPUT_ROOT/preferences_valid_history_only.jsonl}"
export TRAIN_JSONL="${TRAIN_JSONL:-$OUTPUT_ROOT/E2_pairs_train.jsonl}"
export EVAL_JSONL="${EVAL_JSONL:-$OUTPUT_ROOT/E2_pairs_valid.jsonl}"

source /mnt/bms_afs/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"

echo "host=$(hostname)"
echo "repo=$REPO_ROOT"
echo "model=$MODEL_PATH"
echo "train_preference=$TRAIN_PREFERENCE_JSONL"
echo "eval_preference=$EVAL_PREFERENCE_JSONL"
echo "train_pairs=$TRAIN_JSONL"
echo "eval_pairs=$EVAL_JSONL"

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Environment variable $API_KEY_ENV is required to generate history_only preferences." >&2
  exit 2
fi

python "$REPO_ROOT/build_preference_data.py" \
  --input_csv "$TRAIN_FILE" \
  --item_meta_path "$ITEM_META_PATH" \
  --sid_index_path "$SID_INDEX_PATH" \
  --output_jsonl "$TRAIN_PREFERENCE_JSONL" \
  --provider "$PROVIDER" \
  --llm_model "$LLM_MODEL" \
  --api_key_env "$API_KEY_ENV" \
  --base_url "$BASE_URL" \
  --batch_size "$PREFERENCE_BATCH_SIZE" \
  --max_tokens "$PREFERENCE_MAX_TOKENS" \
  --preference_mode history_only

python "$REPO_ROOT/build_preference_data.py" \
  --input_csv "$EVAL_FILE" \
  --item_meta_path "$ITEM_META_PATH" \
  --sid_index_path "$SID_INDEX_PATH" \
  --output_jsonl "$EVAL_PREFERENCE_JSONL" \
  --provider "$PROVIDER" \
  --llm_model "$LLM_MODEL" \
  --api_key_env "$API_KEY_ENV" \
  --base_url "$BASE_URL" \
  --batch_size "$PREFERENCE_BATCH_SIZE" \
  --max_tokens "$PREFERENCE_MAX_TOKENS" \
  --preference_mode history_only

CUDA_VISIBLE_DEVICES="$GENERATION_CUDA_IDS" python "$REPO_ROOT/build_dpo_pairs.py" \
  --model_path "$MODEL_PATH" \
  --preference_jsonl "$TRAIN_PREFERENCE_JSONL" \
  --info_file "$INFO_FILE" \
  --output_jsonl "$TRAIN_JSONL" \
  --num_beams "$NUM_BEAMS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --use_user_preference true \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  --num_negatives_per_positive "$NUM_NEGATIVES_PER_POSITIVE" \
  --filter_history_items true \
  --append_completion_newline true \
  --seed "$SEED"

CUDA_VISIBLE_DEVICES="$GENERATION_CUDA_IDS" python "$REPO_ROOT/build_dpo_pairs.py" \
  --model_path "$MODEL_PATH" \
  --preference_jsonl "$EVAL_PREFERENCE_JSONL" \
  --info_file "$INFO_FILE" \
  --output_jsonl "$EVAL_JSONL" \
  --num_beams "$NUM_BEAMS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --use_user_preference true \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  --num_negatives_per_positive "$NUM_NEGATIVES_PER_POSITIVE" \
  --filter_history_items true \
  --append_completion_newline true \
  --seed "$SEED"

python "$REPO_ROOT/scripts/check_dpo_pairs.py" --input_jsonl "$TRAIN_JSONL"
python "$REPO_ROOT/scripts/check_dpo_pairs.py" --input_jsonl "$EVAL_JSONL"
