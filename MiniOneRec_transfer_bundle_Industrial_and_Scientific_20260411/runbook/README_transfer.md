# MiniOneRec Transfer Runbook

## 1. What This Bundle Is For

This bundle packages the final SID outputs for 4 methods on `Industrial_and_Scientific`, plus per-method converted downstream data for `SFT / RL / Eval`.

Use this bundle on a new machine together with a fresh copy of the `MiniOneRec` repository.

## 2. Important Compatibility Rule

- Directly trainable with the current codebase:
  - `rqvae`
  - `rqkmeans_faiss`
- Do **not** directly train with the current codebase:
  - `constrained_rqkmeans`
  - `rqkmeans_plus`

Reason: current `data.py` hardcodes SID usage as the concatenation of the first 3 SID tokens, while these two methods rely on 4-token deduplication for zero-collision output.

## 3. Expected Directory Variables

```bash
REPO_ROOT=/path/to/MiniOneRec
BUNDLE_ROOT=/path/to/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411
BASE_MODEL=/path/to/your/base/model
OUTPUT_ROOT=/path/to/your/output
```

## 4. Recommended Primary Method: RQ-VAE

### 4.1 SFT

```bash
torchrun --nproc_per_node 8 "$REPO_ROOT/sft.py" \
  --base_model "$BASE_MODEL" \
  --batch_size 1024 \
  --micro_batch_size 16 \
  --train_file "$BUNDLE_ROOT/converted/rqvae/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  --eval_file "$BUNDLE_ROOT/converted/rqvae/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  --output_dir "$OUTPUT_ROOT/rqvae_sft" \
  --wandb_project your_wandb_project \
  --wandb_run_name rqvae_sft \
  --category Industrial_and_Scientific \
  --train_from_scratch False \
  --seed 42 \
  --sid_index_path "$BUNDLE_ROOT/sid_variants/rqvae/Industrial_and_Scientific.index.json" \
  --item_meta_path "$BUNDLE_ROOT/shared/Industrial_and_Scientific.item.json" \
  --freeze_LLM False
```

### 4.2 RL

```bash
HF_ENDPOINT=https://hf-mirror.com accelerate launch \
  --config_file "$REPO_ROOT/config/zero2_opt.yaml" \
  --num_processes 8 \
  --main_process_port 29503 \
  "$REPO_ROOT/rl.py" \
  --model_path "$OUTPUT_ROOT/rqvae_sft/final_checkpoint" \
  --train_batch_size 64 \
  --eval_batch_size 128 \
  --num_train_epochs 2 \
  --gradient_accumulation_steps 2 \
  --train_file "$BUNDLE_ROOT/converted/rqvae/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  --eval_file "$BUNDLE_ROOT/converted/rqvae/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  --info_file "$BUNDLE_ROOT/converted/rqvae/info/Industrial_and_Scientific_5_2016-10-2018-11.txt" \
  --category Industrial_and_Scientific \
  --sample_train False \
  --eval_step 0.0999 \
  --reward_type ranking \
  --num_generations 16 \
  --mask_all_zero False \
  --dynamic_sampling False \
  --sync_ref_model True \
  --beam_search True \
  --test_during_training False \
  --temperature 1.0 \
  --learning_rate 1e-5 \
  --add_gt False \
  --beta 1e-3 \
  --dapo False \
  --output_dir "$OUTPUT_ROOT/rqvae_rl" \
  --wandb_run_name rqvae_rl \
  --sid_index_path "$BUNDLE_ROOT/sid_variants/rqvae/Industrial_and_Scientific.index.json" \
  --item_meta_path "$BUNDLE_ROOT/shared/Industrial_and_Scientific.item.json"
```

### 4.3 Evaluation

```bash
EXP_NAME="$OUTPUT_ROOT/rqvae_rl/final_checkpoint"
TEST_FILE="$BUNDLE_ROOT/converted/rqvae/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"
INFO_FILE="$BUNDLE_ROOT/converted/rqvae/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"
TEMP_DIR="$OUTPUT_ROOT/temp_rqvae_eval"
RESULT_DIR="$OUTPUT_ROOT/results_rqvae_eval"

mkdir -p "$TEMP_DIR" "$RESULT_DIR"

python "$REPO_ROOT/split.py" \
  --input_path "$TEST_FILE" \
  --output_path "$TEMP_DIR" \
  --cuda_list "0,1,2,3,4,5,6,7"

for i in 0 1 2 3 4 5 6 7; do
  if [[ -f "$TEMP_DIR/$i.csv" ]]; then
    CUDA_VISIBLE_DEVICES=$i python -u "$REPO_ROOT/evaluate.py" \
      --base_model "$EXP_NAME" \
      --info_file "$INFO_FILE" \
      --category Industrial_and_Scientific \
      --test_data_path "$TEMP_DIR/$i.csv" \
      --result_json_data "$TEMP_DIR/$i.json" \
      --batch_size 8 \
      --num_beams 50 \
      --max_new_tokens 256 \
      --temperature 1.0 \
      --guidance_scale 1.0 \
      --length_penalty 0.0 &
  fi
done
wait

CUDA_LIST=$(ls "$TEMP_DIR"/*.json | sed 's#^.*/##' | sed 's/\\.json$//' | tr '\\n' ',' | sed 's/,$//')

python "$REPO_ROOT/merge.py" \
  --input_path "$TEMP_DIR" \
  --output_path "$RESULT_DIR/final_result_Industrial_and_Scientific.json" \
  --cuda_list "$CUDA_LIST"

python "$REPO_ROOT/calc.py" \
  --path "$RESULT_DIR/final_result_Industrial_and_Scientific.json" \
  --item_path "$INFO_FILE"
```

## 5. Secondary Direct-Train Baseline: RQ-Kmeans

Use the same commands as above, but replace every `rqvae` path with `rqkmeans_faiss`.

Key path substitutions:

```bash
$BUNDLE_ROOT/converted/rqkmeans_faiss/...
$BUNDLE_ROOT/sid_variants/rqkmeans_faiss/Industrial_and_Scientific.index.json
```

## 6. Why The Other Two Methods Are Bundle-Only

The bundle still contains:

- `sid_variants/constrained_rqkmeans/Industrial_and_Scientific.index.json`
- `sid_variants/rqkmeans_plus/Industrial_and_Scientific.index.json`
- their corresponding converted `train / valid / test / info`

These are included for:

- SID-level comparison
- later code adaptation
- future experiments with variable-length SID support

They are **not** recommended for direct use under the current code because their final zero-collision result depends on 4-token SIDs.

## 7. Integrity Check On The New Machine

After copying the bundle:

```bash
cd "$BUNDLE_ROOT"
sha256sum -c report/SHA256SUMS
```

If all lines return `OK`, the bundle transfer is complete.

