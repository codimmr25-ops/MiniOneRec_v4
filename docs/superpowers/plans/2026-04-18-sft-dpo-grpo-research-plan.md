# MiniOneRec SFT-DPO-GRPO Research Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend MiniOneRec from the current `SFT -> GRPO` workflow to a complete `SFT -> DPO -> GRPO` recommendation research pipeline, then run four controlled experiments and compare them with a unified evaluation suite.

**Architecture:** Keep the existing SID pipeline and constrained decoding path intact. Add a preference-data construction stage between SFT and GRPO, implement DPO as a separate train stage, and expand evaluation so every model checkpoint is compared with the same metrics, bucket analysis, and error statistics.

**Tech Stack:** Python, PyTorch, Transformers, TRL, Accelerate, DeepSpeed, Hugging Face tokenizers, pandas, numpy

---

## Scope And Assumptions

- Base model target: `Qwen2.5-3B`
- Training hardware target:
  - Preferred: `1x H800 80G/94G` or `1x A800 80G`
  - Acceptable fallback: `4x RTX 4090 24G` with PEFT and memory-reduced settings
- Dataset target: `Industrial_and_Scientific`
- Default research paths:
  - `TRAIN_CSV = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv`
  - `VALID_CSV = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv`
  - `TEST_CSV = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\test\Industrial_and_Scientific_5_2016-10-2018-11.csv`
  - `INFO_TXT = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt`
  - `SID_JSON = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json`
  - `ITEM_JSON = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json`
- Experiment matrix:
  - `SFT`
  - `SFT + DPO`
  - `SFT + GRPO`
  - `SFT + DPO + GRPO`
- Unified metrics:
  - `HR@K`
  - `NDCG@K`
  - `invalid generation rate`
  - `coverage`
  - `head/mid/tail bucket performance`
  - `short/long history user bucket performance`

## File Structure

### Existing Files To Reuse

- `D:\code\MiniOneRec_v2\sft.py`
  - Existing SFT trainer and tokenizer extension entrypoint
- `D:\code\MiniOneRec_v2\rl.py`
  - Existing GRPO training entrypoint
- `D:\code\MiniOneRec_v2\minionerec_trainer.py`
  - Existing recommendation-aware GRPO trainer
- `D:\code\MiniOneRec_v2\data.py`
  - Existing SFT, RL, eval datasets and prompt formatting
- `D:\code\MiniOneRec_v2\evaluate.py`
  - Existing constrained decoding evaluator
- `D:\code\MiniOneRec_v2\calc.py`
  - Existing HR/NDCG calculator with invalid-item count logic
- `D:\code\MiniOneRec_v2\config\zero2_opt.yaml`
  - Existing multi-GPU RL launch config

### New Files To Create

- `D:\code\MiniOneRec_v2\dpo.py`
  - DPO training entrypoint
- `D:\code\MiniOneRec_v2\dpo.sh`
  - DPO launch script
- `D:\code\MiniOneRec_v2\build_dpo_pairs.py`
  - Build chosen/rejected pairs from SFT generations and reward rules
- `D:\code\MiniOneRec_v2\analysis\metrics_plus.py`
  - Unified offline metrics: invalid rate, coverage, bucket metrics
- `D:\code\MiniOneRec_v2\analysis\bucket_analysis.py`
  - Head/mid/tail and user-history bucket analysis
- `D:\code\MiniOneRec_v2\analysis\collect_results.py`
  - Aggregate metrics from the four experiments into one table
- `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_sft.yaml`
  - Research SFT settings
- `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_dpo.yaml`
  - Research DPO settings
- `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_grpo.yaml`
  - Research GRPO settings
- `D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`
  - Manual experiment tracker and result log

### Existing Files Likely To Modify

- `D:\code\MiniOneRec_v2\requirements.txt`
  - Ensure PEFT/DPO dependencies are explicit
- `D:\code\MiniOneRec_v2\data.py`
  - Add DPO dataset class and bucket metadata helpers
- `D:\code\MiniOneRec_v2\sft.sh`
  - Replace hard-coded 8-GPU assumptions with research launch pattern
- `D:\code\MiniOneRec_v2\rl.sh`
  - Add reduced-memory defaults for 3B
- `D:\code\MiniOneRec_v2\evaluate.sh`
  - Add research eval settings and result directory conventions
- `D:\code\MiniOneRec_v2\README.md`
  - Add research pipeline section after results are validated

---

### Task 1: Establish The Research Baseline And Runtime Constraints

**Files:**
- Create: `D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`
- Modify: `D:\code\MiniOneRec_v2\sft.sh`
- Modify: `D:\code\MiniOneRec_v2\rl.sh`
- Modify: `D:\code\MiniOneRec_v2\evaluate.sh`

- [ ] **Step 1: Create an experiment tracker with fixed experiment IDs**

Use this file content:

```markdown
# 2026-04-18 Experiment Matrix

| Exp ID | Pipeline | Base Model | SID Variant | Train Checkpoint | Eval Result JSON | Notes |
|---|---|---|---|---|---|---|
| E1 | SFT | Qwen2.5-3B | rqvae |  |  |  |
| E2 | SFT+DPO | Qwen2.5-3B | rqvae |  |  |  |
| E3 | SFT+GRPO | Qwen2.5-3B | rqvae |  |  |  |
| E4 | SFT+DPO+GRPO | Qwen2.5-3B | rqvae |  |  |  |
```

- [ ] **Step 2: Replace hard-coded 8-GPU assumptions in the launch scripts with explicit variables**

Target shell pattern:

```bash
$NPROC = ${env:NPROC}
if (-not $NPROC) { $NPROC = 1 }
```

For bash-style scripts in repo, the equivalent target shape is:

```bash
NPROC=${NPROC:-1}
CUDA_IDS=${CUDA_IDS:-0}
```

- [ ] **Step 3: Define memory-safe defaults for the 3B research route**

Use these baseline defaults in the plan and later configs:

```text
SFT:
- micro_batch_size = 1 or 2
- gradient_accumulation_steps tuned up to target effective batch

DPO:
- train batch per device = 1
- use PEFT
- max prompt len and max target len capped

GRPO:
- per_device_train_batch_size = 1 or 2
- num_generations = 4 first, then 8 only if stable
- max_completion_length = 16 or 32 first

Eval:
- batch_size = 1 or 2
- num_beams = 10 first, then 20 for final
```

- [ ] **Step 4: Record the baseline launch commands in the tracker**

Run targets to document, not yet to optimize:

```bash
python sft.py --help
python rl.py --help
python evaluate.py --help
```

Expected outcome: CLI arguments are frozen into the plan before new code is added.

- [ ] **Step 5: Commit the baseline planning artifacts**

```bash
git add docs/research/2026-04-18-experiment-matrix.md sft.sh rl.sh evaluate.sh
git commit -m "docs: define research experiment matrix and runtime constraints"
```

---

### Task 2: Convert SFT To A Research-Grade Starting Point

**Files:**
- Modify: `D:\code\MiniOneRec_v2\sft.py`
- Modify: `D:\code\MiniOneRec_v2\requirements.txt`
- Create: `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_sft.yaml`

- [ ] **Step 1: Add PEFT-capable arguments to SFT**

Target argument block:

```python
def train(
    base_model: str = "",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    gradient_checkpointing: bool = True,
    ...
):
```

- [ ] **Step 2: Add the PEFT import dependency**

Target requirement:

```text
peft>=0.14.0
```

- [ ] **Step 3: Add the minimal PEFT wrapping path before Trainer construction**

Target logic:

```python
from peft import LoraConfig, get_peft_model

if use_lora:
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules.split(","),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
```

- [ ] **Step 4: Add gradient checkpointing when LoRA is enabled**

Target logic:

```python
if gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
```

- [ ] **Step 5: Create the first research config for SFT**

Use this config skeleton:

```yaml
base_model: Qwen/Qwen2.5-3B
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
batch_size: 64
micro_batch_size: 1
num_epochs: 2
learning_rate: 2e-4
freeze_LLM: false
```

- [ ] **Step 6: Run a smoke SFT with a small subset**

Run:

```bash
python sft.py --base_model Qwen/Qwen2.5-3B --train_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --eval_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --sid_index_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json" --item_meta_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json" --output_dir output/research/sft_smoke --micro_batch_size 1 --batch_size 8 --num_epochs 1 --use_lora True
```

Expected outcome: one short run completes and saves a checkpoint.

- [ ] **Step 7: Commit the SFT research path**

```bash
git add sft.py requirements.txt configs/research/qwen25_3b_sft.yaml
git commit -m "feat: add memory-safe research SFT path for 3B"
```

---

### Task 3: Build DPO Pair Data From SFT Generations

**Files:**
- Create: `D:\code\MiniOneRec_v2\build_dpo_pairs.py`
- Modify: `D:\code\MiniOneRec_v2\data.py`

- [ ] **Step 1: Define the DPO pair JSONL schema**

Use this schema:

```json
{
  "prompt": "### User Input: ...",
  "chosen": "<a_1><b_9><c_3>",
  "rejected": "<a_5><b_2><c_7>",
  "target": "<a_1><b_9><c_3>",
  "reward_chosen": 1.0,
  "reward_rejected": 0.0,
  "bucket_item_popularity": "tail",
  "bucket_history_length": "short"
}
```

- [ ] **Step 2: Implement candidate generation input from an SFT checkpoint**

Target CLI:

```python
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--info_file", type=str, required=True)
parser.add_argument("--output_jsonl", type=str, required=True)
parser.add_argument("--num_beams", type=int, default=8)
parser.add_argument("--max_new_tokens", type=int, default=32)
```

- [ ] **Step 3: Reuse the existing reward rules to rank candidates**

Minimum reward logic:

```python
def reward(candidate: str, target: str, rank_idx: int) -> float:
    if candidate.strip() == target.strip():
        return 1.0
    return -1.0 / math.log2(rank_idx + 2)
```

- [ ] **Step 4: Write one chosen/rejected pair per prompt**

Selection rule:

```python
chosen = best_valid_candidate
rejected = worst_valid_candidate if any_valid else first_invalid_candidate
```

- [ ] **Step 5: Add a DPO dataset loader to `data.py`**

Target dataset shape:

```python
class DPODataset(Dataset):
    def __init__(self, jsonl_path: str):
        ...
    def __getitem__(self, idx):
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }
```

- [ ] **Step 6: Run a pair-generation smoke test**

Run:

```bash
python build_dpo_pairs.py --model_path output/research/sft_smoke --input_csv "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_jsonl output/research/dpo_pairs_smoke.jsonl --num_beams 4 --max_new_tokens 16
```

Expected outcome: output JSONL exists and has at least 100 valid rows.

- [ ] **Step 7: Commit pair generation**

```bash
git add build_dpo_pairs.py data.py
git commit -m "feat: add DPO pair generation from SFT outputs"
```

---

### Task 4: Add DPO Training As A Separate Stage

**Files:**
- Create: `D:\code\MiniOneRec_v2\dpo.py`
- Create: `D:\code\MiniOneRec_v2\dpo.sh`
- Create: `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_dpo.yaml`

- [ ] **Step 1: Create the DPO training entrypoint around TRL DPOTrainer**

Target imports:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
from data import DPODataset
```

- [ ] **Step 2: Use the SFT checkpoint as the DPO starting model**

Target model load shape:

```python
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
```

- [ ] **Step 3: Add PEFT around the DPO model**

Target block:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
)
```

- [ ] **Step 4: Add a conservative DPO config**

Use this starting config:

```python
training_args = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    num_train_epochs=1,
    bf16=True,
    max_prompt_length=512,
    max_length=544,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
)
```

- [ ] **Step 5: Create the launch script**

Use this skeleton:

```bash
python dpo.py \
  --model_path output/research/E1_sft/final_checkpoint \
  --train_jsonl output/research/dpo_pairs_train.jsonl \
  --eval_jsonl output/research/dpo_pairs_valid.jsonl \
  --output_dir output/research/E2_dpo
```

- [ ] **Step 6: Smoke test DPO**

Run:

```bash
python dpo.py --model_path output/research/sft_smoke --train_jsonl output/research/dpo_pairs_smoke.jsonl --eval_jsonl output/research/dpo_pairs_smoke.jsonl --output_dir output/research/dpo_smoke
```

Expected outcome: one checkpoint and no OOM under the target hardware.

- [ ] **Step 7: Commit DPO stage**

```bash
git add dpo.py dpo.sh configs/research/qwen25_3b_dpo.yaml
git commit -m "feat: add DPO stage for recommendation preference optimization"
```

---

### Task 5: Adapt GRPO For The 3B Research Route

**Files:**
- Modify: `D:\code\MiniOneRec_v2\rl.py`
- Modify: `D:\code\MiniOneRec_v2\rl.sh`
- Modify: `D:\code\MiniOneRec_v2\config\zero2_opt.yaml`
- Create: `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_grpo.yaml`

- [ ] **Step 1: Lower the default GRPO rollout pressure**

Target research defaults:

```python
num_generations: int = 4
train_batch_size: int = 1
eval_batch_size: int = 1
gradient_accumulation_steps: int = 16
max_completion_length = 16
```

- [ ] **Step 2: Make the model path configurable for both SFT and DPO checkpoints**

Required experiment paths:

```text
E3: model_path = output/research/E1_sft/final_checkpoint
E4: model_path = output/research/E2_dpo/final_checkpoint
```

- [ ] **Step 3: Add one research config file for GRPO**

Use this config skeleton:

```yaml
model_path: output/research/E1_sft/final_checkpoint
train_batch_size: 1
eval_batch_size: 1
gradient_accumulation_steps: 16
num_generations: 4
num_train_epochs: 1
learning_rate: 5e-6
beta: 1e-3
beam_search: true
reward_type: ranking
```

- [ ] **Step 4: Run a GRPO smoke test from the SFT checkpoint**

Run:

```bash
python rl.py --model_path output/research/sft_smoke --train_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --eval_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --category Industrial_and_Scientific --train_batch_size 1 --eval_batch_size 1 --gradient_accumulation_steps 16 --num_generations 4 --num_train_epochs 1 --sid_index_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json" --item_meta_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json" --output_dir output/research/grpo_smoke
```

Expected outcome: at least one training save without OOM.

- [ ] **Step 5: Commit GRPO research settings**

```bash
git add rl.py rl.sh config/zero2_opt.yaml configs/research/qwen25_3b_grpo.yaml
git commit -m "feat: add memory-safe GRPO settings for 3B research route"
```

---

### Task 6: Expand Offline Evaluation Beyond HR/NDCG

**Files:**
- Create: `D:\code\MiniOneRec_v2\analysis\metrics_plus.py`
- Modify: `D:\code\MiniOneRec_v2\calc.py`
- Modify: `D:\code\MiniOneRec_v2\evaluate.sh`

- [ ] **Step 1: Preserve the existing HR/NDCG path**

Do not remove `calc.py` behavior. Wrap or extend it.

- [ ] **Step 2: Add invalid generation rate**

Target formula:

```python
invalid_rate = invalid_prediction_count / total_predictions
```

- [ ] **Step 3: Add coverage**

Target formula:

```python
coverage = len(unique_predicted_items) / len(all_catalog_items)
```

- [ ] **Step 4: Add per-sample export for downstream analysis**

Target output row:

```json
{
  "target": "<a_1><b_9><c_3>",
  "predict": ["<a_1><b_9><c_3>", "<a_5><b_2><c_7>"],
  "hit@10": 1,
  "ndcg@10": 1.0,
  "is_invalid_top1": 0
}
```

- [ ] **Step 5: Add a one-command research evaluation path**

Run pattern:

```bash
python analysis/metrics_plus.py --result_json results/<exp>/final_result_Industrial_and_Scientific.json --item_info <info_txt> --output_json results/<exp>/metrics_plus.json
```

- [ ] **Step 6: Commit expanded metrics**

```bash
git add analysis/metrics_plus.py calc.py evaluate.sh
git commit -m "feat: add invalid-rate and coverage evaluation"
```

---

### Task 7: Add Bucket Analysis For Items And Users

**Files:**
- Create: `D:\code\MiniOneRec_v2\analysis\bucket_analysis.py`
- Modify: `D:\code\MiniOneRec_v2\convert_dataset.py`
- Modify: `D:\code\MiniOneRec_v2\data.py`

- [ ] **Step 1: Define item popularity buckets**

Use fixed bucket rules:

```python
def item_pop_bucket(freq_rank_ratio: float) -> str:
    if freq_rank_ratio <= 0.2:
        return "head"
    if freq_rank_ratio <= 0.8:
        return "mid"
    return "tail"
```

- [ ] **Step 2: Define user-history buckets**

Use fixed rules:

```python
def history_bucket(length: int) -> str:
    if length <= 3:
        return "short"
    if length <= 10:
        return "medium"
    return "long"
```

- [ ] **Step 3: Ensure converted CSV rows retain enough metadata**

Required columns:

```text
user_id
history_item_id
item_id
history_item_sid
item_sid
history_len
```

- [ ] **Step 4: Compute bucket-level metrics from final eval JSON**

Target output:

```json
{
  "item_bucket": {
    "head": {"hr@10": 0.31, "ndcg@10": 0.18},
    "mid": {"hr@10": 0.27, "ndcg@10": 0.14},
    "tail": {"hr@10": 0.19, "ndcg@10": 0.09}
  },
  "history_bucket": {
    "short": {"hr@10": 0.16, "ndcg@10": 0.08},
    "medium": {"hr@10": 0.25, "ndcg@10": 0.13},
    "long": {"hr@10": 0.34, "ndcg@10": 0.20}
  }
}
```

- [ ] **Step 5: Commit bucket analysis**

```bash
git add analysis/bucket_analysis.py convert_dataset.py data.py
git commit -m "feat: add item and user bucket analysis for recommendation results"
```

---

### Task 8: Run The Four Main Experiments In Order

**Files:**
- Modify: `D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`

- [ ] **Step 1: Run `E1 = SFT`**

Run:

```bash
python sft.py --base_model Qwen/Qwen2.5-3B --train_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --eval_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --sid_index_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json" --item_meta_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json" --output_dir output/research/E1_sft --micro_batch_size 1 --batch_size 64 --num_epochs 2 --use_lora True
```

Output to record:

```text
output/research/E1_sft/final_checkpoint
```

- [ ] **Step 2: Run `E2 = SFT + DPO`**

Run:

```bash
python build_dpo_pairs.py --model_path output/research/E1_sft/final_checkpoint --input_csv "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_jsonl output/research/E2_pairs_train.jsonl
python dpo.py --model_path output/research/E1_sft/final_checkpoint --train_jsonl output/research/E2_pairs_train.jsonl --eval_jsonl output/research/E2_pairs_valid.jsonl --output_dir output/research/E2_dpo
```

- [ ] **Step 3: Run `E3 = SFT + GRPO`**

Run:

```bash
python rl.py --model_path output/research/E1_sft/final_checkpoint --train_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --eval_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --category Industrial_and_Scientific --train_batch_size 1 --eval_batch_size 1 --gradient_accumulation_steps 16 --num_generations 4 --num_train_epochs 1 --sid_index_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json" --item_meta_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json" --output_dir output/research/E3_grpo
```

- [ ] **Step 4: Run `E4 = SFT + DPO + GRPO`**

Run:

```bash
python rl.py --model_path output/research/E2_dpo/final_checkpoint --train_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --eval_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --category Industrial_and_Scientific --train_batch_size 1 --eval_batch_size 1 --gradient_accumulation_steps 16 --num_generations 4 --num_train_epochs 1 --sid_index_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json" --item_meta_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json" --output_dir output/research/E4_dpo_grpo
```

- [ ] **Step 5: Fill the experiment tracker after each run**

Required fields:

```text
checkpoint path
hardware
runtime
best validation checkpoint
eval output path
notes on OOM or instability
```

- [ ] **Step 6: Commit the experiment tracker updates**

```bash
git add docs/research/2026-04-18-experiment-matrix.md
git commit -m "docs: record SFT DPO and GRPO experiment runs"
```

---

### Task 9: Evaluate All Four Experiments With The Same Protocol

**Files:**
- Modify: `D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`
- Create: `D:\code\MiniOneRec_v2\analysis\collect_results.py`

- [ ] **Step 1: Evaluate each checkpoint with the same decoding setup**

Use one fixed setting first:

```text
batch_size = 1
num_beams = 10
max_new_tokens = 32
length_penalty = 0.0
```

- [ ] **Step 2: Run base metrics and expanded metrics for E1-E4**

Per experiment:

```bash
python evaluate.py --base_model output/research/E1_sft/final_checkpoint --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --category Industrial_and_Scientific --test_data_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\test\Industrial_and_Scientific_5_2016-10-2018-11.csv" --result_json_data results/E1/final_result.json --batch_size 1 --num_beams 10 --max_new_tokens 32 --length_penalty 0.0
python calc.py --path results/E1/final_result.json --item_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt"
python analysis/metrics_plus.py --result_json results/E1/final_result.json --item_info "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_json results/E1/metrics_plus.json
python analysis/bucket_analysis.py --result_json results/E1/final_result.json --test_csv "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\test\Industrial_and_Scientific_5_2016-10-2018-11.csv" --item_info "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_json results/E1/bucket_metrics.json

python evaluate.py --base_model output/research/E2_dpo/final_checkpoint --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --category Industrial_and_Scientific --test_data_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\test\Industrial_and_Scientific_5_2016-10-2018-11.csv" --result_json_data results/E2/final_result.json --batch_size 1 --num_beams 10 --max_new_tokens 32 --length_penalty 0.0
python calc.py --path results/E2/final_result.json --item_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt"
python analysis/metrics_plus.py --result_json results/E2/final_result.json --item_info "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_json results/E2/metrics_plus.json
python analysis/bucket_analysis.py --result_json results/E2/final_result.json --test_csv "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\test\Industrial_and_Scientific_5_2016-10-2018-11.csv" --item_info "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_json results/E2/bucket_metrics.json

python evaluate.py --base_model output/research/E3_grpo/final_checkpoint --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --category Industrial_and_Scientific --test_data_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\test\Industrial_and_Scientific_5_2016-10-2018-11.csv" --result_json_data results/E3/final_result.json --batch_size 1 --num_beams 10 --max_new_tokens 32 --length_penalty 0.0
python calc.py --path results/E3/final_result.json --item_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt"
python analysis/metrics_plus.py --result_json results/E3/final_result.json --item_info "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_json results/E3/metrics_plus.json
python analysis/bucket_analysis.py --result_json results/E3/final_result.json --test_csv "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\test\Industrial_and_Scientific_5_2016-10-2018-11.csv" --item_info "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_json results/E3/bucket_metrics.json

python evaluate.py --base_model output/research/E4_dpo_grpo/final_checkpoint --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --category Industrial_and_Scientific --test_data_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\test\Industrial_and_Scientific_5_2016-10-2018-11.csv" --result_json_data results/E4/final_result.json --batch_size 1 --num_beams 10 --max_new_tokens 32 --length_penalty 0.0
python calc.py --path results/E4/final_result.json --item_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt"
python analysis/metrics_plus.py --result_json results/E4/final_result.json --item_info "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_json results/E4/metrics_plus.json
python analysis/bucket_analysis.py --result_json results/E4/final_result.json --test_csv "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\test\Industrial_and_Scientific_5_2016-10-2018-11.csv" --item_info "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_json results/E4/bucket_metrics.json
```

- [ ] **Step 3: Aggregate all results into one table**

Target output table:

```csv
experiment,hr@10,ndcg@10,invalid_rate,coverage,head_hr@10,mid_hr@10,tail_hr@10,short_hr@10,long_hr@10
E1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
```

- [ ] **Step 4: Create the results collector**

Target CLI:

```bash
python analysis/collect_results.py --results_root results --output_csv results/research_summary.csv
```

- [ ] **Step 5: Commit evaluation aggregation**

```bash
git add analysis/collect_results.py docs/research/2026-04-18-experiment-matrix.md
git commit -m "feat: aggregate unified metrics for four-stage research comparison"
```

---

### Task 10: Produce Interview-Ready Outputs

**Files:**
- Modify: `D:\code\MiniOneRec_v2\README.md`
- Create: `D:\code\MiniOneRec_v2\docs\research\2026-04-18-final-summary.md`

- [ ] **Step 1: Create a final results summary template**

Use this section layout:

```markdown
# Final Summary

## Setup
- Base model
- Hardware
- SID method
- Decoding setup

## Main Table
- SFT
- SFT+DPO
- SFT+GRPO
- SFT+DPO+GRPO

## Key Findings
1. Which stage improved HR/NDCG
2. Which stage reduced invalid rate
3. Which stage improved tail items
4. Which stage helped short-history users

## Failure Modes
- Invalid generation
- Popularity bias
- Weak short-history performance
- DPO/GRPO instability if observed
```

- [ ] **Step 2: Add one README section for the new pipeline**

Target text outline:

```markdown
## Research Extension: SFT -> DPO -> GRPO

This repository now supports a four-way comparison:
- SFT
- SFT + DPO
- SFT + GRPO
- SFT + DPO + GRPO
```

- [ ] **Step 3: Fill the summary only after metrics are frozen**

Required claims:

```text
- no metric claim without result file path
- no improvement claim without comparison baseline
- note the final decoding setting used
```

- [ ] **Step 4: Commit the final report**

```bash
git add README.md docs/research/2026-04-18-final-summary.md
git commit -m "docs: summarize SFT DPO GRPO recommendation research results"
```

---

## Execution Order

1. Task 1: baseline/runtime guardrails
2. Task 2: PEFT SFT path
3. Task 3: DPO pair generation
4. Task 4: DPO training
5. Task 5: research-safe GRPO
6. Task 6: expanded metrics
7. Task 7: bucket analysis
8. Task 8: run E1-E4
9. Task 9: unified evaluation and aggregation
10. Task 10: final summary and README

## Risks And Controls

- **Risk:** `3B` still OOMs on 24G cards during DPO/GRPO  
  **Control:** start with PEFT, `batch_size=1`, `num_generations=4`, `max_completion_length=16`

- **Risk:** DPO pairs are noisy because chosen/rejected quality is poor  
  **Control:** generate from SFT validation samples first, inspect at least 100 rows manually

- **Risk:** GRPO signal is unstable on top of DPO  
  **Control:** run `SFT+GRPO` before `SFT+DPO+GRPO`, keep reward type fixed as `ranking`

- **Risk:** final comparison is invalid because decode settings drift  
  **Control:** freeze one eval config and reuse it for E1-E4 before any final high-beam rerun

- **Risk:** bucket analysis is not reproducible  
  **Control:** fix deterministic bucket boundaries and export per-sample metadata

## Success Criteria

- All four experiments produce checkpoints and result JSONs
- Every experiment is evaluated under the same decode settings
- Final summary includes:
  - one main table
  - invalid rate
  - coverage
  - head/mid/tail performance
  - short/medium/long history performance
- The project can be explained as a coherent recommendation research pipeline, not just a set of scripts
