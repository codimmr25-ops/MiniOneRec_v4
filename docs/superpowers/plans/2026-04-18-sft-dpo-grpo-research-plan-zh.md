# MiniOneRec SFT-DPO-GRPO 中文研究计划

> **给执行型 Agent 的要求：** 实施本计划时，必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 子技能，按任务逐项执行。步骤统一使用复选框 `- [ ]` 跟踪。

**目标：** 在 MiniOneRec 当前 `SFT -> GRPO` 流程的基础上，扩展出完整的 `SFT -> DPO -> GRPO` 推荐研究链路，并完成四组受控实验以及统一评估。

**架构思路：** 保持现有 SID 构造、约束解码和离线评估主线不变，在 SFT 与 GRPO 之间增加偏好数据构造和 DPO 训练阶段。同时补齐统一评估脚本，使所有模型在同一套指标、同一套分桶规则、同一套解码设置下对比。

**技术栈：** Python、PyTorch、Transformers、TRL、Accelerate、DeepSpeed、pandas、numpy

---

## 一、范围与前置假设

- 基础模型：`Qwen2.5-3B`
- 目标数据集：`Industrial_and_Scientific`
- 训练硬件优先级：
  - 优先：`1x H800 80G/94G` 或 `1x A800 80G`
  - 次选：`4x RTX 4090 24G`，但必须配合 PEFT 和低显存参数
- 实验矩阵：
  - `SFT`
  - `SFT + DPO`
  - `SFT + GRPO`
  - `SFT + DPO + GRPO`
- 统一评估指标：
  - `HR@K`
  - `NDCG@K`
  - `invalid generation rate`
  - `coverage`
  - `head/mid/tail` 商品分桶表现
  - `short/medium/long` 用户历史长度分桶表现

### 默认研究路径

- `TRAIN_CSV = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv`
- `VALID_CSV = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv`
- `TEST_CSV = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\test\Industrial_and_Scientific_5_2016-10-2018-11.csv`
- `INFO_TXT = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt`
- `SID_JSON = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json`
- `ITEM_JSON = D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json`

---

## 二、文件结构与职责划分

### 现有文件，直接复用

- `D:\code\MiniOneRec_v2\sft.py`
  - 当前 SFT 训练入口
- `D:\code\MiniOneRec_v2\rl.py`
  - 当前 GRPO 训练入口
- `D:\code\MiniOneRec_v2\minionerec_trainer.py`
  - 当前推荐场景的自定义 GRPO Trainer
- `D:\code\MiniOneRec_v2\data.py`
  - 当前 SFT、RL、Eval 的数据集定义
- `D:\code\MiniOneRec_v2\evaluate.py`
  - 当前约束解码评估脚本
- `D:\code\MiniOneRec_v2\calc.py`
  - 当前 HR/NDCG 计算脚本，包含无效生成统计逻辑
- `D:\code\MiniOneRec_v2\config\zero2_opt.yaml`
  - 当前 RL 阶段的 DeepSpeed ZeRO-2 配置

### 新增文件

- `D:\code\MiniOneRec_v2\dpo.py`
  - DPO 训练入口
- `D:\code\MiniOneRec_v2\dpo.sh`
  - DPO 启动脚本
- `D:\code\MiniOneRec_v2\build_dpo_pairs.py`
  - 从 SFT 输出构造 chosen/rejected 偏好对
- `D:\code\MiniOneRec_v2\analysis\metrics_plus.py`
  - 统一指标计算：invalid rate、coverage、辅助明细
- `D:\code\MiniOneRec_v2\analysis\bucket_analysis.py`
  - 商品热度分桶、用户历史长度分桶分析
- `D:\code\MiniOneRec_v2\analysis\collect_results.py`
  - 聚合四组实验结果为总表
- `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_sft.yaml`
  - 3B 研究版 SFT 配置
- `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_dpo.yaml`
  - 3B 研究版 DPO 配置
- `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_grpo.yaml`
  - 3B 研究版 GRPO 配置
- `D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`
  - 实验跟踪表和结果登记页

### 需要修改的现有文件

- `D:\code\MiniOneRec_v2\requirements.txt`
  - 增加 DPO/PEFT 依赖
- `D:\code\MiniOneRec_v2\data.py`
  - 增加 DPO 数据集以及分桶辅助逻辑
- `D:\code\MiniOneRec_v2\sft.sh`
  - 去掉硬编码 8 卡假设
- `D:\code\MiniOneRec_v2\rl.sh`
  - 改成适合 3B 的低显存版本
- `D:\code\MiniOneRec_v2\evaluate.sh`
  - 增加研究版统一评估入口
- `D:\code\MiniOneRec_v2\README.md`
  - 在结果稳定后补充研究扩展说明

---

## 三、任务拆解

### 任务 1：建立研究基线与运行约束

**涉及文件：**
- 新建：`D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`
- 修改：`D:\code\MiniOneRec_v2\sft.sh`
- 修改：`D:\code\MiniOneRec_v2\rl.sh`
- 修改：`D:\code\MiniOneRec_v2\evaluate.sh`

- [ ] **步骤 1：建立固定实验编号的实验跟踪表**

目标内容：

```markdown
# 2026-04-18 Experiment Matrix

| Exp ID | Pipeline | Base Model | SID Variant | Train Checkpoint | Eval Result JSON | Notes |
|---|---|---|---|---|---|---|
| E1 | SFT | Qwen2.5-3B | rqvae |  |  |  |
| E2 | SFT+DPO | Qwen2.5-3B | rqvae |  |  |  |
| E3 | SFT+GRPO | Qwen2.5-3B | rqvae |  |  |  |
| E4 | SFT+DPO+GRPO | Qwen2.5-3B | rqvae |  |  |  |
```

- [ ] **步骤 2：去掉所有默认 8 卡写法，改成显式变量**

目标形态：

```bash
NPROC=${NPROC:-1}
CUDA_IDS=${CUDA_IDS:-0}
```

- [ ] **步骤 3：先固定 3B 研究路线的保守参数**

基线参数：

```text
SFT:
- micro_batch_size = 1 或 2
- gradient_accumulation_steps 放大到目标等效 batch

DPO:
- per_device_train_batch_size = 1
- 使用 PEFT
- 限制 prompt 长度与 target 长度

GRPO:
- per_device_train_batch_size = 1 或 2
- num_generations 先用 4，稳定后再试 8
- max_completion_length 先用 16 或 32

Eval:
- batch_size = 1 或 2
- num_beams 先用 10，最终可试 20
```

- [ ] **步骤 4：把现有 CLI 参数先固化下来**

执行：

```bash
python sft.py --help
python rl.py --help
python evaluate.py --help
```

预期：后续新增代码之前，先把现有参数和运行方式整理清楚。

- [ ] **步骤 5：提交基线规划变更**

```bash
git add docs/research/2026-04-18-experiment-matrix.md sft.sh rl.sh evaluate.sh
git commit -m "docs: define research experiment matrix and runtime constraints"
```

---

### 任务 2：把 SFT 改造成适合 3B 的研究起点

**涉及文件：**
- 修改：`D:\code\MiniOneRec_v2\sft.py`
- 修改：`D:\code\MiniOneRec_v2\requirements.txt`
- 新建：`D:\code\MiniOneRec_v2\configs\research\qwen25_3b_sft.yaml`

- [ ] **步骤 1：给 SFT 增加 PEFT/LoRA 参数**

目标参数块：

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

- [ ] **步骤 2：补 PEFT 依赖**

```text
peft>=0.14.0
```

- [ ] **步骤 3：在 Trainer 初始化前套上 LoRA**

目标逻辑：

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

- [ ] **步骤 4：启用 gradient checkpointing**

```python
if gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
```

- [ ] **步骤 5：新建研究版 SFT 配置**

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

- [ ] **步骤 6：先跑一个小规模 smoke test**

```bash
python sft.py --base_model Qwen/Qwen2.5-3B --train_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --eval_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --sid_index_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json" --item_meta_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json" --output_dir output/research/sft_smoke --micro_batch_size 1 --batch_size 8 --num_epochs 1 --use_lora True
```

预期：成功保存一个小模型 checkpoint。

- [ ] **步骤 7：提交 SFT 研究路径**

```bash
git add sft.py requirements.txt configs/research/qwen25_3b_sft.yaml
git commit -m "feat: add memory-safe research SFT path for 3B"
```

---

### 任务 3：从 SFT 结果构造 DPO 偏好对

**涉及文件：**
- 新建：`D:\code\MiniOneRec_v2\build_dpo_pairs.py`
- 修改：`D:\code\MiniOneRec_v2\data.py`

- [ ] **步骤 1：定义 DPO pair 的 JSONL 格式**

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

- [ ] **步骤 2：实现从 SFT checkpoint 生成候选**

目标 CLI：

```python
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--info_file", type=str, required=True)
parser.add_argument("--output_jsonl", type=str, required=True)
parser.add_argument("--num_beams", type=int, default=8)
parser.add_argument("--max_new_tokens", type=int, default=32)
```

- [ ] **步骤 3：复用现有 reward 规则给候选打分**

最小逻辑：

```python
def reward(candidate: str, target: str, rank_idx: int) -> float:
    if candidate.strip() == target.strip():
        return 1.0
    return -1.0 / math.log2(rank_idx + 2)
```

- [ ] **步骤 4：为每个 prompt 选一个 chosen 和 rejected**

规则：

```python
chosen = best_valid_candidate
rejected = worst_valid_candidate if any_valid else first_invalid_candidate
```

- [ ] **步骤 5：在 `data.py` 增加 DPO 数据集类**

目标形态：

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

- [ ] **步骤 6：跑一个 pair-generation smoke test**

```bash
python build_dpo_pairs.py --model_path output/research/sft_smoke --input_csv "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_jsonl output/research/dpo_pairs_smoke.jsonl --num_beams 4 --max_new_tokens 16
```

预期：输出至少 100 条可用 pair。

- [ ] **步骤 7：提交 DPO 数据构造**

```bash
git add build_dpo_pairs.py data.py
git commit -m "feat: add DPO pair generation from SFT outputs"
```

---

### 任务 4：新增 DPO 训练阶段

**涉及文件：**
- 新建：`D:\code\MiniOneRec_v2\dpo.py`
- 新建：`D:\code\MiniOneRec_v2\dpo.sh`
- 新建：`D:\code\MiniOneRec_v2\configs\research\qwen25_3b_dpo.yaml`

- [ ] **步骤 1：基于 TRL DPOTrainer 搭建训练入口**

核心 import：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
from data import DPODataset
```

- [ ] **步骤 2：把 SFT checkpoint 作为 DPO 起点**

```python
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
```

- [ ] **步骤 3：在 DPO 阶段也使用 LoRA**

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

- [ ] **步骤 4：使用保守的 DPO 参数**

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

- [ ] **步骤 5：增加 DPO 启动脚本**

```bash
python dpo.py \
  --model_path output/research/E1_sft/final_checkpoint \
  --train_jsonl output/research/dpo_pairs_train.jsonl \
  --eval_jsonl output/research/dpo_pairs_valid.jsonl \
  --output_dir output/research/E2_dpo
```

- [ ] **步骤 6：先做 DPO smoke test**

```bash
python dpo.py --model_path output/research/sft_smoke --train_jsonl output/research/dpo_pairs_smoke.jsonl --eval_jsonl output/research/dpo_pairs_smoke.jsonl --output_dir output/research/dpo_smoke
```

预期：在目标硬件上不 OOM，并产生 checkpoint。

- [ ] **步骤 7：提交 DPO 阶段**

```bash
git add dpo.py dpo.sh configs/research/qwen25_3b_dpo.yaml
git commit -m "feat: add DPO stage for recommendation preference optimization"
```

---

### 任务 5：把 GRPO 调整到适合 3B 的研究版本

**涉及文件：**
- 修改：`D:\code\MiniOneRec_v2\rl.py`
- 修改：`D:\code\MiniOneRec_v2\rl.sh`
- 修改：`D:\code\MiniOneRec_v2\config\zero2_opt.yaml`
- 新建：`D:\code\MiniOneRec_v2\configs\research\qwen25_3b_grpo.yaml`

- [ ] **步骤 1：降低默认 rollout 压力**

```python
num_generations: int = 4
train_batch_size: int = 1
eval_batch_size: int = 1
gradient_accumulation_steps: int = 16
max_completion_length = 16
```

- [ ] **步骤 2：让 GRPO 能接受 SFT 和 DPO 两种起点**

固定约定：

```text
E3: model_path = output/research/E1_sft/final_checkpoint
E4: model_path = output/research/E2_dpo/final_checkpoint
```

- [ ] **步骤 3：新建研究版 GRPO 配置**

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

- [ ] **步骤 4：先从 SFT checkpoint 跑一个 GRPO smoke test**

```bash
python rl.py --model_path output/research/sft_smoke --train_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --eval_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --category Industrial_and_Scientific --train_batch_size 1 --eval_batch_size 1 --gradient_accumulation_steps 16 --num_generations 4 --num_train_epochs 1 --sid_index_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json" --item_meta_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json" --output_dir output/research/grpo_smoke
```

预期：能完成至少一个保存周期。

- [ ] **步骤 5：提交 GRPO 研究版**

```bash
git add rl.py rl.sh config/zero2_opt.yaml configs/research/qwen25_3b_grpo.yaml
git commit -m "feat: add memory-safe GRPO settings for 3B research route"
```

---

### 任务 6：扩展离线评估指标

**涉及文件：**
- 新建：`D:\code\MiniOneRec_v2\analysis\metrics_plus.py`
- 修改：`D:\code\MiniOneRec_v2\calc.py`
- 修改：`D:\code\MiniOneRec_v2\evaluate.sh`

- [ ] **步骤 1：保留现有 HR/NDCG 逻辑，不做破坏式修改**

- [ ] **步骤 2：增加 invalid generation rate**

```python
invalid_rate = invalid_prediction_count / total_predictions
```

- [ ] **步骤 3：增加 coverage**

```python
coverage = len(unique_predicted_items) / len(all_catalog_items)
```

- [ ] **步骤 4：为后续分桶分析保留逐样本明细**

```json
{
  "target": "<a_1><b_9><c_3>",
  "predict": ["<a_1><b_9><c_3>", "<a_5><b_2><c_7>"],
  "hit@10": 1,
  "ndcg@10": 1.0,
  "is_invalid_top1": 0
}
```

- [ ] **步骤 5：增加研究版统一评估入口**

```bash
python analysis/metrics_plus.py --result_json results/<exp>/final_result_Industrial_and_Scientific.json --item_info <info_txt> --output_json results/<exp>/metrics_plus.json
```

- [ ] **步骤 6：提交扩展指标**

```bash
git add analysis/metrics_plus.py calc.py evaluate.sh
git commit -m "feat: add invalid-rate and coverage evaluation"
```

---

### 任务 7：补齐商品和用户分桶分析

**涉及文件：**
- 新建：`D:\code\MiniOneRec_v2\analysis\bucket_analysis.py`
- 修改：`D:\code\MiniOneRec_v2\convert_dataset.py`
- 修改：`D:\code\MiniOneRec_v2\data.py`

- [ ] **步骤 1：定义商品热度分桶**

```python
def item_pop_bucket(freq_rank_ratio: float) -> str:
    if freq_rank_ratio <= 0.2:
        return "head"
    if freq_rank_ratio <= 0.8:
        return "mid"
    return "tail"
```

- [ ] **步骤 2：定义用户历史长度分桶**

```python
def history_bucket(length: int) -> str:
    if length <= 3:
        return "short"
    if length <= 10:
        return "medium"
    return "long"
```

- [ ] **步骤 3：保证转换后的 CSV 保留足够的分析字段**

至少保留：

```text
user_id
history_item_id
item_id
history_item_sid
item_sid
history_len
```

- [ ] **步骤 4：从评估结果中产出分桶指标**

目标输出：

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

- [ ] **步骤 5：提交分桶分析**

```bash
git add analysis/bucket_analysis.py convert_dataset.py data.py
git commit -m "feat: add item and user bucket analysis for recommendation results"
```

---

### 任务 8：按顺序运行四组主实验

**涉及文件：**
- 修改：`D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`

- [ ] **步骤 1：运行 `E1 = SFT`**

```bash
python sft.py --base_model Qwen/Qwen2.5-3B --train_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --eval_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --sid_index_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json" --item_meta_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json" --output_dir output/research/E1_sft --micro_batch_size 1 --batch_size 64 --num_epochs 2 --use_lora True
```

目标输出：

```text
output/research/E1_sft/final_checkpoint
```

- [ ] **步骤 2：运行 `E2 = SFT + DPO`**

```bash
python build_dpo_pairs.py --model_path output/research/E1_sft/final_checkpoint --input_csv "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --output_jsonl output/research/E2_pairs_train.jsonl
python dpo.py --model_path output/research/E1_sft/final_checkpoint --train_jsonl output/research/E2_pairs_train.jsonl --eval_jsonl output/research/E2_pairs_valid.jsonl --output_dir output/research/E2_dpo
```

- [ ] **步骤 3：运行 `E3 = SFT + GRPO`**

```bash
python rl.py --model_path output/research/E1_sft/final_checkpoint --train_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --eval_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --category Industrial_and_Scientific --train_batch_size 1 --eval_batch_size 1 --gradient_accumulation_steps 16 --num_generations 4 --num_train_epochs 1 --sid_index_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json" --item_meta_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json" --output_dir output/research/E3_grpo
```

- [ ] **步骤 4：运行 `E4 = SFT + DPO + GRPO`**

```bash
python rl.py --model_path output/research/E2_dpo/final_checkpoint --train_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\train\Industrial_and_Scientific_5_2016-10-2018-11.csv" --eval_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\valid\Industrial_and_Scientific_5_2016-10-2018-11.csv" --info_file "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\converted\rqvae\info\Industrial_and_Scientific_5_2016-10-2018-11.txt" --category Industrial_and_Scientific --train_batch_size 1 --eval_batch_size 1 --gradient_accumulation_steps 16 --num_generations 4 --num_train_epochs 1 --sid_index_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\sid_variants\rqvae\Industrial_and_Scientific.index.json" --item_meta_path "D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\shared\Industrial_and_Scientific.item.json" --output_dir output/research/E4_dpo_grpo
```

- [ ] **步骤 5：每跑完一组实验，都更新实验记录表**

必须记录：

```text
checkpoint 路径
硬件
运行时长
最佳验证 checkpoint
评估结果路径
OOM 或训练不稳定备注
```

- [ ] **步骤 6：提交实验记录**

```bash
git add docs/research/2026-04-18-experiment-matrix.md
git commit -m "docs: record SFT DPO and GRPO experiment runs"
```

---

### 任务 9：用同一套协议评估四组实验

**涉及文件：**
- 修改：`D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`
- 新建：`D:\code\MiniOneRec_v2\analysis\collect_results.py`

- [ ] **步骤 1：冻结统一解码配置**

统一设置：

```text
batch_size = 1
num_beams = 10
max_new_tokens = 32
length_penalty = 0.0
```

- [ ] **步骤 2：对 E1-E4 分别跑基础评估、扩展指标和分桶分析**

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

- [ ] **步骤 3：把四组结果聚合成一张总表**

目标表头：

```csv
experiment,hr@10,ndcg@10,invalid_rate,coverage,head_hr@10,mid_hr@10,tail_hr@10,short_hr@10,long_hr@10
E1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
```

- [ ] **步骤 4：新建结果聚合脚本**

```bash
python analysis/collect_results.py --results_root results --output_csv results/research_summary.csv
```

- [ ] **步骤 5：提交统一评估聚合**

```bash
git add analysis/collect_results.py docs/research/2026-04-18-experiment-matrix.md
git commit -m "feat: aggregate unified metrics for four-stage research comparison"
```

---

### 任务 10：整理面试可用的最终产物

**涉及文件：**
- 修改：`D:\code\MiniOneRec_v2\README.md`
- 新建：`D:\code\MiniOneRec_v2\docs\research\2026-04-18-final-summary.md`

- [ ] **步骤 1：建立最终总结文档模板**

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
1. 哪个阶段提升了 HR/NDCG
2. 哪个阶段降低了 invalid rate
3. 哪个阶段改善了 tail item
4. 哪个阶段改善了短历史用户

## Failure Modes
- 无效生成
- 热门偏置
- 短历史用户表现弱
- DPO/GRPO 不稳定现象
```

- [ ] **步骤 2：在 README 里补一节研究扩展说明**

```markdown
## Research Extension: SFT -> DPO -> GRPO

This repository now supports a four-way comparison:
- SFT
- SFT + DPO
- SFT + GRPO
- SFT + DPO + GRPO
```

- [ ] **步骤 3：只有在指标冻结后再填写总结**

必须遵守：

```text
- 没有结果文件路径，不写指标结论
- 没有基线对比，不写“提升”
- 必须写清最终使用的解码设置
```

- [ ] **步骤 4：提交最终总结**

```bash
git add README.md docs/research/2026-04-18-final-summary.md
git commit -m "docs: summarize SFT DPO GRPO recommendation research results"
```

---

## 四、执行顺序

1. 任务 1：建立研究基线与运行约束
2. 任务 2：完成 PEFT 化 SFT
3. 任务 3：构造 DPO 偏好对
4. 任务 4：完成 DPO 训练
5. 任务 5：整理研究版 GRPO
6. 任务 6：扩展评估指标
7. 任务 7：补齐分桶分析
8. 任务 8：跑四组主实验
9. 任务 9：统一评估与聚合
10. 任务 10：输出最终总结和 README

---

## 五、风险与控制

- **风险：** `3B` 在 24G 卡上 DPO/GRPO 仍然 OOM  
  **控制：** 从 PEFT、`batch_size=1`、`num_generations=4`、`max_completion_length=16` 起步

- **风险：** DPO pair 噪声太大  
  **控制：** 先从验证集生成 pair，人工抽查至少 100 条

- **风险：** 在 DPO 基础上继续 GRPO 不稳定  
  **控制：** 先完成 `SFT+GRPO`，再跑 `SFT+DPO+GRPO`，且 reward 固定为 `ranking`

- **风险：** 四组实验评估设置不一致，导致对比无效  
  **控制：** 先冻结统一解码配置，再批量评估

- **风险：** 分桶分析不可复现  
  **控制：** 固定桶边界并导出逐样本元数据

---

## 六、成功标准

- 四组实验都能产出 checkpoint 和评估结果 JSON
- 四组实验全部使用同一套解码设置评估
- 最终总结文档至少包含：
  - 一张总对比表
  - invalid rate
  - coverage
  - head/mid/tail 指标
  - short/medium/long 指标
- 项目能被讲成一条完整的推荐研究链路，而不是若干独立脚本的堆叠
