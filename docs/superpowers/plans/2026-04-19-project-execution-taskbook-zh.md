# MiniOneRec Project Execution Taskbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定环境 `Ubuntu 22.04 / Python 3.12 / PyTorch 2.5.1 / CUDA 12.4` 下，把 MiniOneRec 当前仓库推进到可复现的 `SFT -> DPO -> GRPO` 研究项目，完成四组对照实验、统一评估和最终总结材料。

**Architecture:** 项目分成两条主线并行但有依赖。第一条是环境主线，先在低配租卡节点完成环境安装、锁文件导出和 bundle 校验，再在 `A800-80GB` 节点恢复同环境并完成 readiness gate；第二条是代码与实验主线，基于现有 `rqvae` transfer bundle 依次完成运行脚本收敛、SFT 改造、DPO 数据与训练、GRPO 收敛、评估扩展、四组实验与最终文档输出。

**Tech Stack:** Ubuntu 22.04, Miniforge, Python 3.12, PyTorch 2.5.1, CUDA 12.4, Transformers, TRL, Accelerate, DeepSpeed, bitsandbytes, pandas, numpy

---

## Project Scope

- 目标基座模型：`Qwen/Qwen2.5-3B`
- 目标数据集：`Industrial_and_Scientific`
- 当前只走 `rqvae` 主路径
- 最终实验矩阵：
  - `E1 = SFT`
  - `E2 = SFT + DPO`
  - `E3 = SFT + GRPO`
  - `E4 = SFT + DPO + GRPO`
- 最终交付物：
  - 可复现环境清单
  - 单卡 A800 可运行的研究训练路径
  - 四组实验 checkpoint
  - 统一评估结果
  - 最终总结文档和 README 扩展

## Existing Plans To Execute

- 环境计划：
  - [2026-04-19-rental-gpu-environment-setup-plan-zh.md](/D:/code/MiniOneRec_v2/docs/superpowers/plans/2026-04-19-rental-gpu-environment-setup-plan-zh.md)
- 研究代码与实验计划：
  - [2026-04-18-sft-dpo-grpo-research-plan.md](/D:/code/MiniOneRec_v2/docs/superpowers/plans/2026-04-18-sft-dpo-grpo-research-plan.md)

## Time Model

- `工程时间`：你或我真正需要投入的开发/排障/整理时间
- `GPU时间`：训练、生成 pair、评估的机器运行时间
- `日历时间`：工程时间 + GPU时间 + 排队 + 失败重跑缓冲

---

### Task 1: Freeze Project Inputs And Success Gates

**Files:**
- Reuse: `D:\code\MiniOneRec_v2\docs\superpowers\plans\2026-04-18-sft-dpo-grpo-research-plan.md`
- Reuse: `D:\code\MiniOneRec_v2\docs\superpowers\plans\2026-04-19-rental-gpu-environment-setup-plan-zh.md`
- Reuse: `D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\runbook\README_transfer.md`

**Estimate:**
- 工程时间：`0.5 天`
- GPU时间：`0`

- [ ] **Step 1: Freeze the model, dataset, and SID route**

Use this fixed baseline:

```text
base_model = Qwen/Qwen2.5-3B
dataset = Industrial_and_Scientific
sid_variant = rqvae
hardware = 1x A800-80GB
```

Expected: no later task changes these defaults unless a blocker is documented.

- [ ] **Step 2: Freeze the project root and bundle root**

Run:

```bash
cd "$HOME/MiniOneRec"
export REPO_ROOT="$PWD"
export BUNDLE_ROOT="$REPO_ROOT/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411"
```

Expected: both environment variables are set in the shell.

- [ ] **Step 3: Validate the six project-critical data paths**

Run:

```bash
python - <<'PY'
from pathlib import Path
paths = [
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/converted/rqvae/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/converted/rqvae/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/converted/rqvae/test/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/converted/rqvae/info/Industrial_and_Scientific_5_2016-10-2018-11.txt",
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/sid_variants/rqvae/Industrial_and_Scientific.index.json",
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/shared/Industrial_and_Scientific.item.json",
]
missing = [p for p in paths if not Path(p).exists()]
if missing:
    print("MISSING", missing)
    raise SystemExit(1)
print("ALL_PROJECT_INPUTS_OK")
PY
```

Expected: `ALL_PROJECT_INPUTS_OK`

- [ ] **Step 4: Freeze the final acceptance criteria**

Use this project done-definition:

```text
1. A800 environment is reproducible and locked
2. SFT, DPO, GRPO code paths all run
3. E1-E4 all produce final checkpoints
4. E1-E4 all produce unified eval outputs
5. Research summary and README are updated
```

- [ ] **Step 5: Record the kickoff gate in a note or issue**

Record:

```text
inputs frozen
paths verified
acceptance criteria frozen
```

Expected: no ambiguity remains about what counts as "done".

---

### Task 2: Build And Lock The Rental GPU Environment

**Files:**
- Reuse: `D:\code\MiniOneRec_v2\docs\superpowers\plans\2026-04-19-rental-gpu-environment-setup-plan-zh.md`
- Create: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.from-history.yml`
- Create: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.pip.txt`
- Create: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.lock.txt`
- Create: `D:\code\MiniOneRec_v2\env\validate_runtime.py`
- Create: `D:\code\MiniOneRec_v2\env\README_rental_gpu.md`

**Estimate:**
- 工程时间：`0.5 到 1 天`
- GPU时间：`2 到 4 小时`

- [ ] **Step 1: Execute environment plan Task 1 through Task 5 on the low-config node**

Run in order:

```text
Task 1: create env manifests
Task 2: provision low-config node
Task 3: create Python/Torch/CUDA environment
Task 4: validate runtime without real training
Task 5: export lockfile and wheel cache
```

Expected outputs:

```text
env/mini_one_rec_py312_torch251_cu124.from-history.yml
env/mini_one_rec_py312_torch251_cu124.pip.txt
env/mini_one_rec_py312_torch251_cu124.lock.txt
env/validate_runtime.py
env/README_rental_gpu.md
```

- [ ] **Step 2: Confirm the low-config node exit criteria**

Use this checklist:

```text
[ ] python env/validate_runtime.py succeeds
[ ] python sft.py --help succeeds
[ ] python rl.py --help succeeds
[ ] python evaluate.py --help succeeds
[ ] python -m unittest data_test.py -v succeeds
[ ] sha256sum -c report/SHA256SUMS succeeds
```

Expected: all six checks pass on the low-config node.

- [ ] **Step 3: Copy the repo, lockfiles, and optional wheelhouse to the A800 node**

Copy at minimum:

```text
MiniOneRec/
MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/
env/mini_one_rec_py312_torch251_cu124.from-history.yml
env/mini_one_rec_py312_torch251_cu124.pip.txt
env/mini_one_rec_py312_torch251_cu124.lock.txt
```

Expected: A800 node has identical repo state and transfer bundle.

- [ ] **Step 4: Execute environment plan Task 6 and Task 7 on the A800 node**

Run in order:

```text
Task 6: restore the same environment on A800
Task 7: mark the environment ready for research tasks
```

Expected:

```text
gpu_name contains A800
bf16_supported=True
project input paths verified again
```

---

### Task 3: Normalize The Runtime Entry Points For Single-GPU Research

**Files:**
- Modify: `D:\code\MiniOneRec_v2\sft.sh`
- Modify: `D:\code\MiniOneRec_v2\rl.sh`
- Modify: `D:\code\MiniOneRec_v2\evaluate.sh`
- Create: `D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`

**Estimate:**
- 工程时间：`0.5 天`
- GPU时间：`0.5 小时`

- [ ] **Step 1: Execute research plan Task 1**

Deliverables:

```text
experiment tracker created
8-GPU assumptions removed from launch scripts
single-GPU safe defaults documented
CLI arguments frozen
```

- [ ] **Step 2: Run the baseline command checks on A800**

Run:

```bash
cd "$HOME/MiniOneRec"
conda activate minionerec312
python sft.py --help
python rl.py --help
python evaluate.py --help
```

Expected: all three commands succeed on the final training machine.

- [ ] **Step 3: Mark Task 3 complete only if the launch scripts no longer assume 8 GPUs**

Use this pass condition:

```text
sft.sh can be parameterized by NPROC/CUDA_IDS
rl.sh can be parameterized by accelerate num_processes
evaluate.sh can be run in 1-GPU mode
```

---

### Task 4: Make SFT The Stable Starting Point

**Files:**
- Modify: `D:\code\MiniOneRec_v2\sft.py`
- Modify: `D:\code\MiniOneRec_v2\requirements.txt`
- Create: `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_sft.yaml`

**Estimate:**
- 工程时间：`1 到 1.5 天`
- GPU时间：`2 到 6 小时`

- [ ] **Step 1: Execute research plan Task 2**

Required outputs:

```text
SFT supports LoRA/PEFT
gradient checkpointing is available
research config exists
SFT smoke test saves a checkpoint
```

- [ ] **Step 2: Require a successful smoke checkpoint before moving on**

Run target:

```text
output/research/sft_smoke/
```

Expected:

```text
trainer starts
no import error
no immediate OOM
final checkpoint exists
```

- [ ] **Step 3: Promote SFT to experiment E1 only after smoke passes**

Run target:

```text
output/research/E1_sft/final_checkpoint
```

Expected: `E1` final checkpoint exists and is recorded in the experiment tracker.

---

### Task 5: Add DPO Data And DPO Training

**Files:**
- Create: `D:\code\MiniOneRec_v2\build_dpo_pairs.py`
- Create: `D:\code\MiniOneRec_v2\dpo.py`
- Create: `D:\code\MiniOneRec_v2\dpo.sh`
- Create: `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_dpo.yaml`
- Modify: `D:\code\MiniOneRec_v2\data.py`

**Estimate:**
- 工程时间：`1.5 到 2 天`
- GPU时间：`4 到 10 小时`

- [ ] **Step 1: Execute research plan Task 3**

Required outputs:

```text
DPO pair schema defined
pair-generation CLI created
DPODataset added to data.py
pair-generation smoke test passes
```

- [ ] **Step 2: Execute research plan Task 4**

Required outputs:

```text
dpo.py created
dpo.sh created
LoRA-based DPO training path works
DPO smoke test saves a checkpoint
```

- [ ] **Step 3: Gate full E2 on pair quality**

Manual inspection rule:

```text
inspect at least 100 pair rows
confirm chosen is usually better than rejected
confirm invalid outputs are handled deterministically
```

Expected: only after this inspection passes should `E2` full training start.

- [ ] **Step 4: Promote DPO to experiment E2**

Run target:

```text
output/research/E2_dpo/final_checkpoint
```

Expected: `E2` final checkpoint exists and is recorded in the experiment tracker.

---

### Task 6: Make GRPO Stable On Top Of SFT And DPO

**Files:**
- Modify: `D:\code\MiniOneRec_v2\rl.py`
- Modify: `D:\code\MiniOneRec_v2\rl.sh`
- Modify: `D:\code\MiniOneRec_v2\config\zero2_opt.yaml`
- Create: `D:\code\MiniOneRec_v2\configs\research\qwen25_3b_grpo.yaml`

**Estimate:**
- 工程时间：`1 天`
- GPU时间：`4 到 12 小时`

- [ ] **Step 1: Execute research plan Task 5**

Required outputs:

```text
GRPO defaults reduced to single-GPU safe values
model_path is configurable for both E3 and E4
research config exists
GRPO smoke test passes from SFT checkpoint
```

- [ ] **Step 2: Freeze the GRPO safe defaults before the full runs**

Use:

```text
train_batch_size = 1
eval_batch_size = 1
gradient_accumulation_steps = 16
num_generations = 4
max_completion_length = 16 or 32
reward_type = ranking
```

Expected: these values are used consistently in E3 and E4 unless instability forces a documented change.

- [ ] **Step 3: Run E3 before E4**

Execution order:

```text
E3 = SFT + GRPO
E4 = SFT + DPO + GRPO
```

Expected: GRPO stability is first proven on the simpler SFT base.

---

### Task 7: Expand Evaluation And Analysis Before Final Comparison

**Files:**
- Create: `D:\code\MiniOneRec_v2\analysis\metrics_plus.py`
- Create: `D:\code\MiniOneRec_v2\analysis\bucket_analysis.py`
- Create: `D:\code\MiniOneRec_v2\analysis\collect_results.py`
- Modify: `D:\code\MiniOneRec_v2\calc.py`
- Modify: `D:\code\MiniOneRec_v2\evaluate.sh`
- Modify: `D:\code\MiniOneRec_v2\convert_dataset.py`
- Modify: `D:\code\MiniOneRec_v2\data.py`

**Estimate:**
- 工程时间：`1 到 1.5 天`
- GPU时间：`1 到 3 小时`

- [ ] **Step 1: Execute research plan Task 6 and Task 7**

Required outputs:

```text
invalid rate added
coverage added
per-sample result export added
item bucket analysis added
history bucket analysis added
```

- [ ] **Step 2: Freeze one unified evaluation protocol**

Use:

```text
batch_size = 1
num_beams = 10
max_new_tokens = 32
length_penalty = 0.0
```

Expected: these exact decode settings are used for E1-E4 initial comparison.

- [ ] **Step 3: Validate the new analysis commands on one smoke result**

Run:

```bash
python analysis/metrics_plus.py --help
python analysis/bucket_analysis.py --help
python analysis/collect_results.py --help
```

Expected: all three CLIs import successfully before any full experiment evaluation begins.

---

### Task 8: Run The Four Main Experiments

**Files:**
- Modify: `D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`

**Estimate:**
- 工程时间：`0.5 到 1 天`
- GPU时间：`2 到 4 天`

- [ ] **Step 1: Run E1**

Target output:

```text
output/research/E1_sft/final_checkpoint
```

Expected runtime:

```text
6 to 12 hours on 1x A800-80GB
```

- [ ] **Step 2: Run E2**

Target output:

```text
output/research/E2_dpo/final_checkpoint
```

Expected runtime:

```text
pair generation: 4 to 8 hours
DPO train: 4 to 10 hours
```

- [ ] **Step 3: Run E3**

Target output:

```text
output/research/E3_grpo/final_checkpoint
```

Expected runtime:

```text
8 to 20 hours
```

- [ ] **Step 4: Run E4**

Target output:

```text
output/research/E4_dpo_grpo/final_checkpoint
```

Expected runtime:

```text
8 to 20 hours
```

- [ ] **Step 5: Update the experiment tracker after every run**

Record for every experiment:

```text
checkpoint path
hardware
runtime
best validation checkpoint
eval output path
notes on OOM or instability
```

Expected: tracker is always current before the next experiment starts.

---

### Task 9: Evaluate, Aggregate, And Compare E1 Through E4

**Files:**
- Modify: `D:\code\MiniOneRec_v2\docs\research\2026-04-18-experiment-matrix.md`
- Create: `D:\code\MiniOneRec_v2\results\research_summary.csv` (generated)

**Estimate:**
- 工程时间：`0.5 天`
- GPU时间：`6 到 18 小时`

- [ ] **Step 1: Run unified evaluation for E1-E4**

Per checkpoint:

```text
evaluate.py
calc.py
analysis/metrics_plus.py
analysis/bucket_analysis.py
```

Expected outputs:

```text
results/E1/final_result.json
results/E1/metrics_plus.json
results/E1/bucket_metrics.json
results/E2/final_result.json
results/E2/metrics_plus.json
results/E2/bucket_metrics.json
results/E3/final_result.json
results/E3/metrics_plus.json
results/E3/bucket_metrics.json
results/E4/final_result.json
results/E4/metrics_plus.json
results/E4/bucket_metrics.json
```

- [ ] **Step 2: Aggregate all result files into one summary table**

Run:

```bash
python analysis/collect_results.py --results_root results --output_csv results/research_summary.csv
```

Expected: `results/research_summary.csv` exists and contains one row per experiment.

- [ ] **Step 3: Freeze the comparison set**

Pass condition:

```text
all four experiments were evaluated with identical decode settings
all summary numbers trace back to concrete result files
```

Expected: no further metric claims are made before this condition is met.

---

### Task 10: Produce The Final Deliverables

**Files:**
- Modify: `D:\code\MiniOneRec_v2\README.md`
- Create: `D:\code\MiniOneRec_v2\docs\research\2026-04-18-final-summary.md`
- Reuse: `D:\code\MiniOneRec_v2\results\research_summary.csv`

**Estimate:**
- 工程时间：`0.5 到 1 天`
- GPU时间：`0`

- [ ] **Step 1: Execute research plan Task 10**

Required outputs:

```text
final summary document created
README research extension section added
claims are tied to actual result files
```

- [ ] **Step 2: Write the final summary around four questions**

Use:

```text
Which stage improved HR/NDCG?
Which stage reduced invalid rate?
Which stage improved tail items?
Which stage improved short-history users?
```

Expected: each answer cites the experiment rows that support it.

- [ ] **Step 3: Package the final handoff**

Final handoff bundle:

```text
env/README_rental_gpu.md
docs/research/2026-04-18-experiment-matrix.md
results/research_summary.csv
docs/research/2026-04-18-final-summary.md
README.md
```

Expected: a reviewer can reproduce the environment, locate the checkpoints, inspect the metrics, and read the conclusion in one pass.

---

## Critical Path Timeline

| Phase | 工程时间 | GPU时间 | 日历时间建议 |
|---|---:|---:|---:|
| Task 1 冻结输入 | 0.5 天 | 0 | 0.5 天 |
| Task 2 环境搭建与锁定 | 0.5 到 1 天 | 2 到 4 小时 | 1 天 |
| Task 3 运行入口收敛 | 0.5 天 | 0.5 小时 | 0.5 天 |
| Task 4 SFT 改造与 E1 | 1 到 1.5 天 | 2 到 6 小时 | 1.5 天 |
| Task 5 DPO 数据与 E2 | 1.5 到 2 天 | 4 到 10 小时 | 2 天 |
| Task 6 GRPO 收敛与 E3/E4 准备 | 1 天 | 4 到 12 小时 | 1 到 1.5 天 |
| Task 7 评估扩展 | 1 到 1.5 天 | 1 到 3 小时 | 1 到 1.5 天 |
| Task 8 四组主实验 | 0.5 到 1 天 | 2 到 4 天 | 2 到 4 天 |
| Task 9 统一评估汇总 | 0.5 天 | 6 到 18 小时 | 1 天 |
| Task 10 最终交付 | 0.5 到 1 天 | 0 | 0.5 到 1 天 |

## Total Time Estimate

- 最乐观：
  - 工程时间：`6.5 天`
  - GPU时间：`约 3 天`
  - 日历时间：`7 到 9 天`
- 中位预估：
  - 工程时间：`8 到 9 天`
  - GPU时间：`约 4 天`
  - 日历时间：`10 到 12 天`
- 含排障缓冲：
  - 工程时间：`10 到 12 天`
  - GPU时间：`4 到 6 天`
  - 日历时间：`12 到 15 天`

## Biggest Schedule Risks

- `DPO pair` 质量不够，导致 E2 训练前要返工 pair 构造
- `GRPO` 在 DPO checkpoint 上不稳定，E4 可能需要多轮调参
- 单卡评估速度慢，`E1-E4` 统一评估可能比预期久
- 租卡平台镜像或驱动差异导致环境重建失败

## Execution Order

1. 完成 Task 1 和 Task 2，先把环境锁死
2. 完成 Task 3，清掉 8 卡脚本假设
3. 完成 Task 4，先拿到稳定的 E1
4. 完成 Task 5，拿到稳定的 E2
5. 完成 Task 6，按 `E3 -> E4` 顺序推进
6. 完成 Task 7，补齐统一评估能力
7. 完成 Task 8 和 Task 9，产出四组结果
8. 完成 Task 10，输出总结和 README

## Success Criteria

- A800 节点环境可重建且被锁定
- E1-E4 全部产出 final checkpoint
- E1-E4 全部产出统一评估结果
- 汇总表包含：
  - `HR@10`
  - `NDCG@10`
  - `invalid_rate`
  - `coverage`
  - `head/mid/tail`
  - `short/medium/long`
- 最终文档能解释：
  - 项目如何搭建
  - 实验如何运行
  - 结果如何比较
  - 结论和风险分别是什么
