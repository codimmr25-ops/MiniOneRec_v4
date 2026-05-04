# MiniOneRec Rental GPU Environment Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `Ubuntu 22.04 / Python 3.12 / PyTorch 2.5.1 / CUDA 12.4` 这一固定软件栈下，为 MiniOneRec 当前研究路线建立一套可复现、可迁移、适合租卡平台的环境配置流程，支持先在低配 GPU 节点完成环境搭建与烟雾验证，再无偏差迁移到 `A800-80GB` 节点进入后续 `SFT / DPO / GRPO / Eval` 工作。

**Architecture:** 不直接安装仓库根目录的 [requirements.txt](/D:/code/MiniOneRec_v2/requirements.txt)，因为它目前把 `torch` 锁到 `2.6.0`，同时还固定了 `torchrec==0.6.0+cu118`、`fbgemm_gpu==0.8.0+cu118` 和 `--extra-index-url https://download.pytorch.org/whl/cu118`，这与目标栈 `torch 2.5.1 + cu124` 冲突。环境分为 6 段实施：系统依赖、Conda 环境、PyTorch/CUDA、研究任务依赖、低配卡验证、A800 恢复验证；只覆盖当前已有 transfer bundle 的 `rqvae -> SFT / RL / Eval` 主路径和后续计划中的 `DPO` 扩展，不把完整 SID 构建链路的可选依赖混入基础环境。

**Tech Stack:** Ubuntu 22.04, Miniforge, Python 3.12, PyTorch 2.5.1, CUDA 12.4, Transformers 4.57.1, TRL 0.24.0, Accelerate 1.10.1, DeepSpeed 0.18.0, bitsandbytes 0.48.1, pandas, numpy

---

## Scope And Guardrails

- 目标环境固定为：
  - `Ubuntu 22.04`
  - `Python 3.12`
  - `PyTorch 2.5.1`
  - `CUDA 12.4`
- 当前计划仅覆盖你后续明确要做的任务：
  - 基于 `MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411` 的 `rqvae` 路径
  - `sft.py`
  - `rl.py`
  - `evaluate.py`
  - 之后会补进来的 `dpo.py / build_dpo_pairs.py / analysis/*`
- 当前计划明确**不**覆盖的内容：
  - 全量 SID 重建
  - `rqkmeans_plus / constrained_rqkmeans` 的直接训练
  - `faiss-gpu / k_means_constrained / polars / openai` 这些只在完整 SID 构建或文本增强场景才需要的额外依赖
- 低配 GPU 节点的职责是：
  - 完成环境安装
  - 跑 import / CLI / 单元测试 / 路径校验
  - 导出可迁移的环境清单
- 低配 GPU 节点**不是**正式训练节点：
  - 当前 [sft.py](/D:/code/MiniOneRec_v2/sft.py) 与 [rl.py](/D:/code/MiniOneRec_v2/rl.py) 都默认使用 `bf16`
  - 如果低配 GPU 不支持 BF16，只做环境验证，不做训练 smoke
- 当前 [sft.sh](/D:/code/MiniOneRec_v2/sft.sh)、[rl.sh](/D:/code/MiniOneRec_v2/rl.sh)、[evaluate.sh](/D:/code/MiniOneRec_v2/evaluate.sh) 都按 8 卡写死；环境阶段不要直接运行这些脚本

## File Structure

### Files To Create

- `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.from-history.yml`
  - Conda 最小环境清单，只负责创建 Python 3.12 基础环境
- `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.pip.txt`
  - 与当前研究路线匹配的 pip 依赖清单，显式避开 `torch==2.6.0` 和 `cu118` 绑定
- `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.lock.txt`
  - 在低配节点完成安装后导出的精确锁文件，用于 A800 节点恢复
- `D:\code\MiniOneRec_v2\env\validate_runtime.py`
  - 统一验证 Python、Torch、CUDA、关键依赖、GPU、BF16 能力
- `D:\code\MiniOneRec_v2\env\README_rental_gpu.md`
  - 记录低配节点与 A800 节点的验证结果、驱动版本、恢复方式

### Generated Directories

- `D:\code\MiniOneRec_v2\env\wheelhouse\`
  - 低配节点预下载的 pip wheel 缓存，不纳入 git

### Existing Files To Reuse

- `D:\code\MiniOneRec_v2\docs\superpowers\plans\2026-04-18-sft-dpo-grpo-research-plan.md`
  - 后续研究主计划
- `D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\runbook\README_transfer.md`
  - transfer bundle 的兼容性与路径说明
- `D:\code\MiniOneRec_v2\data_test.py`
  - 当前仓库最适合作为环境级 smoke test 的现有测试文件

---

### Task 1: Create Canonical Environment Manifests In Repo

**Files:**
- Create: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.from-history.yml`
- Create: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.pip.txt`
- Create: `D:\code\MiniOneRec_v2\env\validate_runtime.py`
- Create: `D:\code\MiniOneRec_v2\env\README_rental_gpu.md`

- [ ] **Step 1: Create the `env` directory**

Run:

```bash
cd "$HOME/MiniOneRec"
mkdir -p env
```

Expected: `env/` exists in repo root.

- [ ] **Step 2: Write the minimal Conda environment file**

Write:

```yaml
name: minionerec312
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
```

Save to:

```text
env/mini_one_rec_py312_torch251_cu124.from-history.yml
```

- [ ] **Step 3: Write the pip package manifest for the research route**

Write:

```text
accelerate==1.10.1
aiofiles==24.1.0
aiohttp==3.10.3
bitsandbytes==0.48.1
datasets==4.2.0
deepspeed==0.18.0
einops==0.8.0
fire==0.7.1
huggingface-hub==0.35.3
numpy==1.26.3
nvitop==1.5.3
pandas==2.2.2
peft>=0.14,<0.18
psutil==6.0.0
PyYAML==6.0.1
requests==2.32.3
safetensors==0.6.2
scikit-learn==1.5.2
scipy==1.14.0
tokenizers==0.22.1
transformers==4.57.1
trl==0.24.0
wandb==0.22.2
```

Save to:

```text
env/mini_one_rec_py312_torch251_cu124.pip.txt
```

Notes:

- Do **not** put `torch`, `torchaudio`, or `torchvision` into this file; those are installed separately from the official `cu124` index.
- Do **not** copy `torchrec==0.6.0+cu118`, `fbgemm_gpu==0.8.0+cu118`, or the repo’s `cu118` extra index into this file.

- [ ] **Step 4: Write the runtime validation script**

Write:

```python
import importlib
import os
import platform
import subprocess
import sys

import torch

REQUIRED_MODULES = [
    "accelerate",
    "bitsandbytes",
    "datasets",
    "deepspeed",
    "fire",
    "numpy",
    "pandas",
    "peft",
    "requests",
    "safetensors",
    "sklearn",
    "tokenizers",
    "transformers",
    "trl",
    "wandb",
]


def version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def main() -> int:
    print(f"python_version={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")

    if platform.python_version_tuple()[:2] != ("3", "12"):
        print("ERROR: Python is not 3.12")
        return 2

    if not torch.__version__.startswith("2.5.1"):
        print("ERROR: torch is not 2.5.1")
        return 3

    if torch.version.cuda != "12.4":
        print("ERROR: torch CUDA runtime is not 12.4")
        return 4

    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            print(f"{module_name}={version_of(module_name)}")
        except Exception as exc:
            print(f"IMPORT_ERROR {module_name}: {exc}")
            missing.append(module_name)

    if missing:
        print(f"ERROR: missing modules: {missing}")
        return 5

    if not torch.cuda.is_available():
        print("ERROR: CUDA device is not visible to torch")
        return 6

    device_name = torch.cuda.get_device_name(0)
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"gpu_name={device_name}")
    print(f"bf16_supported={bf16_supported}")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        print("nvidia_smi=" + result.stdout.strip())
    except Exception as exc:
        print(f"WARNING: nvidia-smi query failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Save to:

```text
env/validate_runtime.py
```

- [ ] **Step 5: Write the environment readme template**

Write:

```markdown
# Rental GPU Environment Notes

## Canonical Stack
- Ubuntu 22.04
- Python 3.12
- PyTorch 2.5.1
- CUDA 12.4

## Low-Config Node Validation
- `python env/validate_runtime.py`
- `python sft.py --help`
- `python rl.py --help`
- `python evaluate.py --help`
- `python -m unittest data_test.py -v`

## A800 Node Validation
- `python env/validate_runtime.py`
- confirm `gpu_name` contains `A800`
- confirm `bf16_supported=True`

## Notes
- Do not run `pip install -r requirements.txt`
- Do not run `sft.sh`, `rl.sh`, or `evaluate.sh` before they are refactored for single-GPU research mode
```

Save to:

```text
env/README_rental_gpu.md
```

- [ ] **Step 6: Commit the environment manifests**

Run:

```bash
git add env/mini_one_rec_py312_torch251_cu124.from-history.yml env/mini_one_rec_py312_torch251_cu124.pip.txt env/validate_runtime.py env/README_rental_gpu.md
git commit -m "chore: add canonical rental GPU environment manifests"
```

Expected: one commit containing only environment manifests and validator files.

---

### Task 2: Provision The Low-Config Rental GPU Node

**Files:**
- Reuse: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.from-history.yml`
- Reuse: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.pip.txt`

- [ ] **Step 1: Check OS, GPU, driver, and disk before installing anything**

Run:

```bash
lsb_release -a
uname -r
nvidia-smi
df -h
```

Expected:

- Ubuntu release is `22.04`
- `nvidia-smi` succeeds
- at least `40G` free disk remains for env + caches + model downloads

- [ ] **Step 2: Install system packages required by pip builds and DeepSpeed**

Run:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs curl wget ca-certificates pciutils libaio-dev ninja-build pkg-config numactl
git lfs install
```

Expected: `git`, `gcc`, `g++`, `curl`, `wget`, and `ninja` are available in `PATH`.

- [ ] **Step 3: Install Miniforge in the current user account**

Run:

```bash
cd "$HOME"
wget -O Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda init bash
```

Expected:

- `conda --version` works after shell reload
- installation path is `$HOME/miniforge3`

- [ ] **Step 4: Reopen the shell and confirm conda is active-capable**

Run:

```bash
source "$HOME/.bashrc"
conda --version
conda info --base
```

Expected:

- `conda --version` prints a version
- `conda info --base` prints `$HOME/miniforge3`

---

### Task 3: Create The Python 3.12 / Torch 2.5.1 / CUDA 12.4 Environment

**Files:**
- Reuse: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.from-history.yml`
- Reuse: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.pip.txt`

- [ ] **Step 1: Create the Conda environment**

Run:

```bash
cd "$HOME/MiniOneRec"
source "$HOME/.bashrc"
conda env create -f env/mini_one_rec_py312_torch251_cu124.from-history.yml
conda activate minionerec312
```

Expected: shell prompt shows `(minionerec312)`.

- [ ] **Step 2: Upgrade pip tooling inside the environment**

Run:

```bash
python -m pip install --upgrade pip setuptools wheel
python -V
pip -V
```

Expected:

- Python version starts with `Python 3.12`
- `pip` is bound to the `minionerec312` environment

- [ ] **Step 3: Install the official PyTorch 2.5.1 CUDA 12.4 build**

Run:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Expected:

- `torch`, `torchvision`, `torchaudio` install without falling back to CPU wheels
- no `cu118` packages are pulled

- [ ] **Step 4: Install the MiniOneRec research-route dependencies**

Run:

```bash
DS_BUILD_OPS=0 python -m pip install -r env/mini_one_rec_py312_torch251_cu124.pip.txt
```

Expected:

- `deepspeed`, `transformers`, `trl`, `bitsandbytes`, and `peft` install successfully
- the install does not attempt to replace `torch==2.5.1`

- [ ] **Step 5: Verify the final package versions immediately**

Run:

```bash
python - <<'PY'
import torch
import transformers
import trl
import accelerate
print("python_ok", True)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("accelerate", accelerate.__version__)
PY
```

Expected:

- `torch 2.5.1`
- `torch_cuda 12.4`
- `transformers 4.57.1`
- `trl 0.24.0`
- `accelerate 1.10.1`

---

### Task 4: Run Low-Config Node Validation Without Starting Real Training

**Files:**
- Reuse: `D:\code\MiniOneRec_v2\env\validate_runtime.py`
- Reuse: `D:\code\MiniOneRec_v2\data_test.py`

- [ ] **Step 1: Run the runtime validator**

Run:

```bash
cd "$HOME/MiniOneRec"
conda activate minionerec312
python env/validate_runtime.py
```

Expected:

- exit code `0`
- `torch_version=2.5.1...`
- `torch_cuda=12.4`
- `cuda_available=True`

- [ ] **Step 2: Check whether this low-config GPU supports BF16**

Run:

```bash
python - <<'PY'
import torch
print("bf16_supported", torch.cuda.is_bf16_supported())
PY
```

Expected:

- If output is `bf16_supported True`, this node can also run very small training smoke tests later
- If output is `bf16_supported False`, stop at environment validation on this node and defer all training smoke to the A800 node

- [ ] **Step 3: Validate the repo CLI entrypoints**

Run:

```bash
python sft.py --help
python rl.py --help
python evaluate.py --help
```

Expected:

- all three commands print help and exit successfully
- no import error from `bitsandbytes`, `trl`, `deepspeed`, or `transformers`

- [ ] **Step 4: Run the existing unit tests that do not require a real model**

Run:

```bash
python -m unittest data_test.py -v
```

Expected: the test runner ends with `OK`.

- [ ] **Step 5: Validate the transfer bundle integrity**

Run:

```bash
cd "$HOME/MiniOneRec/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411"
sha256sum -c report/SHA256SUMS
```

Expected: every line ends with `OK`.

- [ ] **Step 6: Record the low-config validation result**

Append the actual machine information to:

```text
env/README_rental_gpu.md
```

Required fields to record:

```text
date
GPU name
driver version
torch version
torch CUDA runtime
bf16_supported
validate_runtime.py result
data_test.py result
bundle SHA256 result
```

---

### Task 5: Export A Reusable Lockfile And Wheel Cache From The Low-Config Node

**Files:**
- Create: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.lock.txt`
- Create: `D:\code\MiniOneRec_v2\env\wheelhouse\` (generated)

- [ ] **Step 1: Export the exact pip lockfile**

Run:

```bash
cd "$HOME/MiniOneRec"
conda activate minionerec312
python -m pip freeze > env/mini_one_rec_py312_torch251_cu124.lock.txt
```

Expected: lockfile exists and contains exact resolved versions from the validated node.

- [ ] **Step 2: Re-export the conda file from history**

Run:

```bash
conda env export --from-history -n minionerec312 > env/mini_one_rec_py312_torch251_cu124.from-history.yml
```

Expected: the history file still only contains the minimal Conda seed packages.

- [ ] **Step 3: Pre-download pip wheels for faster restore**

Run:

```bash
mkdir -p env/wheelhouse
python -m pip download -r env/mini_one_rec_py312_torch251_cu124.pip.txt -d env/wheelhouse
```

Expected: `env/wheelhouse/` contains wheels for the research-route pip dependencies.

- [ ] **Step 4: Optionally pre-download the base model used in the research plan**

Run:

```bash
mkdir -p "$HOME/.cache/huggingface"
huggingface-cli download Qwen/Qwen2.5-3B --local-dir "$HOME/.cache/huggingface/Qwen2.5-3B"
```

Expected: the base model is cached locally for later copy or reuse.

- [ ] **Step 5: Commit the tracked environment artifacts**

Run:

```bash
git add env/mini_one_rec_py312_torch251_cu124.from-history.yml env/mini_one_rec_py312_torch251_cu124.lock.txt env/README_rental_gpu.md
git commit -m "chore: capture validated rental GPU environment lockfiles"
```

Expected: only tracked manifests and notes are committed; `env/wheelhouse/` stays untracked.

---

### Task 6: Restore The Same Environment On The A800-80GB Node

**Files:**
- Reuse: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.from-history.yml`
- Reuse: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.pip.txt`
- Reuse: `D:\code\MiniOneRec_v2\env\mini_one_rec_py312_torch251_cu124.lock.txt`
- Reuse: `D:\code\MiniOneRec_v2\env\validate_runtime.py`

- [ ] **Step 1: Provision the A800 node with the same system packages**

Run:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs curl wget ca-certificates pciutils libaio-dev ninja-build pkg-config numactl
git lfs install
```

Expected: the A800 node has the same system toolchain as the low-config node.

- [ ] **Step 2: Recreate the Conda environment from the repo manifest**

Run:

```bash
cd "$HOME/MiniOneRec"
source "$HOME/.bashrc"
conda env create -f env/mini_one_rec_py312_torch251_cu124.from-history.yml
conda activate minionerec312
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
DS_BUILD_OPS=0 python -m pip install -r env/mini_one_rec_py312_torch251_cu124.pip.txt
```

Expected: the A800 node reproduces the same validated environment.

- [ ] **Step 3: If `env/wheelhouse` was copied, install from local wheels first**

Run:

```bash
python -m pip install --no-index --find-links env/wheelhouse -r env/mini_one_rec_py312_torch251_cu124.pip.txt
```

Expected: installs succeed without re-downloading pip packages from the network.

- [ ] **Step 4: Run the runtime validator again on A800**

Run:

```bash
python env/validate_runtime.py
```

Expected:

- exit code `0`
- `gpu_name` contains `A800`
- `bf16_supported=True`

- [ ] **Step 5: Re-run the repo smoke checks on A800**

Run:

```bash
python sft.py --help
python rl.py --help
python evaluate.py --help
python -m unittest data_test.py -v
```

Expected: all four commands succeed exactly as on the low-config node.

- [ ] **Step 6: Re-check the transfer bundle after migration**

Run:

```bash
cd "$HOME/MiniOneRec/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411"
sha256sum -c report/SHA256SUMS
```

Expected: every line ends with `OK`.

---

### Task 7: Mark The Environment Ready For The Next Research Tasks

**Files:**
- Reuse: `D:\code\MiniOneRec_v2\docs\superpowers\plans\2026-04-18-sft-dpo-grpo-research-plan.md`
- Reuse: `D:\code\MiniOneRec_v2\MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411\runbook\README_transfer.md`

- [ ] **Step 1: Export the canonical environment variables for the research route**

Run:

```bash
cd "$HOME/MiniOneRec"
export REPO_ROOT="$PWD"
export BUNDLE_ROOT="$REPO_ROOT/MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411"
export BASE_MODEL="Qwen/Qwen2.5-3B"
export OUTPUT_ROOT="$REPO_ROOT/output/research"
mkdir -p "$OUTPUT_ROOT"
```

Expected: all four variables are set for the current shell session.

- [ ] **Step 2: Verify that the plan-critical files exist before any code change**

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
print("ALL_RESEARCH_PATHS_OK")
PY
```

Expected: `ALL_RESEARCH_PATHS_OK`.

- [ ] **Step 3: Record the exact readiness gate**

Append this checklist result to:

```text
env/README_rental_gpu.md
```

Checklist:

```text
[ ] Ubuntu 22.04 confirmed
[ ] Python 3.12 confirmed
[ ] torch 2.5.1 confirmed
[ ] torch CUDA runtime 12.4 confirmed
[ ] A800 visible
[ ] bf16_supported=True
[ ] sft.py --help works
[ ] rl.py --help works
[ ] evaluate.py --help works
[ ] data_test.py passes
[ ] transfer bundle SHA256 passes
[ ] rqvae paths exist
```

- [ ] **Step 4: Commit the readiness notes**

Run:

```bash
git add env/README_rental_gpu.md
git commit -m "docs: mark rental GPU environment ready for research execution"
```

Expected: repo now contains the exact environment state that future SFT/DPO/GRPO work should start from.

---

## Risks And Controls

- **Risk:** 低配 GPU 节点不支持 BF16，导致训练脚本即使依赖齐全也无法安全 smoke  
  **Control:** 环境阶段只要求 `validate_runtime.py`、CLI、单元测试、bundle 校验通过；真正训练 smoke 推迟到 `A800-80GB`

- **Risk:** 工程师习惯性执行 `pip install -r requirements.txt`，把环境拉回 `torch 2.6.0 + cu118`  
  **Control:** 所有环境文档、readme、validator 都明确禁止直接安装根目录 requirements

- **Risk:** `deepspeed` 在租卡节点编译自定义算子失败  
  **Control:** 默认使用 `DS_BUILD_OPS=0 pip install deepspeed==0.18.0`，先保证训练框架可导入、可被 accelerate 调起

- **Risk:** 跨节点恢复时包版本漂移  
  **Control:** 同时保留 `from-history.yml`、`pip.txt` 和 `lock.txt`，并尽量预下载 `wheelhouse`

- **Risk:** 后续研究任务直接调用 8 卡脚本，导致单卡节点误启动失败  
  **Control:** 在环境准备阶段只允许运行 `python sft.py --help`、`python rl.py --help`、`python evaluate.py --help`

## Success Criteria

- 低配节点可以在不触发真实训练的前提下完成：
  - 环境安装
  - import 校验
  - CLI 校验
  - `data_test.py` 校验
  - transfer bundle SHA256 校验
- A800 节点可以重建相同环境，并满足：
  - `torch==2.5.1`
  - `torch.version.cuda == "12.4"`
  - `bf16_supported=True`
  - `gpu_name` 为 `A800`
- 研究主计划开始前，仓库中已经存在：
  - `env/mini_one_rec_py312_torch251_cu124.from-history.yml`
  - `env/mini_one_rec_py312_torch251_cu124.pip.txt`
  - `env/mini_one_rec_py312_torch251_cu124.lock.txt`
  - `env/validate_runtime.py`
  - `env/README_rental_gpu.md`
- 下一阶段工作可以直接从：
  - [2026-04-18-sft-dpo-grpo-research-plan.md](/D:/code/MiniOneRec_v2/docs/superpowers/plans/2026-04-18-sft-dpo-grpo-research-plan.md)
  - 的 `Task 1` 开始执行，而无需再返工环境
