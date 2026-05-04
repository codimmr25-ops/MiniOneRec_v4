# MiniOneRec Rental GPU Environment

## Canonical Stack

- Repo root: `/root/autodl-tmp/MiniOneRec_v2`
- Conda env: `/root/miniconda3/envs/minionerec312`
- Model path: `/root/models/Qwen2.5-3B`
- HF cache: `/root/hf_cache`
- Pip cache: `/root/.cache/pip`
- Python: `3.12`
- PyTorch: `2.5.1`
- CUDA runtime: `12.4`

## Create The Environment

```bash
cd /root/autodl-tmp/MiniOneRec_v2
conda env create -f env/mini_one_rec_py312_torch251_cu124.from-history.yml
conda activate minionerec312
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
DS_BUILD_OPS=0 python -m pip install -r env/mini_one_rec_py312_torch251_cu124.pip.txt
```

Do not install the repository-level `requirements.txt`; it pins `torch==2.6.0` and `cu118` packages.

## Download The Base Model

```bash
export HF_HOME=/root/hf_cache
export HF_HUB_CACHE=/root/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-3B --local-dir /root/models/Qwen2.5-3B --local-dir-use-symlinks False
```

## No-GPU Validation

```bash
cd /root/autodl-tmp/MiniOneRec_v2
conda activate minionerec312
export HF_HOME=/root/hf_cache
export HF_HUB_CACHE=/root/hf_cache/hub
export PIP_CACHE_DIR=/root/.cache/pip
python env/validate_runtime.py
python sft.py --help
python rl.py --help
python evaluate.py --help
```

`cuda_available=False` is acceptable during no-GPU preparation.
If `deepspeed` emits a CPU-only warning during this stage, treat it as non-blocking and re-check it on the A800 node.

## A800 Readiness Gate

```bash
cd /root/autodl-tmp/MiniOneRec_v2
conda activate minionerec312
python env/validate_runtime.py --require-cuda --require-bf16
bash sft.sh smoke
```

Do not start a full experiment until the smoke run saves a checkpoint.
