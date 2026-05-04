import argparse
import importlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


REQUIRED_MODULES = [
    "accelerate",
    "bitsandbytes",
    "datasets",
    "deepspeed",
    "fire",
    "huggingface_hub",
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

SOFT_FAIL_MODULES_WITHOUT_CUDA = {"deepspeed"}

PROJECT_INPUTS = [
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/converted/rqvae/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/converted/rqvae/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/converted/rqvae/test/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/converted/rqvae/info/Industrial_and_Scientific_5_2016-10-2018-11.txt",
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/sid_variants/rqvae/Industrial_and_Scientific.index.json",
    "MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411/shared/Industrial_and_Scientific.item.json",
]

MODEL_REQUIRED_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def run_python_snippet(snippet: str, retries: int = 1) -> str:
    last_error = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()

        stderr = result.stderr.strip().replace("\n", " | ")
        last_error = stderr or f"python snippet exited with code {result.returncode}"
        if attempt < retries:
            time.sleep(2)

    raise RuntimeError(last_error)


def version_in_subprocess(module_name: str) -> str:
    return run_python_snippet(
        "import importlib; "
        f"module = importlib.import_module('{module_name}'); "
        "print(getattr(module, '__version__', 'unknown'))",
        retries=3,
    )


def torch_info() -> dict[str, str]:
    output = run_python_snippet(
        "import torch\n"
        "print(f'torch_version={torch.__version__}')\n"
        "print(f'torch_cuda={torch.version.cuda}')\n"
        "print(f'cuda_available={torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'gpu_name={torch.cuda.get_device_name(0)}')\n"
        "    print(f'bf16_supported={torch.cuda.is_bf16_supported()}')\n",
        retries=3,
    )
    info = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key] = value
    return info


def print_nvidia_smi() -> None:
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
    except Exception as exc:
        print(f"nvidia_smi_unavailable={exc}")
        return

    print("nvidia_smi=" + result.stdout.strip().replace("\n", " | "))


def validate_project_inputs(repo_root: Path) -> bool:
    missing = [path for path in PROJECT_INPUTS if not (repo_root / path).exists()]
    if missing:
        print("project_inputs_ok=False")
        for path in missing:
            print(f"missing_project_input={path}")
        return False

    print("project_inputs_ok=True")
    return True


def validate_model_dir(model_dir: Path) -> bool:
    if not model_dir.exists():
        print(f"model_dir_exists=False path={model_dir}")
        return False

    missing = [name for name in MODEL_REQUIRED_FILES if not (model_dir / name).exists()]
    weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if missing or not weight_files:
        print(f"model_dir_exists=True path={model_dir}")
        for name in missing:
            print(f"missing_model_file={name}")
        if not weight_files:
            print("missing_model_weights=True")
        return False

    print(f"model_dir_ok=True path={model_dir}")
    print(f"model_weight_files={len(weight_files)}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MiniOneRec research runtime.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail when CUDA is unavailable.")
    parser.add_argument("--require-bf16", action="store_true", help="Fail when the visible GPU lacks BF16 support.")
    parser.add_argument("--repo-root", default="/root/autodl-tmp/MiniOneRec_v2")
    parser.add_argument("--model-dir", default="/root/models/Qwen2.5-3B")
    parser.add_argument("--skip-model", action="store_true", help="Skip local model file validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch_runtime = torch_info()
    print(f"python_version={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"torch_version={torch_runtime['torch_version']}")
    print(f"torch_cuda={torch_runtime['torch_cuda']}")
    print(f"cuda_available={torch_runtime['cuda_available']}")
    print(f"HF_HOME={os.environ.get('HF_HOME', '')}")
    print(f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', '')}")
    print(f"PIP_CACHE_DIR={os.environ.get('PIP_CACHE_DIR', '')}")

    errors = []
    if platform.python_version_tuple()[:2] != ("3", "12"):
        errors.append("Python is not 3.12")
    if not torch_runtime["torch_version"].startswith("2.5.1"):
        errors.append("torch is not 2.5.1")
    if torch_runtime["torch_cuda"] != "12.4":
        errors.append("torch CUDA runtime is not 12.4")

    missing_modules = []
    soft_failed_modules = []
    for module_name in REQUIRED_MODULES:
        try:
            print(f"{module_name}={version_in_subprocess(module_name)}")
        except Exception as exc:
            print(f"IMPORT_ERROR {module_name}: {exc}")
            if module_name in SOFT_FAIL_MODULES_WITHOUT_CUDA and torch_runtime["cuda_available"] != "True":
                soft_failed_modules.append(module_name)
            else:
                missing_modules.append(module_name)
    if missing_modules:
        errors.append(f"missing modules: {missing_modules}")
    if soft_failed_modules:
        print(f"soft_failed_modules={soft_failed_modules}")

    if torch_runtime["cuda_available"] == "True":
        print(f"gpu_name={torch_runtime['gpu_name']}")
        print(f"bf16_supported={torch_runtime['bf16_supported']}")
        print_nvidia_smi()
    else:
        print_nvidia_smi()
        if args.require_cuda:
            errors.append("CUDA device is not visible to torch")

    if args.require_bf16 and not (
        torch_runtime["cuda_available"] == "True" and torch_runtime.get("bf16_supported") == "True"
    ):
        errors.append("BF16 is not supported by the visible GPU")

    if not validate_project_inputs(Path(args.repo_root)):
        errors.append("project inputs are missing")
    if not args.skip_model and not validate_model_dir(Path(args.model_dir)):
        errors.append("local model files are missing")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("RUNTIME_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
