import importlib.util
import json
import os
import argparse


def parse_target_modules(target_modules: str):
    return [module.strip() for module in str(target_modules).split(",") if module.strip()]


def _missing_runtime_dependencies():
    missing = []
    for module_name in ("transformers", "trl", "peft", "datasets"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def normalize_completion(value):
    text = str(value).strip()
    if not text.endswith("\n"):
        text += "\n"
    return text


def load_dpo_hf_dataset(jsonl_path: str):
    from datasets import Dataset

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            rows.append(
                {
                    "prompt": sample["prompt"],
                    "chosen": normalize_completion(sample["chosen"]),
                    "rejected": normalize_completion(sample["rejected"]),
                }
            )
    return Dataset.from_list(rows)


def _is_peft_adapter_dir(model_path: str) -> bool:
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))


def load_model_and_tokenizer_for_dpo(model_path: str, torch_dtype):
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not _is_peft_adapter_dir(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer, False

    peft_config = PeftConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    current_vocab_size = model.get_input_embeddings().num_embeddings
    target_vocab_size = len(tokenizer)
    if current_vocab_size != target_vocab_size:
        try:
            model.resize_token_embeddings(target_vocab_size, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(target_vocab_size)
    model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
    model.config._name_or_path = model_path
    return model, tokenizer, True


def train(
    model_path: str = "",
    train_jsonl: str = "",
    eval_jsonl: str = "",
    output_dir: str = "",
    learning_rate: float = 1e-5,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    bf16: bool = True,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    max_prompt_length: int = 512,
    max_length: int = 544,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
    logging_steps: int = 10,
    eval_steps: int = 100,
    save_steps: int = 100,
    resume_from_checkpoint: str = None,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    missing = _missing_runtime_dependencies()
    if missing:
        raise ImportError(
            "dpo.py requires the following runtime dependencies in the active environment: "
            + ", ".join(missing)
        )

    import torch
    from peft import LoraConfig, TaskType
    from trl import DPOConfig, DPOTrainer

    if not model_path:
        raise ValueError("Please specify --model_path")
    if not train_jsonl:
        raise ValueError("Please specify --train_jsonl")
    if not eval_jsonl:
        raise ValueError("Please specify --eval_jsonl")
    if not output_dir:
        raise ValueError("Please specify --output_dir")

    print("DPO training configuration:")
    print(f"  model_path: {model_path}")
    print(f"  train_jsonl: {train_jsonl}")
    print(f"  eval_jsonl: {eval_jsonl}")
    print(f"  beta: {beta}")
    print(f"  loss_type: {loss_type}")
    print(f"  label_smoothing: {label_smoothing}")
    print(f"  max_prompt_length: {max_prompt_length}")
    print(f"  max_length: {max_length}")
    print(f"  use_lora: {use_lora}")

    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_MODE"] = "offline" if wandb_project else "disabled"

    train_dataset = load_dpo_hf_dataset(train_jsonl)
    eval_dataset = load_dpo_hf_dataset(eval_jsonl)

    model, tokenizer, loaded_peft_adapter = load_model_and_tokenizer_for_dpo(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.config.use_cache = False

    peft_config = None
    if loaded_peft_adapter:
        print("Loaded existing PEFT adapter from model_path; continuing DPO on the SFT LoRA adapter.")
    elif use_lora:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=parse_target_modules(target_modules),
        )

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        bf16=bf16,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        beta=beta,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to=None,
        run_name=wandb_run_name,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    final_checkpoint = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)
    return final_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO training for MiniOneRec.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--bf16", type=lambda value: str(value).lower() == "true", default=True)
    parser.add_argument("--use_lora", type=lambda value: str(value).lower() == "true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=544)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", default="sigmoid")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_run_name", default="")
    args = parser.parse_args()
    train(**vars(args))
