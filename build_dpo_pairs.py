import argparse
import ast
import importlib.util
import json
import os
import random
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


def normalize_sid(value) -> str:
    return str(value).strip().strip('"').strip()


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def format_sid_completion(sid: str, append_completion_newline: bool = True) -> str:
    completion = normalize_sid(sid)
    if append_completion_newline and completion and not completion.endswith("\n"):
        completion += "\n"
    return completion


def _safe_list(value) -> List[object]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return [text]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def make_dpo_prompt(
    history_item_sids: Sequence[str],
    user_preference: str = "",
    use_user_preference: bool = False,
) -> str:
    history = ", ".join(normalize_sid(sid) for sid in history_item_sids if normalize_sid(sid))
    preference = str(user_preference or "").strip()
    if use_user_preference and preference:
        input_text = (
            f"The user has interacted with items {history} in chronological order. "
            f"The user's inferred preference is: {preference} "
            "Can you predict the next possible item that the user may expect?"
        )
    else:
        input_text = (
            f"The user has interacted with items {history} in chronological order. "
            "Can you predict the next possible item that the user may expect?"
        )
    return f"""### User Input: 
{input_text}

### Response:\n"""


def load_jsonl(path: str) -> List[Dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, object]]):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_fallback_sids(info_file: str) -> List[str]:
    with open(info_file, "r", encoding="utf-8") as fh:
        return [line.split("\t")[0].strip() for line in fh if line.strip()]


def _is_peft_adapter_dir(model_path: str) -> bool:
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))


def _batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def select_rejected_sids(
    candidates: Sequence[str],
    target_sid: str,
    fallback_sids: Sequence[str],
    history_item_sids: Sequence[str] = (),
    num_negatives: int = 1,
    fallback_strategy: str = "random",
    filter_history_items: bool = True,
    rng: random.Random = None,
) -> Tuple[List[Tuple[str, str, int]], str]:
    rng = rng or random.Random()
    normalized_target = normalize_sid(target_sid)
    blocked = {normalized_target}
    if filter_history_items:
        blocked.update(normalize_sid(sid) for sid in history_item_sids if normalize_sid(sid))

    selected = []
    seen = set()
    for rank, candidate in enumerate(candidates, start=1):
        normalized = normalize_sid(candidate)
        if normalized and normalized not in blocked and normalized not in seen:
            selected.append((normalized, "model_hard_negative", rank))
            seen.add(normalized)
            if len(selected) >= num_negatives:
                return selected, ""

    if fallback_strategy != "random":
        raise ValueError(f"Unsupported fallback_strategy: {fallback_strategy}")

    fallback_pool = []
    for candidate in fallback_sids:
        normalized = normalize_sid(candidate)
        if normalized and normalized not in blocked and normalized not in seen:
            fallback_pool.append(normalized)

    rng.shuffle(fallback_pool)
    for candidate in fallback_pool:
        selected.append((candidate, "random_fallback", 0))
        seen.add(candidate)
        if len(selected) >= num_negatives:
            return selected, ""

    if selected:
        return selected, "partial_negative"
    return [], "missing_negative"


def build_dpo_records(
    preference_records: Sequence[Dict[str, object]],
    candidate_generator: Callable[[List[str]], List[List[str]]],
    fallback_sids: Sequence[str],
    use_user_preference: bool = False,
    generation_batch_size: int = 16,
    num_negatives_per_positive: int = 1,
    fallback_strategy: str = "random",
    filter_history_items: bool = True,
    append_completion_newline: bool = True,
    seed: int = 2026,
    include_prompt_token_length: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    prompts = []
    history_groups = []
    used_preferences = []
    for record in preference_records:
        history_item_sids = _safe_list(record.get("history_item_sid", []))
        history_groups.append(history_item_sids)
        user_preference = str(record.get("user_preference", "") or "").strip()
        used_user_preference = bool(use_user_preference and user_preference)
        used_preferences.append(used_user_preference)
        prompts.append(
            make_dpo_prompt(
                history_item_sids,
                user_preference=user_preference,
                use_user_preference=use_user_preference,
            )
        )

    candidate_groups = []
    for prompt_batch in _batched(prompts, max(1, generation_batch_size)):
        candidate_groups.extend(candidate_generator(prompt_batch))
    if len(candidate_groups) < len(preference_records):
        candidate_groups = list(candidate_groups) + [[] for _ in range(len(preference_records) - len(candidate_groups))]

    rng = random.Random(seed)
    records = []
    failures = []
    for record, prompt, candidates, history_item_sids, used_user_preference in zip(
        preference_records,
        prompts,
        candidate_groups,
        history_groups,
        used_preferences,
    ):
        target_sid = normalize_sid(record["target_item_sid"])
        rejected_items, failure_reason = select_rejected_sids(
            candidates=candidates,
            target_sid=target_sid,
            fallback_sids=fallback_sids,
            history_item_sids=history_item_sids,
            num_negatives=max(1, num_negatives_per_positive),
            fallback_strategy=fallback_strategy,
            filter_history_items=filter_history_items,
            rng=rng,
        )
        if not rejected_items:
            failures.append(
                {
                    "user": record.get("user", ""),
                    "target_item_sid": target_sid,
                    "reason": failure_reason,
                    "num_model_candidates": len(candidates),
                }
            )
            continue

        if failure_reason:
            failures.append(
                {
                    "user": record.get("user", ""),
                    "target_item_sid": target_sid,
                    "reason": failure_reason,
                    "num_model_candidates": len(candidates),
                    "num_rejected": len(rejected_items),
                }
            )

        for rejected_sid, source, rejected_rank in rejected_items:
            row = {
                "prompt": prompt,
                "chosen": format_sid_completion(target_sid, append_completion_newline),
                "rejected": format_sid_completion(rejected_sid, append_completion_newline),
                "target": target_sid,
                "user_preference": record.get("user_preference", ""),
                "user": record.get("user", ""),
                "split": record.get("split", "train"),
                "history_item_sid": record.get("history_item_sid", []),
                "target_item_sid": target_sid,
                "negative_source": source,
                "rejected_rank": rejected_rank,
                "num_model_candidates": len(candidates),
                "used_user_preference": used_user_preference,
            }
            if include_prompt_token_length:
                row["prompt_token_length"] = len(prompt.split())
            records.append(row)

    return records, failures


def _get_hash(values: Iterable[object]) -> str:
    return "-".join(str(value) for value in values)


class ConstrainedSidCandidateGenerator:
    def __init__(self, model_path: str, info_file: str, num_beams: int = 8, max_new_tokens: int = 32):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "build_dpo_pairs.py requires torch and transformers in the active environment"
            ) from exc

        logit_processor_path = os.path.join(os.path.dirname(__file__), "LogitProcessor.py")
        spec = importlib.util.spec_from_file_location("mini_one_rec_logit_processor", logit_processor_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load ConstrainedLogitsProcessor from {logit_processor_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ConstrainedLogitsProcessor = module.ConstrainedLogitsProcessor

        self.torch = torch
        self.GenerationConfig = GenerationConfig
        self.LogitsProcessorList = LogitsProcessorList
        self.ConstrainedLogitsProcessor = ConstrainedLogitsProcessor
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.model = self._load_model(model_path, AutoModelForCausalLM, torch)
        self.model.eval()

        self.prefix_index = 4 if "gpt2" in model_path.lower() else 3
        self.hash_dict = self._build_hash_dict(info_file)

    def _load_model(self, model_path: str, AutoModelForCausalLM, torch):
        if not _is_peft_adapter_dir(model_path):
            return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Loading an SFT LoRA adapter requires peft in the active environment") from exc

        peft_config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        current_vocab_size = model.get_input_embeddings().num_embeddings
        target_vocab_size = len(self.tokenizer)
        if current_vocab_size != target_vocab_size:
            try:
                model.resize_token_embeddings(target_vocab_size, mean_resizing=False)
            except TypeError:
                model.resize_token_embeddings(target_vocab_size)
        model = PeftModel.from_pretrained(model, model_path)
        model.config._name_or_path = model_path
        return model

    def _build_hash_dict(self, info_file: str) -> Dict[str, List[int]]:
        with open(info_file, "r", encoding="utf-8") as fh:
            semantic_ids = [line.split("\t")[0].strip() + "\n" for line in fh if line.strip()]

        info_semantic = [f"### Response:\n{semantic_id}" for semantic_id in semantic_ids]
        if "llama" in str(self.model.config._name_or_path).lower():
            prefix_ids = [self.tokenizer(item).input_ids[1:] for item in info_semantic]
        else:
            prefix_ids = [self.tokenizer(item).input_ids for item in info_semantic]

        hash_dict = {}
        for token_ids in prefix_ids:
            token_ids.append(self.tokenizer.eos_token_id)
            for index in range(self.prefix_index, len(token_ids)):
                if index == self.prefix_index:
                    hash_number = _get_hash(token_ids[:index])
                else:
                    hash_number = _get_hash(token_ids[self.prefix_index:index])
                hash_dict.setdefault(hash_number, set()).add(token_ids[index])

        return {key: list(values) for key, values in hash_dict.items()}

    def _prefix_allowed_tokens_fn(self, batch_id, input_ids):
        return self.hash_dict.get(_get_hash(input_ids), [])

    def __call__(self, prompts: List[str]) -> List[List[str]]:
        if not prompts:
            return []

        encoded = [self.tokenizer(prompt).input_ids for prompt in prompts]
        max_len = max(len(item) for item in encoded)
        input_ids = []
        attention_mask = []
        for item in encoded:
            pad_len = max_len - len(item)
            input_ids.append([self.tokenizer.pad_token_id] * pad_len + item)
            attention_mask.append([0] * pad_len + [1] * len(item))

        generation_config = self.GenerationConfig(
            num_beams=self.num_beams,
            num_return_sequences=self.num_beams,
            length_penalty=0.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            top_k=None,
            top_p=None,
        )

        logits_processor = self.LogitsProcessorList(
            [
                self.ConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn,
                    num_beams=self.num_beams,
                    base_model=str(self.model.config._name_or_path),
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            ]
        )

        device = self.model.device
        with self.torch.no_grad():
            outputs = self.model.generate(
                self.torch.tensor(input_ids).to(device),
                attention_mask=self.torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                logits_processor=logits_processor,
            )

        completions = outputs.sequences[:, max_len:]
        decoded = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        decoded = [text.split("Response:\n")[-1].strip() for text in decoded]
        return [
            decoded[index : index + self.num_beams]
            for index in range(0, len(decoded), self.num_beams)
        ]


def failure_log_path(output_jsonl: str) -> str:
    if output_jsonl.endswith(".jsonl"):
        return output_jsonl[: -len(".jsonl")] + ".failures.jsonl"
    return output_jsonl + ".failures.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Build DPO pairs for MiniOneRec.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--preference_jsonl", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--use_user_preference", type=str_to_bool, default=False)
    parser.add_argument("--generation_batch_size", type=int, default=16)
    parser.add_argument("--num_negatives_per_positive", type=int, default=1)
    parser.add_argument("--fallback_strategy", choices=["random"], default="random")
    parser.add_argument("--filter_history_items", type=str_to_bool, default=True)
    parser.add_argument("--append_completion_newline", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    preference_records = load_jsonl(args.preference_jsonl)
    fallback_sids = load_fallback_sids(args.info_file)
    generator = ConstrainedSidCandidateGenerator(
        model_path=args.model_path,
        info_file=args.info_file,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    records, failures = build_dpo_records(
        preference_records=preference_records,
        candidate_generator=generator,
        fallback_sids=fallback_sids,
        use_user_preference=args.use_user_preference,
        generation_batch_size=args.generation_batch_size,
        num_negatives_per_positive=args.num_negatives_per_positive,
        fallback_strategy=args.fallback_strategy,
        filter_history_items=args.filter_history_items,
        append_completion_newline=args.append_completion_newline,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    write_jsonl(args.output_jsonl, records)
    write_jsonl(failure_log_path(args.output_jsonl), failures)

    print(f"Wrote {len(records)} DPO rows to {args.output_jsonl}")
    print(f"Wrote {len(failures)} failure rows to {failure_log_path(args.output_jsonl)}")


if __name__ == "__main__":
    main()
