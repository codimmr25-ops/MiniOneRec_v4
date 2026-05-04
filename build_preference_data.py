import argparse
import ast
import importlib.util
import json
import os
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd


DEFAULT_PREFERENCE_PROMPT_1 = (
    "A user has bought a variety of {dataset_full_name} items in chronological order: \n{item_titles}. "
    "\nAfter purchasing these items, the user then chose to buy the following item: {target_item_info}. "
    "Based on this purchasing pattern and the final choice, please analyze what this reveals about the user's "
    "personalized preferences. Provide a brief third-person summary that explains the logical progression from "
    "the historical purchases to this final choice, highlighting the key factors that influence the user's "
    "decisions. In your analysis, do not mention the specific name/title of the target item, but focus on the "
    "characteristics and features that drove the user to make this particular choice. Your analysis should be "
    "brief and in the third person."
)

DEFAULT_PREFERENCE_PROMPT_2 = (
    "A user has purchased a chronological list of {dataset_full_name} items: \n{item_titles}. \nThen, the user "
    "made a final purchase: {target_item_info}. Based on this purchasing progression, please provide a concise "
    "analysis of what drove the user to choose this specific type of item. Focus on the logical connection between "
    "the user's purchase history and this final choice. Explain the purchasing pattern and what it reveals about "
    "the user's evolving needs or interests. Your analysis should be brief and in the third person."
)

HISTORY_ONLY_PREFERENCE_PROMPT = (
    "A user has bought the following {dataset_full_name} items in chronological order:\n{history_item_info}\n\n"
    "Using only this purchase history, write a concise third-person summary of the user's likely preferences. "
    "Do not infer from any future or target item. Keep the summary brief, about 30-80 tokens."
)


def _safe_literal(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
    return [value]


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_description(description, title: str) -> str:
    if not description:
        return ""
    if isinstance(description, list):
        candidates = [str(item).strip() for item in description if str(item).strip()]
        return max(candidates, key=len) if candidates else ""
    if isinstance(description, str):
        stripped = description.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return stripped
            return _normalize_description(parsed, title)
        return stripped
    return str(description).strip()


def _combine_sid(item_id, sid_index: Dict[str, Sequence[str]]) -> str:
    item_id = str(item_id)
    if item_id not in sid_index:
        return item_id
    tokens = sid_index[item_id]
    return "".join(tokens[:3]) if isinstance(tokens, list) else str(tokens)


def _infer_split(input_csv: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(input_csv))).lower()
    if parent in {"train", "valid", "test"}:
        return parent
    return "train"


def _infer_dataset_label(item_meta_path: str) -> str:
    filename = os.path.basename(item_meta_path)
    if filename.endswith(".item.json"):
        filename = filename[: -len(".item.json")]
    return filename.replace("_", " ")


def _load_prompt_templates() -> Tuple[str, str]:
    utils_path = os.path.join(os.path.dirname(__file__), "rq", "text2emb", "utils.py")
    if not os.path.exists(utils_path):
        return DEFAULT_PREFERENCE_PROMPT_1, DEFAULT_PREFERENCE_PROMPT_2

    spec = importlib.util.spec_from_file_location("mini_one_rec_text2emb_utils", utils_path)
    if spec is None or spec.loader is None:
        return DEFAULT_PREFERENCE_PROMPT_1, DEFAULT_PREFERENCE_PROMPT_2

    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return DEFAULT_PREFERENCE_PROMPT_1, DEFAULT_PREFERENCE_PROMPT_2

    return (
        getattr(module, "preference_prompt_1", DEFAULT_PREFERENCE_PROMPT_1),
        getattr(module, "preference_prompt_2", DEFAULT_PREFERENCE_PROMPT_2),
    )


def _load_get_res_batch():
    utils_path = os.path.join(os.path.dirname(__file__), "rq", "text2emb", "utils.py")
    if not os.path.exists(utils_path):
        raise FileNotFoundError(f"Cannot find text2emb utils at {utils_path}")

    spec = importlib.util.spec_from_file_location("mini_one_rec_text2emb_utils_runtime", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load text2emb utils from {utils_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_res_batch


def _build_target_item_info(target_title: str, target_description: str) -> str:
    if target_description:
        return f'title="{target_title}", description="{target_description}"'
    return f'title="{target_title}"'


def _build_history_item_info(
    history_item_ids: Sequence[object],
    history_titles: Sequence[str],
    item_meta: Dict[str, Dict[str, object]],
) -> str:
    items = []
    for index, (item_id, title) in enumerate(zip(history_item_ids, history_titles), start=1):
        normalized_title = _normalize_text(title)
        meta = item_meta.get(str(item_id), {})
        description = _normalize_description(meta.get("description"), normalized_title)
        if description:
            items.append(f'{index}. title="{normalized_title}", description="{description}"')
        else:
            items.append(f'{index}. title="{normalized_title}"')
    return "\n".join(items)


def _build_target_conditioned_prompt(
    history_titles: List[str],
    target_title: str,
    target_description: str,
    dataset_label: str,
    prompt_1: str,
    prompt_2: str,
) -> str:
    history_str = ", ".join(f'"{title}"' for title in history_titles)
    target_item_info = _build_target_item_info(target_title, target_description)
    template = prompt_1 if target_description else prompt_2
    return template.format(
        dataset_full_name=dataset_label,
        item_titles=history_str,
        target_item_info=target_item_info,
    )


def _build_history_only_prompt(
    history_item_ids: Sequence[object],
    history_titles: List[str],
    item_meta: Dict[str, Dict[str, object]],
    dataset_label: str,
) -> str:
    return HISTORY_ONLY_PREFERENCE_PROMPT.format(
        dataset_full_name=dataset_label,
        history_item_info=_build_history_item_info(history_item_ids, history_titles, item_meta),
    )


def _batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def build_api_info(provider: str, api_key_env: str, base_url: str = "") -> Dict[str, object]:
    env_value = os.getenv(api_key_env, "").strip()
    if not env_value:
        raise ValueError(f"Environment variable {api_key_env} is empty or unset")

    if env_value.startswith("["):
        keys = json.loads(env_value)
    else:
        keys = [chunk.strip() for chunk in env_value.replace("\n", ",").split(",") if chunk.strip()]

    if not keys:
        raise ValueError(f"Environment variable {api_key_env} does not contain any API keys")

    api_info = {
        "provider": provider,
        "api_key_list": keys,
    }
    if base_url:
        api_info["base_url"] = base_url
    return api_info


def build_preference_records(
    dataframe: pd.DataFrame,
    item_meta: Dict[str, Dict[str, object]],
    sid_index: Dict[str, Sequence[str]],
    llm_fn: Callable[[str, List[str], int, Dict[str, object]], List[str]],
    provider: str,
    llm_model: str,
    dataset_label: str,
    batch_size: int = 4,
    max_attempts: int = 3,
    max_tokens: int = 256,
    api_info: Dict[str, object] = None,
    split: str = "train",
    prompt_templates: Tuple[str, str] = None,
    preference_mode: str = "history_only",
):
    if preference_mode not in {"history_only", "target_conditioned_debug", "none"}:
        raise ValueError(f"Unsupported preference_mode: {preference_mode}")

    prompt_1, prompt_2 = prompt_templates or _load_prompt_templates()

    pending = []
    records = []
    failures = []
    for _, row in dataframe.iterrows():
        history_item_ids = _safe_literal(row.get("history_item_id"))
        history_item_titles = _safe_literal(row.get("history_item_title"))
        history_item_sids = _safe_literal(row.get("history_item_sid"))

        if not history_item_ids or not history_item_titles or not history_item_sids:
            failures.append(
                {
                    "user": _normalize_text(row.get("user_id") or row.get("user_id_original_str")),
                    "reason": "insufficient_history",
                }
            )
            continue

        target_item_id = _normalize_text(row.get("item_id"))
        target_item_title = _normalize_text(row.get("item_title"))
        target_item_sid = _normalize_text(row.get("item_sid")) or _combine_sid(target_item_id, sid_index)
        target_meta = item_meta.get(target_item_id, {})
        target_description = _normalize_description(target_meta.get("description"), target_item_title)

        base_record = {
            "split": split,
            "user": _normalize_text(row.get("user_id") or row.get("user_id_original_str")),
            "history_item_id": history_item_ids,
            "history_item_title": history_item_titles,
            "history_item_sid": history_item_sids,
            "target_item_id": target_item_id,
            "target_item_title": target_item_title,
            "target_item_sid": target_item_sid,
            "preference_source": f"{provider}:{llm_model}:{preference_mode}",
        }

        if preference_mode == "none":
            record = dict(base_record)
            record["user_preference"] = ""
            records.append(record)
            continue

        normalized_history_titles = [_normalize_text(title) for title in history_item_titles]
        if preference_mode == "history_only":
            prompt = _build_history_only_prompt(
                history_item_ids=history_item_ids,
                history_titles=normalized_history_titles,
                item_meta=item_meta,
                dataset_label=dataset_label,
            )
        else:
            prompt = _build_target_conditioned_prompt(
                history_titles=normalized_history_titles,
                target_title=target_item_title,
                target_description=target_description,
                dataset_label=dataset_label,
                prompt_1=prompt_1,
                prompt_2=prompt_2,
            )

        pending.append(
            {
                "prompt": prompt,
                "record": base_record,
            }
        )

    for _, batch in _batched(pending, max(1, batch_size)):
        prompts = [item["prompt"] for item in batch]
        batch_results = None
        last_error = None

        for _attempt in range(max_attempts):
            try:
                batch_results = llm_fn(llm_model, prompts, max_tokens, api_info or {})
                if batch_results is None:
                    raise RuntimeError("llm_fn returned None")
                break
            except Exception as exc:  # pragma: no cover - exercised in tests with injected failures
                last_error = exc

        if batch_results is None:
            for item in batch:
                failures.append(
                    {
                        "user": item["record"]["user"],
                        "reason": "llm_error",
                        "error": str(last_error) if last_error else "unknown_error",
                    }
                )
            continue

        if len(batch_results) < len(batch):
            batch_results = list(batch_results) + [""] * (len(batch) - len(batch_results))

        for item, response in zip(batch, batch_results):
            preference_text = _normalize_text(response)
            if not preference_text:
                failures.append(
                    {
                        "user": item["record"]["user"],
                        "reason": "empty_response",
                    }
                )
                continue

            record = dict(item["record"])
            record["user_preference"] = preference_text
            records.append(record)

    return records, failures


def write_jsonl(path: str, rows: List[Dict[str, object]]):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def failure_log_path(output_jsonl: str) -> str:
    if output_jsonl.endswith(".jsonl"):
        return output_jsonl[: -len(".jsonl")] + ".failures.jsonl"
    return output_jsonl + ".failures.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Build user preference JSONL for MiniOneRec DPO.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--item_meta_path", required=True)
    parser.add_argument("--sid_index_path", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--provider", default="")
    parser.add_argument("--llm_model", default="")
    parser.add_argument("--api_key_env", default="")
    parser.add_argument("--base_url", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument(
        "--preference_mode",
        choices=["history_only", "target_conditioned_debug", "none"],
        default="history_only",
    )
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input_csv)
    with open(args.item_meta_path, "r", encoding="utf-8") as fh:
        item_meta = json.load(fh)
    with open(args.sid_index_path, "r", encoding="utf-8") as fh:
        sid_index = json.load(fh)

    api_info = None
    get_res_batch = None
    if args.preference_mode != "none":
        if not args.provider or not args.llm_model or not args.api_key_env:
            raise ValueError("--provider, --llm_model, and --api_key_env are required unless --preference_mode=none")
        api_info = build_api_info(args.provider, args.api_key_env, args.base_url)
        get_res_batch = _load_get_res_batch()
    dataset_label = _infer_dataset_label(args.item_meta_path)
    split = _infer_split(args.input_csv)

    records, failures = build_preference_records(
        dataframe=dataframe,
        item_meta=item_meta,
        sid_index=sid_index,
        llm_fn=get_res_batch,
        provider=args.provider,
        llm_model=args.llm_model,
        dataset_label=dataset_label,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
        api_info=api_info,
        split=split,
        preference_mode=args.preference_mode,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    write_jsonl(args.output_jsonl, records)
    write_jsonl(failure_log_path(args.output_jsonl), failures)

    print(f"Wrote {len(records)} preference rows to {args.output_jsonl}")
    print(f"Wrote {len(failures)} failure rows to {failure_log_path(args.output_jsonl)}")


if __name__ == "__main__":
    main()
