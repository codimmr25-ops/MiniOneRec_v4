import argparse
import json
import sys
from collections import Counter


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
    return rows


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def main():
    parser = argparse.ArgumentParser(description="Check MiniOneRec DPO pair JSONL quality.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--fallback_threshold", type=float, default=0.5)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    total = len(rows)

    chosen_eq_rejected = 0
    empty_prompt = 0
    empty_chosen = 0
    empty_rejected = 0
    chosen_newline = 0
    rejected_newline = 0
    negative_sources = Counter()
    rejected_values = Counter()
    used_user_preference = 0
    prompt_lengths = []

    for row in rows:
        prompt = str(row.get("prompt", ""))
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))

        if chosen.strip() == rejected.strip():
            chosen_eq_rejected += 1
        if not prompt.strip():
            empty_prompt += 1
        if not chosen.strip():
            empty_chosen += 1
        if not rejected.strip():
            empty_rejected += 1
        if chosen.endswith("\n"):
            chosen_newline += 1
        if rejected.endswith("\n"):
            rejected_newline += 1

        negative_sources[str(row.get("negative_source", "missing"))] += 1
        rejected_values[rejected.strip()] += 1
        if bool(row.get("used_user_preference", False)):
            used_user_preference += 1

        if "prompt_token_length" in row:
            try:
                prompt_lengths.append(int(row["prompt_token_length"]))
            except (TypeError, ValueError):
                pass
        else:
            prompt_lengths.append(len(prompt.split()))

    fallback_count = sum(
        count
        for source, count in negative_sources.items()
        if "fallback" in source or source in {"rule_fallback", "random_fallback"}
    )
    fallback_ratio = fallback_count / total if total else 0.0

    print(f"rows: {total}")
    print(f"chosen == rejected: {chosen_eq_rejected}")
    print(f"empty prompt: {empty_prompt}")
    print(f"empty chosen: {empty_chosen}")
    print(f"empty rejected: {empty_rejected}")
    print("negative_source distribution:")
    for source, count in negative_sources.most_common():
        print(f"  {source}: {count} ({pct(count, total):.2f}%)")
    print(f"fallback negative ratio: {fallback_ratio:.4f} ({pct(fallback_count, total):.2f}%)")
    print(f"used_user_preference: {used_user_preference} ({pct(used_user_preference, total):.2f}%)")
    print(f"chosen endswith newline: {chosen_newline}/{total} ({pct(chosen_newline, total):.2f}%)")
    print(f"rejected endswith newline: {rejected_newline}/{total} ({pct(rejected_newline, total):.2f}%)")
    print("top repeated rejected:")
    for rejected, count in rejected_values.most_common(20):
        if count <= 1:
            break
        print(f"  {rejected}: {count}")

    if prompt_lengths:
        sorted_lengths = sorted(prompt_lengths)
        p50 = sorted_lengths[len(sorted_lengths) // 2]
        p95 = sorted_lengths[int(0.95 * (len(sorted_lengths) - 1))]
        print(
            "prompt token length distribution: "
            f"min={sorted_lengths[0]}, p50={p50}, p95={p95}, max={sorted_lengths[-1]}"
        )

    failed = False
    if chosen_eq_rejected > 0:
        print("FAIL: chosen == rejected rows found", file=sys.stderr)
        failed = True
    if empty_chosen > 0 or empty_rejected > 0:
        print("FAIL: empty chosen/rejected rows found", file=sys.stderr)
        failed = True
    if fallback_ratio > args.fallback_threshold:
        print(
            f"FAIL: fallback ratio {fallback_ratio:.4f} exceeds threshold {args.fallback_threshold:.4f}",
            file=sys.stderr,
        )
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
