#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import Counter


def load_index(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(key): list(value) for key, value in data.items()}


def summarize(method, path):
    index = load_index(path)
    codes = [tuple(value) for _, value in sorted(index.items(), key=lambda item: int(item[0]))]
    total = len(codes)
    counts = Counter(codes)
    unique_full_paths = len(counts)
    collision_groups = sum(1 for count in counts.values() if count > 1)
    collided_items = sum(count for count in counts.values() if count > 1)
    max_conflict = max(counts.values()) if counts else 0
    num_levels = max((len(code) for code in codes), default=0)
    unique_per_level = []
    for level in range(num_levels):
        unique_per_level.append(len({code[level] for code in codes if len(code) > level}))

    return {
        "method": method,
        "path": path,
        "total_items": total,
        "num_levels": num_levels,
        "unique_full_paths": unique_full_paths,
        "collision_rate": 0.0 if total == 0 else 1.0 - unique_full_paths / total,
        "collision_groups": collision_groups,
        "collided_items": collided_items,
        "max_conflict": max_conflict,
        "unique_per_level": ";".join(str(value) for value in unique_per_level),
    }


def parse_method_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use METHOD=/path/to/index.json")
    method, path = value.split("=", 1)
    method = method.strip()
    path = path.strip()
    if not method:
        raise argparse.ArgumentTypeError("Method name cannot be empty")
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"Index file not found: {path}")
    return method, path


def main():
    parser = argparse.ArgumentParser(description="Compare MiniOneRec SID index collision statistics.")
    parser.add_argument("--index", action="append", type=parse_method_arg, required=True, help="METHOD=/path/to/index.json")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    rows = [summarize(method, path) for method, path in args.index]
    rows.sort(key=lambda row: (row["collision_rate"], row["num_levels"], row["method"]))

    print("\t".join([
        "method",
        "items",
        "levels",
        "unique_paths",
        "collision_rate",
        "collision_groups",
        "collided_items",
        "max_conflict",
        "unique_per_level",
    ]))
    for row in rows:
        print("\t".join([
            row["method"],
            str(row["total_items"]),
            str(row["num_levels"]),
            str(row["unique_full_paths"]),
            f"{row['collision_rate']:.8f}",
            str(row["collision_groups"]),
            str(row["collided_items"]),
            str(row["max_conflict"]),
            row["unique_per_level"],
        ]))

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
