import argparse
import json
import os

from sllm.common.io import read_jsonl


def _is_missing_reason(row: dict) -> bool:
    if "reason" not in row:
        return True
    value = row.get("reason")
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _write_jsonl(path: str, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", default=None)
    ap.add_argument("--in_place", action="store_true")
    args = ap.parse_args()

    if args.in_place and args.output_file:
        raise ValueError("use either --in_place or --output_file, not both")

    output_file = args.input_file if args.in_place else (args.output_file or "train_data/train_filtered.jsonl")

    kept = []
    total = 0
    removed = 0
    for row in read_jsonl(args.input_file):
        total += 1
        if _is_missing_reason(row):
            removed += 1
            continue
        kept.append(row)

    _write_jsonl(output_file, kept)

    print("input_file:", args.input_file)
    print("output_file:", output_file)
    print("total_rows:", total)
    print("kept_rows:", len(kept))
    print("removed_rows_missing_reason:", removed)


if __name__ == "__main__":
    main()
