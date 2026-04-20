import argparse
import difflib
import json
import os

from sllm.common.data_split import auto_split_eval_tag, iter_labeled_split_rows
from sllm.common.io import read_jsonl, write_json
from sllm.infer.predict import LegacyActionSLLM


INPUT_TEMPLATE = {
    "record_id": "40004",
    "system": "OMS",
    "message": "User with idadmin access to ip10.0.0.225and checked the/dw/main/mainPageSD",
    "create_date": "2026-04-09 11:04:19",
    "state": "PAGE"
}


def load_input_payload(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), "input_file"

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(INPUT_TEMPLATE, f, ensure_ascii=False, indent=2)
    return INPUT_TEMPLATE, "template"


def payload_to_input_record(payload):
    if isinstance(payload, dict):
        return {
            k: v
            for k, v in payload.items()
            if k not in {"command", "reason", "accuracy"}
        }
    if isinstance(payload, str):
        return payload
    raise ValueError("input json must be a dict or string")


def _normalize_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _normalize_value(value):
    if isinstance(value, dict):
        return {k: _normalize_value(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    return _normalize_text(value)


def _message_from_input_record(value) -> str:
    if isinstance(value, dict):
        return str(value.get("message", "")).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _tie_break_score(input_record, candidate_record):
    if not isinstance(input_record, dict) or not isinstance(candidate_record, dict):
        return (0, 0)
    input_fields = _extract_record_fields(input_record)
    candidate_fields = _extract_record_fields(candidate_record)
    shared_keys = sorted(set(input_fields.keys()) & set(candidate_fields.keys()) - {"message"})
    exact_match_count = 0
    for key in shared_keys:
        if _normalize_value(input_fields.get(key)) == _normalize_value(candidate_fields.get(key)):
            exact_match_count += 1
    # More exact auxiliary matches wins first, then more shared auxiliary keys.
    return (exact_match_count, len(shared_keys))


def _extract_record_fields(row):
    if not isinstance(row, dict):
        return {}
    return {
        k: v
        for k, v in row.items()
        if k not in {"record_id", "command", "reason", "accuracy"}
    }


def find_reference_row_by_message(
    input_record,
    valid_file: str | None = None,
    *,
    train_file: str | None = None,
    train_ratio: float = 0.8,
    auto_split: bool = False,
):
    if auto_split:
        if not train_file or not (os.path.isfile(train_file) or os.path.isdir(train_file)):
            return None, "train_file_not_found"
        rows = list(iter_labeled_split_rows(train_file, train_ratio, want_train=False))
    else:
        if not valid_file or not os.path.exists(valid_file):
            return None, "validation_file_not_found"
        rows = list(read_jsonl(valid_file))
    if not rows:
        return None, "validation_file_empty"

    input_msg = _normalize_text(_message_from_input_record(input_record))
    if not input_msg:
        return None, "input_message_missing"

    exact_matches = []
    for row in rows:
        row_record = _extract_record_fields(row)
        row_msg = _normalize_text(_message_from_input_record(row_record))
        if row_msg and row_msg == input_msg:
            exact_matches.append(row)

    if exact_matches:
        best = max(exact_matches, key=lambda row: _tie_break_score(input_record, _extract_record_fields(row)))
        return best, "message_exact"

    scored = []
    for row in rows:
        row_record = _extract_record_fields(row)
        row_msg = _normalize_text(_message_from_input_record(row_record))
        if not row_msg:
            continue
        ratio = difflib.SequenceMatcher(None, input_msg, row_msg).ratio()
        score = (ratio, _tie_break_score(input_record, row_record))
        scored.append((score, row))

    if not scored:
        return None, "validation_message_missing"

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], "message_similarity"


def build_compare_report(
    input_record, model_result, matched_row, match_strategy, reference_tag: str
):
    if matched_row is None:
        return {
            "reference_source": reference_tag,
            "valid_file": reference_tag,
            "match_strategy": match_strategy,
            "input_message": _message_from_input_record(input_record),
            "matched_record_id": None,
            "matched_input_record": None,
            "expected": None,
            "comparison": {
                "command_match": None,
            },
        }

    expected = {
        "command": matched_row.get("command"),
        "reason": matched_row.get("reason"),
    }
    return {
        "reference_source": reference_tag,
        "valid_file": reference_tag,
        "match_strategy": match_strategy,
        "input_message": _message_from_input_record(input_record),
        "matched_record_id": matched_row.get("record_id"),
        "matched_input_record": _extract_record_fields(matched_row),
        "expected": expected,
        "comparison": {
            "command_match": model_result.get("command") == expected["command"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="artifacts/model_dev")
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--input_file", default="sample_data/infer_input.json")
    ap.add_argument("--output_file", default=None)
    ap.add_argument("--valid_file", default=None)
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--auto_split", action="store_true")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--compare_output_file", default=None)
    args = ap.parse_args()

    payload, payload_source = load_input_payload(args.input_file)
    if payload_source == "template":
        print("saved:", args.input_file)
        print("input file not found. used INPUT_TEMPLATE for this run.")

    input_record = payload_to_input_record(payload)
    engine = LegacyActionSLLM(model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir)
    result = engine.predict(input_record)

    output_file = args.output_file or os.path.join(args.model_dir, "infer_result.json")
    parent = os.path.dirname(output_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_json(output_file, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved:", output_file)

    if args.auto_split:
        if not args.train_file:
            raise ValueError("--train_file is required when --auto_split is set")
        reference_tag = auto_split_eval_tag(args.train_file, args.train_ratio)
        matched_row, match_strategy = find_reference_row_by_message(
            input_record,
            auto_split=True,
            train_file=args.train_file,
            train_ratio=args.train_ratio,
        )
    else:
        vf = args.valid_file or "sample_data/valid.jsonl"
        reference_tag = os.path.abspath(vf) if os.path.exists(vf) else vf
        matched_row, match_strategy = find_reference_row_by_message(input_record, vf)

    compare_report = build_compare_report(
        input_record=input_record,
        model_result=result,
        matched_row=matched_row,
        match_strategy=match_strategy,
        reference_tag=reference_tag,
    )
    compare_output_file = args.compare_output_file or os.path.join(args.model_dir, "infer_compare.json")
    compare_parent = os.path.dirname(compare_output_file)
    if compare_parent:
        os.makedirs(compare_parent, exist_ok=True)
    write_json(compare_output_file, compare_report)
    print("saved:", compare_output_file)


if __name__ == "__main__":
    main()
