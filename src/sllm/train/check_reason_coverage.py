import argparse
import os
from typing import Dict, List

from sllm.common.data_split import iter_labeled_split_rows
from sllm.common.io import read_jsonl, write_json
from sllm.infer.predict import LegacyActionSLLM


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_file", default=None, help="JSONL file for inference coverage check")
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--auto_split", action="store_true", help="use validation bucket of train_file (same as training)")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--sample_limit", type=int, default=20, help="max saved bad-case samples")
    ap.add_argument("--output_file", default=None)
    args = ap.parse_args()

    if args.auto_split:
        if not args.train_file:
            raise ValueError("--train_file is required when --auto_split is set")
        row_iter = iter_labeled_split_rows(args.train_file, args.train_ratio, want_train=False)
    else:
        if not args.input_file:
            raise ValueError("--input_file is required unless --auto_split is set")
        row_iter = read_jsonl(args.input_file)

    engine = LegacyActionSLLM(model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir)

    total = 0
    reason_missing = 0
    reason_null = 0
    reason_blank = 0
    malformed_command = 0
    samples: List[Dict] = []

    for row in row_iter:
        total += 1
        input_record = {
            k: v
            for k, v in row.items()
            if k not in {"command", "reason", "accuracy"}
        }
        pred = engine.predict(input_record)

        has_reason_key = "reason" in pred
        reason_val = pred.get("reason")
        command_val = pred.get("command")

        bad_flags = []
        if not has_reason_key:
            reason_missing += 1
            bad_flags.append("reason_missing_key")
        elif reason_val is None:
            reason_null += 1
            bad_flags.append("reason_null")
        elif _is_blank(reason_val):
            reason_blank += 1
            bad_flags.append("reason_blank")

        # Heuristic: command should usually look like ACTION_NAME.
        if isinstance(command_val, str):
            cmd = command_val.strip()
            if cmd and not (cmd.upper() == cmd and "_" in cmd):
                malformed_command += 1
                bad_flags.append("command_malformed_shape")
        else:
            malformed_command += 1
            bad_flags.append("command_not_string")

        if bad_flags and len(samples) < args.sample_limit:
            samples.append(
                {
                    "record_id": row.get("record_id"),
                    "message": input_record.get("message"),
                    "predicted_command": command_val,
                    "predicted_reason": reason_val,
                    "bad_flags": bad_flags,
                }
            )

    report = {
        "model_dir": os.path.abspath(args.model_dir),
        "input_file": os.path.abspath(args.train_file) + " (auto_split valid)" if args.auto_split else os.path.abspath(args.input_file),
        "total": total,
        "reason_coverage": {
            "missing_key_count": reason_missing,
            "null_count": reason_null,
            "blank_count": reason_blank,
            "coverage_ratio": round(
                1.0 - ((reason_missing + reason_null + reason_blank) / total), 4
            )
            if total
            else 0.0,
        },
        "command_shape": {
            "malformed_count": malformed_command,
            "malformed_ratio": round(malformed_command / total, 4) if total else 0.0,
        },
        "samples": samples,
    }

    out_path = args.output_file or os.path.join(args.model_dir, "reason_coverage_report.json")
    write_json(out_path, report)
    print("total:", total)
    print("reason_missing_key:", reason_missing)
    print("reason_null:", reason_null)
    print("reason_blank:", reason_blank)
    print("reason_coverage_ratio:", report["reason_coverage"]["coverage_ratio"])
    print("command_malformed_shape:", malformed_command)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
