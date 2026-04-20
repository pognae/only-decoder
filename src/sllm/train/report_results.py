import argparse
import glob
import json
import os
from typing import Dict, Optional

from sllm.common.io import write_json
from sllm.infer.predict import LegacyActionSLLM
from sllm.common.data_split import auto_split_eval_tag
from sllm.infer.run_infer_json import (
    build_compare_report,
    find_reference_row_by_message,
    load_input_payload,
    payload_to_input_record,
)


def _read_json_if_exists(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_trainer_state(model_dir: str) -> Optional[str]:
    root_state = os.path.join(model_dir, "trainer_state.json")
    if os.path.exists(root_state):
        return root_state

    candidates = []
    for checkpoint_dir in glob.glob(os.path.join(model_dir, "checkpoint-*")):
        state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if not os.path.exists(state_path):
            continue
        name = os.path.basename(checkpoint_dir)
        try:
            step = int(name.split("-")[-1])
        except ValueError:
            step = -1
        candidates.append((step, state_path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _extract_training_info(state: Optional[Dict], train_metrics: Optional[Dict]) -> Dict:
    metrics = train_metrics or {}
    if not state:
        return {
            "global_step": metrics.get("global_step"),
            "epoch": metrics.get("epoch"),
            "max_steps": metrics.get("max_steps"),
            "train_loss": metrics.get("train_loss"),
            "train_runtime": metrics.get("train_runtime"),
            "eval_loss": metrics.get("eval_loss"),
        }

    log_history = state.get("log_history", [])
    train_log = next((x for x in reversed(log_history) if "train_loss" in x or "train_runtime" in x), {})
    eval_log = next((x for x in reversed(log_history) if "eval_loss" in x), {})

    return {
        "global_step": state.get("global_step"),
        "epoch": metrics.get("epoch", state.get("epoch")),
        "max_steps": state.get("max_steps"),
        "train_loss": metrics.get("train_loss", train_log.get("train_loss")),
        "train_runtime": metrics.get("train_runtime", train_log.get("train_runtime")),
        "eval_loss": eval_log.get("eval_loss"),
    }


def build_training_result(model_dir: str) -> Dict:
    trainer_state_path = _find_latest_trainer_state(model_dir)
    trainer_state = _read_json_if_exists(trainer_state_path) if trainer_state_path else None
    train_metrics_path = os.path.join(model_dir, "train_metrics.json")
    train_metrics = _read_json_if_exists(train_metrics_path)
    metrics_path = os.path.join(model_dir, "metrics.json")
    metrics = _read_json_if_exists(metrics_path)

    training_info = _extract_training_info(trainer_state, train_metrics)
    return {
        "summary": {
            **training_info,
            "command_accuracy": (metrics or {}).get("command_accuracy"),
            "evaluation_total": (metrics or {}).get("total"),
        },
        "sources": {
            "trainer_state": trainer_state_path,
            "train_metrics": train_metrics_path if train_metrics is not None else None,
            "metrics": metrics_path if metrics is not None else None,
        },
    }


def save_training_result(model_dir: str) -> str:
    out_path = os.path.join(model_dir, "training_result.json")
    report = build_training_result(model_dir)
    write_json(out_path, report)
    return out_path


def build_infer_result(
    model_dir: str,
    input_file: str,
    valid_file: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    *,
    train_file: Optional[str] = None,
    auto_split: bool = False,
    train_ratio: float = 0.8,
) -> Dict:
    payload, _ = load_input_payload(input_file)
    input_record = payload_to_input_record(payload)
    engine = LegacyActionSLLM(model_dir=model_dir, tokenizer_dir=tokenizer_dir)
    prediction = engine.predict(input_record)
    if auto_split:
        if not train_file:
            raise ValueError("train_file is required when auto_split is True")
        matched_row, match_strategy = find_reference_row_by_message(
            input_record,
            auto_split=True,
            train_file=train_file,
            train_ratio=train_ratio,
        )
        reference_tag = auto_split_eval_tag(train_file, train_ratio)
    else:
        vf = valid_file or "sample_data/valid.jsonl"
        matched_row, match_strategy = find_reference_row_by_message(input_record, vf)
        reference_tag = os.path.abspath(vf) if os.path.exists(vf) else vf
    compare_report = build_compare_report(
        input_record=input_record,
        model_result=prediction,
        matched_row=matched_row,
        match_strategy=match_strategy,
        reference_tag=reference_tag,
    )

    return {
        "message": compare_report.get("input_message"),
        "command": prediction.get("command"),
        "reason": prediction.get("reason"),
        "accuracy": prediction.get("accuracy", {}),
        "reference": {
            "match_strategy": compare_report.get("match_strategy"),
            "matched_record_id": compare_report.get("matched_record_id"),
            "expected": compare_report.get("expected"),
            "comparison": compare_report.get("comparison"),
        },
    }


def save_infer_result(
    model_dir: str,
    input_file: str,
    valid_file: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    *,
    train_file: Optional[str] = None,
    auto_split: bool = False,
    train_ratio: float = 0.8,
) -> str:
    out_path = output_file or os.path.join(model_dir, "report_result.json")
    report = build_infer_result(
        model_dir=model_dir,
        input_file=input_file,
        valid_file=valid_file,
        tokenizer_dir=tokenizer_dir,
        train_file=train_file,
        auto_split=auto_split,
        train_ratio=train_ratio,
    )
    write_json(out_path, report)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_file", default=None)
    ap.add_argument("--valid_file", default=None)
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--auto_split", action="store_true")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--output_file", default=None)
    args = ap.parse_args()
    if args.input_file:
        if args.auto_split and not args.train_file:
            raise ValueError("--train_file is required when --auto_split is set")
        out_path = save_infer_result(
            model_dir=args.model_dir,
            input_file=args.input_file,
            valid_file=args.valid_file,
            tokenizer_dir=args.tokenizer_dir,
            output_file=args.output_file,
            train_file=args.train_file,
            auto_split=args.auto_split,
            train_ratio=args.train_ratio,
        )
        with open(out_path, "r", encoding="utf-8") as f:
            print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
        print("saved:", out_path)
        return

    out_path = save_training_result(args.model_dir)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
