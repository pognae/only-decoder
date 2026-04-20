import argparse
import hashlib
import json
import os
import torch
from transformers import AutoModelForCausalLM
from sllm.common.commands import ensure_commands_file, get_commands_path, load_commands, normalize_prediction_with_catalog
from sllm.common.device import resolve_runtime_device
from sllm.common.data_split import auto_split_eval_tag, iter_labeled_split_rows
from sllm.common.io import read_jsonl, write_json
from sllm.common.modeling import load_tokenizer
from sllm.common.prompting import build_inference_prompt, parse_json_fragment
from sllm.train.report_results import save_training_result


def _record_key(row):
    rid = row.get("record_id")
    if rid is not None:
        return str(rid)
    payload = {
        k: v
        for k, v in row.items()
        if k not in {"record_id", "command", "reason", "accuracy"}
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_existing_eval_map(path: str, valid_file_tag: str):
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("valid_file") != valid_file_tag:
                continue
            key = row.get("record_key")
            if key:
                result[key] = row
    return result


def _flush_batch(
    tokenizer,
    model,
    runtime_device,
    pending_batch,
    new_rows,
    eval_map,
    max_new_tokens,
    valid_file_tag,
    commands,
):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        prompts = [build_inference_prompt(row) for _, row in pending_batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    finally:
        tokenizer.padding_side = original_padding_side
    inputs = {k: v.to(runtime_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for (key, row), text in zip(pending_batch, decoded):
        parsed = parse_json_fragment(text)
        parsed = normalize_prediction_with_catalog(parsed, commands, input_record={k: v for k, v in row.items() if k not in {"command", "reason", "accuracy"}})
        detail = {
            "record_key": key,
            "record_id": row.get("record_id"),
            "valid_file": valid_file_tag,
            "expected_command": row.get("command"),
            "predicted_command": parsed.get("command"),
            "command_match": parsed.get("command") == row.get("command"),
        }
        new_rows.append(detail)
        eval_map[key] = detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid_file", default=None, help="JSONL validation set (if not using --auto_split)")
    ap.add_argument(
        "--train_file",
        default=None,
        help="with --auto_split: JSONL file or directory (recursive *.jsonl, same merge as training)",
    )
    ap.add_argument(
        "--auto_split",
        action="store_true",
        help="take validation rows from train_file using same stable split as train_decoder",
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="train fraction when --auto_split (default 0.8)",
    )
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--resume", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    if args.auto_split:
        if not args.train_file:
            raise ValueError("--train_file is required when --auto_split is set")
        if args.valid_file:
            raise ValueError("do not pass --valid_file together with --auto_split")
        if not (os.path.isfile(args.train_file) or os.path.isdir(args.train_file)):
            raise FileNotFoundError(f"train_file not found: {args.train_file!r}")
        row_iter = iter_labeled_split_rows(args.train_file, args.train_ratio, want_train=False)
        valid_file_tag = auto_split_eval_tag(args.train_file, args.train_ratio)
    else:
        if not args.valid_file:
            raise ValueError("--valid_file is required unless --auto_split is set")
        if not os.path.isfile(args.valid_file):
            raise FileNotFoundError(f"valid_file not found: {args.valid_file!r}")
        row_iter = read_jsonl(args.valid_file)
        valid_file_tag = os.path.abspath(args.valid_file)

    runtime_device = resolve_runtime_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.to(runtime_device)
    model.eval()
    commands_path = get_commands_path(args.model_dir)
    ensure_commands_file(commands_path)
    commands = load_commands(commands_path)
    detail_path = os.path.join(args.model_dir, "eval_results.jsonl")
    eval_map = {}
    if args.resume != "never":
        if args.resume == "always" and not os.path.exists(detail_path):
            raise RuntimeError("resume requested but eval_results.jsonl not found")
        eval_map = _load_existing_eval_map(detail_path, valid_file_tag)
    elif os.path.exists(detail_path):
        os.remove(detail_path)

    new_rows = []
    pending_batch = []
    for row in row_iter:
        key = _record_key(row)
        if args.resume != "never" and key in eval_map:
            continue
        pending_batch.append((key, row))
        if len(pending_batch) >= args.batch_size:
            _flush_batch(
                tokenizer=tokenizer,
                model=model,
                runtime_device=runtime_device,
                pending_batch=pending_batch,
                new_rows=new_rows,
                eval_map=eval_map,
                max_new_tokens=args.max_new_tokens,
                valid_file_tag=valid_file_tag,
                commands=commands,
            )
            pending_batch = []

    if pending_batch:
        _flush_batch(
            tokenizer=tokenizer,
            model=model,
            runtime_device=runtime_device,
            pending_batch=pending_batch,
            new_rows=new_rows,
            eval_map=eval_map,
            max_new_tokens=args.max_new_tokens,
            valid_file_tag=valid_file_tag,
            commands=commands,
        )

    if new_rows:
        with open(detail_path, "a", encoding="utf-8") as f:
            for item in new_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = len(eval_map)
    cmd_ok = sum(1 for x in eval_map.values() if x.get("command_match"))

    metrics = {
        "total": total,
        "command_accuracy": round(cmd_ok / total, 4) if total else 0.0,
    }
    out_path = os.path.join(args.model_dir, "metrics.json")
    write_json(out_path, metrics)
    report_path = save_training_result(args.model_dir)
    print(metrics)
    print("runtime device:", runtime_device)
    print("resume mode:", args.resume)
    print("eval detail rows (existing/new):", len(eval_map), len(new_rows))
    print("saved:", detail_path)
    print("saved:", out_path)
    print("saved:", report_path)

if __name__ == "__main__":
    main()
