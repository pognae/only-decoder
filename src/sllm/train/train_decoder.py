import argparse
import inspect
import math
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List
from datasets import Dataset, Features, Sequence, Value
import torch
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from sllm.common.device import resolve_runtime_device
from sllm.common.data_split import has_supervision_labels, is_train_row
from sllm.common.io import iter_jsonl_source, jsonl_source_description, write_json
from sllm.common.modeling import build_model_from_config, load_tokenizer
from sllm.common.prompting import build_prompt, build_target
from sllm.train.report_results import save_training_result


def load_checkpoint_precision(checkpoint_dir: str) -> Dict[str, bool]:
    path = os.path.join(checkpoint_dir, "training_args.bin")
    if not os.path.exists(path):
        return {}
    try:
        ckpt_args = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_args = torch.load(path, map_location="cpu")
    except Exception:
        return {}

    bf16 = getattr(ckpt_args, "bf16", None)
    fp16 = getattr(ckpt_args, "fp16", None)
    result: Dict[str, bool] = {}
    if bf16 is not None:
        result["bf16"] = bool(bf16)
    if fp16 is not None:
        result["fp16"] = bool(fp16)
    return result


def _reset_output_dir_for_fresh_run(output_dir: str) -> None:
    """
    When resume=auto but no checkpoint exists, we may still have stray files in output_dir
    (e.g., data_quality_report.json). In that case, wipe output_dir so a fresh training run
    can proceed without requiring manual deletion.
    """
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            try:
                os.remove(path)
            except IsADirectoryError:
                shutil.rmtree(path)


def _default_hf_datasets_cache_dir() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(repo_root, "artifacts", "_hf_datasets_cache")


@dataclass
class JsonDataCollator:
    tokenizer: object
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]):
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [self.label_pad_token_id] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_file",
        required=True,
        help="JSONL file, or directory (recursive *.jsonl merged in sorted path order)",
    )
    ap.add_argument("--valid_file", default=None)
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--resume", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--auto_split", action="store_true", help="split train_file into train/valid automatically")
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="train fraction when auto_split is enabled (default 0.8 => 20%% validation)",
    )
    args = ap.parse_args()

    # Keep datasets cache inside repo by default to avoid permission issues on locked home dirs.
    hf_cache_dir = os.path.abspath(
        os.environ.get("SLLM_HF_DATASETS_CACHE", _default_hf_datasets_cache_dir())
    )
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", hf_cache_dir)

    runtime_device = resolve_runtime_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    model, cfg = build_model_from_config(args.config_file)
    train_cfg = cfg["training"]
    model.resize_token_embeddings(len(tokenizer))

    max_length = train_cfg["max_length"]
    tokenized_features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )

    def preprocess(example):
        prompt = build_prompt(example)
        target = build_target(example) + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + target_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def build_tokenized_dataset(source: str, *, is_train: bool | None = None) -> Dataset:
        empty = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }

        def rows():
            skipped_unlabeled = 0
            total = 0
            for row in iter_jsonl_source(source):
                total += 1
                # Supervised training rows must have labels. Treat missing command/reason as unlabeled.
                if not has_supervision_labels(row):
                    skipped_unlabeled += 1
                    continue
                if args.auto_split:
                    keep_train = is_train_row(row, args.train_ratio)
                    if is_train is True and not keep_train:
                        continue
                    if is_train is False and keep_train:
                        continue
                yield preprocess(row)
            if skipped_unlabeled:
                print(
                    f"skipped unlabeled rows (missing command/reason): {skipped_unlabeled}/{total} from "
                    f"{jsonl_source_description(source)}"
                )
        try:
            return Dataset.from_generator(
                rows,
                features=tokenized_features,
                cache_dir=hf_cache_dir,
            )
        except ValueError as e:
            # datasets can raise "Instruction train corresponds to no data!" on empty generators.
            if "no data" in str(e).lower():
                return Dataset.from_dict(empty, features=tokenized_features)
            raise

    if not args.valid_file and not args.auto_split:
        raise ValueError("--valid_file is required unless --auto_split is set")

    if args.auto_split:
        train_dataset = build_tokenized_dataset(args.train_file, is_train=True)
        valid_dataset = build_tokenized_dataset(args.train_file, is_train=False)
    else:
        train_dataset = build_tokenized_dataset(args.train_file)
        valid_dataset = build_tokenized_dataset(args.valid_file)
    if len(train_dataset) == 0 and len(valid_dataset) == 0:
        raise ValueError("no labeled rows found for training/evaluation (command/reason required)")
    if len(train_dataset) == 0:
        print("warning: training split is empty; using validation rows for training", flush=True)
        train_dataset = valid_dataset
    if len(valid_dataset) == 0:
        print("warning: validation split is empty; using one training row as validation smoke set", flush=True)
        valid_dataset = train_dataset.select(range(min(1, len(train_dataset))))
    train_rows = len(train_dataset)
    effective_batch = int(train_cfg["per_device_train_batch_size"]) * int(train_cfg["gradient_accumulation_steps"])
    steps_per_epoch = max(1, math.ceil(train_rows / max(1, effective_batch)))
    num_train_epochs = float(train_cfg["num_train_epochs"])
    total_steps_estimate = max(1, math.ceil(steps_per_epoch * num_train_epochs))
    if "warmup_steps" in train_cfg:
        warmup_steps = int(train_cfg["warmup_steps"])
    else:
        warmup_steps = int(total_steps_estimate * float(train_cfg.get("warmup_ratio", 0.0)))

    resume_checkpoint = None
    output_dir_exists = os.path.isdir(args.output_dir)
    last_checkpoint = get_last_checkpoint(args.output_dir) if output_dir_exists else None
    train_metrics_path = os.path.join(args.output_dir, "train_metrics.json")
    output_dir_has_files = output_dir_exists and bool(os.listdir(args.output_dir))
    if args.resume == "always":
        if not last_checkpoint:
            raise RuntimeError("resume requested but no checkpoint found in output_dir")
        resume_checkpoint = last_checkpoint
    elif args.resume == "auto":
        if last_checkpoint:
            resume_checkpoint = last_checkpoint
        elif os.path.exists(train_metrics_path):
            print("training already completed:", args.output_dir)
            print("saved:", train_metrics_path)
            return
        # If output_dir has files but no checkpoint, we treat this as a fresh run and allow overwrite.

    cuda_bf16 = runtime_device == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    use_bf16 = cuda_bf16
    use_fp16 = (runtime_device == "cuda" and not cuda_bf16)
    if resume_checkpoint:
        ckpt_precision = load_checkpoint_precision(resume_checkpoint)
        if runtime_device == "cuda" and ("bf16" in ckpt_precision or "fp16" in ckpt_precision):
            use_bf16 = ckpt_precision.get("bf16", False)
            use_fp16 = ckpt_precision.get("fp16", False)
            # Keep precision flags mutually exclusive.
            if use_bf16 and use_fp16:
                use_fp16 = False

    dataloader_workers = min(8, max(1, (os.cpu_count() or 1) // 2)) if runtime_device in {"cuda", "mps"} else 0
    use_epoch_eval_save = train_rows >= 5000

    training_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        logging_steps=max(int(train_cfg["logging_steps"]), max(50, steps_per_epoch // 20)),
        save_strategy="epoch" if use_epoch_eval_save else "steps",
        warmup_steps=warmup_steps,
        weight_decay=float(train_cfg["weight_decay"]),
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
    )
    # If resume=auto but no checkpoint exists, wipe output_dir so we can do a clean fresh run.
    if args.resume == "auto" and not resume_checkpoint and output_dir_has_files:
        print("output_dir is not empty but no checkpoint found; resetting for fresh run:", args.output_dir)
        _reset_output_dir_for_fresh_run(args.output_dir)
    if not use_epoch_eval_save:
        training_kwargs["save_steps"] = int(train_cfg["save_steps"])
        training_kwargs["eval_steps"] = int(train_cfg["eval_steps"])

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "tf32" in ta_params:
        training_kwargs["tf32"] = (runtime_device == "cuda")
    if "dataloader_num_workers" in ta_params:
        training_kwargs["dataloader_num_workers"] = dataloader_workers
    if "dataloader_persistent_workers" in ta_params:
        training_kwargs["dataloader_persistent_workers"] = (dataloader_workers > 0)
    if "dataloader_pin_memory" in ta_params:
        training_kwargs["dataloader_pin_memory"] = (runtime_device == "cuda")
    if "use_cpu" in ta_params:
        training_kwargs["use_cpu"] = (runtime_device == "cpu")
    elif "no_cuda" in ta_params:
        training_kwargs["no_cuda"] = (runtime_device == "cpu")
    if runtime_device == "mps" and "use_mps_device" in ta_params:
        training_kwargs["use_mps_device"] = True

    eval_strategy_value = "epoch" if use_epoch_eval_save else "steps"
    # Transformers v5 renamed evaluation_strategy -> eval_strategy.
    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = eval_strategy_value
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy_value

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=JsonDataCollator(tokenizer),
    )
    train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    train_metrics_path = f"{args.output_dir}/train_metrics.json"
    write_json(train_metrics_path, train_output.metrics)
    report_path = save_training_result(args.output_dir)
    print("saved model:", args.output_dir)
    print("resume checkpoint:", resume_checkpoint)
    print("runtime device:", runtime_device)
    print("datasets cache dir:", hf_cache_dir)
    print("precision mode:", "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"))
    print("eval/save strategy:", "epoch" if use_epoch_eval_save else "steps")
    print("dataloader workers:", dataloader_workers)
    print("saved:", train_metrics_path)
    print("saved:", report_path)

if __name__ == "__main__":
    main()
