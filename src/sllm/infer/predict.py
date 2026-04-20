import json
import os
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM

from sllm.common.commands import (
    ensure_commands_file,
    get_commands_path,
    load_commands,
    normalize_prediction_with_catalog,
)
from sllm.common.device import resolve_runtime_device
from sllm.common.modeling import load_tokenizer
from sllm.common.prompting import FAIL_SAFE_REASON, build_inference_prompt, parse_json_fragment


def _fail_safe_response():
    return {
        "command": "NO_ACTION",
        "reason": FAIL_SAFE_REASON,
        "accuracy": {"command_accuracy": None},
    }


def _sanitize_input_record(input_record):
    if isinstance(input_record, dict):
        return {
            k: v
            for k, v in input_record.items()
            if k not in {"command", "reason", "accuracy"}
        }
    return input_record


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    text = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return text in {"1", "true", "yes", "on", "y"}


class LegacyActionSLLM:
    def __init__(self, model_dir: str, tokenizer_dir: str = None):
        self.model_dir = model_dir
        self.tokenizer_dir = tokenizer_dir or model_dir
        self.tokenizer = load_tokenizer(self.tokenizer_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()
        self.runtime_device = resolve_runtime_device("auto")
        self.model.to(self.runtime_device)
        self.metrics = {}
        self.commands_path = os.environ.get("SLLM_COMMANDS_FILE", get_commands_path(self.model_dir))
        ensure_commands_file(self.commands_path)

        # Safer deterministic generation defaults for malformed JSON reduction.
        self.max_new_tokens = max(32, _env_int("SLLM_MAX_NEW_TOKENS", 96))
        self.repetition_penalty = max(1.0, _env_float("SLLM_REPETITION_PENALTY", 1.08))
        self.no_repeat_ngram_size = max(0, _env_int("SLLM_NO_REPEAT_NGRAM_SIZE", 4))

        # Inference debug artifacts (for fast root-cause analysis when NO_ACTION appears).
        self.save_infer_debug = _env_bool("SLLM_SAVE_INFER_DEBUG", True)
        self.append_infer_debug = _env_bool("SLLM_APPEND_INFER_DEBUG", True)
        self.debug_last_path = os.path.join(self.model_dir, "infer_debug_last.json")
        self.debug_history_path = os.path.join(self.model_dir, "infer_debug_history.jsonl")

        mp = os.path.join(model_dir, "metrics.json")
        if os.path.exists(mp):
            with open(mp, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    def _generation_kwargs(self):
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.model.config, "eos_token_id", None)
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "eos_token_id": eos_token_id,
            "pad_token_id": eos_token_id,
            "repetition_penalty": self.repetition_penalty,
            "use_cache": True,
        }
        if self.no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        return kwargs

    def _write_infer_debug(self, payload: dict):
        if not self.save_infer_debug:
            return
        row = dict(payload)
        row["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            with open(self.debug_last_path, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
            if self.append_infer_debug:
                with open(self.debug_history_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            # Debug writing must never block inference.
            return

    def predict(self, input_record):
        safe_input_record = _sanitize_input_record(input_record)
        prompt = ""
        full_text = ""
        generated_text = ""
        parsed = {}
        try:
            prompt = build_inference_prompt(safe_input_record)
            max_ctx = getattr(self.model.config, "max_position_embeddings", None)
            # Keep a small buffer for generated tokens.
            max_input_len = max(1, int(max_ctx) - 128) if max_ctx else None
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=(max_input_len is not None),
                max_length=max_input_len,
            )
            inputs = {k: v.to(self.runtime_device) for k, v in inputs.items()}
            input_len = int(inputs["input_ids"].shape[-1])
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self._generation_kwargs(),
                )
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            gen_tokens = outputs[0][input_len:]
            generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            parsed = parse_json_fragment(generated_text if generated_text.strip() else full_text)
        except Exception as e:
            result = _fail_safe_response()
            self._write_infer_debug(
                {
                    "input_record": safe_input_record,
                    "prompt": prompt,
                    "generated_full_text": full_text,
                    "generated_text_only": generated_text,
                    "parsed": parsed,
                    "result": result,
                    "exception": repr(e),
                    "generation_kwargs": self._generation_kwargs(),
                }
            )
            return result

        if parsed.get("_parse_failed"):
            result = _fail_safe_response()
            self._write_infer_debug(
                {
                    "input_record": safe_input_record,
                    "prompt": prompt,
                    "generated_full_text": full_text,
                    "generated_text_only": generated_text,
                    "parsed": parsed,
                    "result": result,
                    "exception": None,
                    "generation_kwargs": self._generation_kwargs(),
                }
            )
            return result

        normalized = normalize_prediction_with_catalog(
            parsed,
            load_commands(self.commands_path),
            input_record=safe_input_record,
        )
        result = {
            "command": normalized.get("command", "NO_ACTION"),
            "reason": str(normalized.get("reason", "")).strip() or FAIL_SAFE_REASON,
            "accuracy": {
                "command_accuracy": self.metrics.get("command_accuracy"),
            },
        }
        self._write_infer_debug(
            {
                "input_record": safe_input_record,
                "prompt": prompt,
                "generated_full_text": full_text,
                "generated_text_only": generated_text,
                "parsed": parsed,
                "normalized": normalized,
                "result": result,
                "exception": None,
                "generation_kwargs": self._generation_kwargs(),
            }
        )
        return result
