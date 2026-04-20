"""Stable train/validation split for JSONL legacy rows (matches train_decoder / evaluate)."""

import hashlib
import json
import os
from typing import Any, Dict, Iterator

from sllm.common.io import iter_jsonl_source, resolve_jsonl_source_paths


def has_supervision_labels(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    cmd = row.get("command")
    reason = row.get("reason")
    if cmd is None or (isinstance(cmd, str) and cmd.strip() == ""):
        return False
    if reason is None or (isinstance(reason, str) and reason.strip() == ""):
        return False
    return True


def stable_split_bucket(row: dict) -> int:
    rid = row.get("record_id") if isinstance(row, dict) else None
    if rid is not None:
        key = str(rid)
    else:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 10000


def is_train_row(row: dict, train_ratio: float) -> bool:
    bucket = stable_split_bucket(row)
    threshold = int(max(0.0, min(1.0, train_ratio)) * 10000)
    return bucket < threshold


def iter_labeled_split_rows(
    train_source: str, train_ratio: float, *, want_train: bool
) -> Iterator[Dict[str, Any]]:
    """Yield labeled rows from one JSONL file or merged ``*.jsonl`` under a directory."""
    for row in iter_jsonl_source(train_source):
        if not has_supervision_labels(row):
            continue
        in_train = is_train_row(row, train_ratio)
        if want_train and not in_train:
            continue
        if not want_train and in_train:
            continue
        yield row


def iter_bucket_rows(train_source: str, train_ratio: float, *, want_train: bool) -> Iterator[Dict[str, Any]]:
    """All JSONL rows in train or valid bucket (includes unlabeled; for reporting)."""
    for row in iter_jsonl_source(train_source):
        if not isinstance(row, dict):
            continue
        in_train = is_train_row(row, train_ratio)
        if want_train and not in_train:
            continue
        if not want_train and in_train:
            continue
        yield row


def auto_split_eval_tag(train_source: str, train_ratio: float) -> str:
    """Stable tag for eval resume; includes file-set signature when multiple JSONL."""
    paths = resolve_jsonl_source_paths(train_source)
    base = os.path.abspath(os.path.expanduser(train_source))
    if len(paths) == 1:
        return f"auto_split:{paths[0]}:{train_ratio}"
    sig = hashlib.sha1("\n".join(paths).encode("utf-8")).hexdigest()[:16]
    return f"auto_split:multi:{base}:{sig}:{train_ratio}"
