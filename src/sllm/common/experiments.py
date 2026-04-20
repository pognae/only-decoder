import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


_RUN_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_RUN_ID = "__default__"


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: str
    model_dir: str
    tokenizer_dir: str


def sanitize_run_id(run_id: str) -> str:
    text = (run_id or "").strip()
    if not text:
        raise ValueError("run_id is empty")
    text = _RUN_ID_RE.sub("_", text)
    text = text.strip("._-")
    if not text:
        raise ValueError("run_id is invalid")
    return text[:80]


def make_timestamp_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def experiments_root_from_env(default_root: str = "artifacts/experiments") -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get("SLLM_EXPERIMENTS_DIR", default_root)))


def build_run_paths(run_id: str, *, root_dir: Optional[str] = None) -> RunPaths:
    rid = sanitize_run_id(run_id)
    root = os.path.abspath(os.path.expanduser(root_dir or experiments_root_from_env()))
    run_dir = os.path.join(root, rid)
    return RunPaths(
        run_id=rid,
        run_dir=run_dir,
        model_dir=os.path.join(run_dir, "model"),
        tokenizer_dir=os.path.join(run_dir, "tokenizer"),
    )


def list_run_ids(*, root_dir: Optional[str] = None) -> List[str]:
    root = os.path.abspath(os.path.expanduser(root_dir or experiments_root_from_env()))
    if not os.path.isdir(root):
        return []
    rows = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        model_dir = os.path.join(path, "model")
        marker = os.path.join(model_dir, "training_result.json")
        score = os.path.getmtime(marker) if os.path.exists(marker) else os.path.getmtime(path)
        rows.append((score, name))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in rows]
