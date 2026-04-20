import json
import os
from typing import Any, Dict, Iterator, List

INPUT_EXCLUDE_KEYS = {"record_id", "command", "reason", "accuracy"}


def flatten_input_fields(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        ordered_keys = []
        if "message" in value:
            ordered_keys.append("message")
        ordered_keys.extend(k for k in sorted(value.keys()) if k != "message")
        return " | ".join(f"{k}={value[k]}" for k in ordered_keys)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def stable_json_dumps_message_first(value: Any) -> str:
    """
    Deterministic JSON string.
    If dict and has 'message', render it first, then remaining keys sorted.
    """
    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    ordered: Dict[str, Any] = {}
    if "message" in value:
        ordered["message"] = value.get("message")
    for k in sorted(value.keys()):
        if k == "message":
            continue
        ordered[k] = value.get(k)
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def extract_input_fields(record: Any) -> Any:
    if isinstance(record, dict):
        return {k: v for k, v in record.items() if k not in INPUT_EXCLUDE_KEYS}
    return record

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_jsonl_source_paths(source_path: str) -> List[str]:
    """
    Single JSONL file, or every ``*.jsonl`` under ``source_path`` (recursive).
    Paths are absolute, sorted for stable merge order.
    """
    source_path = os.path.abspath(os.path.expanduser(source_path))
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"JSONL source not found: {source_path!r}")
    if os.path.isfile(source_path):
        return [source_path]
    if os.path.isdir(source_path):
        found: List[str] = []
        for root, _dirs, names in os.walk(source_path):
            for name in sorted(names):
                if name.endswith(".jsonl"):
                    found.append(os.path.join(root, name))
        found.sort()
        if not found:
            raise FileNotFoundError(f"no .jsonl files under directory: {source_path!r}")
        return found
    raise ValueError(f"not a file or directory: {source_path!r}")


def iter_jsonl_source(source_path: str) -> Iterator[Any]:
    """Yield JSON objects from one file or merged ``*.jsonl`` under a directory."""
    for path in resolve_jsonl_source_paths(source_path):
        yield from read_jsonl(path)


def count_jsonl_lines_in_source(source_path: str) -> int:
    n = 0
    for path in resolve_jsonl_source_paths(source_path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    return n


def jsonl_source_description(source_path: str) -> str:
    paths = resolve_jsonl_source_paths(source_path)
    if len(paths) == 1:
        return paths[0]
    root = os.path.abspath(os.path.expanduser(source_path))
    return f"{root} ({len(paths)} jsonl files)"


def write_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
