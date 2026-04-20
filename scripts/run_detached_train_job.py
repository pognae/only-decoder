import argparse
import json
import os
import subprocess
import sys
import time


def _safe_write_json(path: str, obj: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_path", required=True)
    ap.add_argument("--log_path", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--command", required=True, help="shell command string to execute")
    args = ap.parse_args()

    meta_path = os.path.abspath(args.meta_path)
    log_path = os.path.abspath(args.log_path)
    workdir = os.path.abspath(args.workdir)
    cmd = str(args.command)

    # Load initial meta (created by API).
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        meta = {}

    meta["status"] = "running"
    meta["started_at"] = meta.get("started_at") or time.time()
    meta["pid"] = os.getpid()
    meta["error"] = None
    _safe_write_json(meta_path, meta)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    rc = None
    try:
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"[runner] pid={os.getpid()}\n")
            logf.write(f"[runner] cmd={cmd}\n")
            logf.write(f"[runner] workdir={workdir}\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                shell=True,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
            rc = proc.wait()
    except Exception as e:
        meta["status"] = "failed"
        meta["finished_at"] = time.time()
        meta["exit_code"] = None
        meta["error"] = f"runner_exception:{repr(e)}"
        _safe_write_json(meta_path, meta)
        return 1

    meta["finished_at"] = time.time()
    meta["exit_code"] = int(rc) if rc is not None else None
    meta["status"] = "succeeded" if rc == 0 else "failed"
    meta["error"] = None if rc == 0 else f"exit_code={rc}"
    _safe_write_json(meta_path, meta)
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())

