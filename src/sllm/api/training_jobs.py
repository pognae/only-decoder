import os
import subprocess
import time
import uuid
import json
import shlex
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class TrainJob:
    job_id: str
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    status: str  # queued | running | succeeded | failed
    command: str
    input_path: str
    run_id: str
    log_path: str
    pid: Optional[int]
    exit_code: Optional[int]
    error: Optional[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


import threading

_lock = threading.Lock()
_jobs: Dict[str, TrainJob] = {}
_REPO_ROOT: Optional[str] = None


def _job_meta_path(repo_root: str, job_id: str) -> str:
    return os.path.join(repo_root, "artifacts", "jobs", f"train_{job_id}.json")


def _safe_write_json(path: str, obj: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _persist_job(repo_root: str, job: TrainJob) -> None:
    try:
        _safe_write_json(_job_meta_path(repo_root, job.job_id), job.to_dict())
    except Exception:
        return


def _pid_is_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _configure_repo_root(repo_root: str) -> None:
    global _REPO_ROOT
    _REPO_ROOT = os.path.abspath(repo_root)


def _read_job_meta(repo_root: str, job_id: str) -> Optional[dict]:
    path = _job_meta_path(repo_root, job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _refresh_job_from_disk(job_id: str) -> Optional[TrainJob]:
    if not _REPO_ROOT:
        return None
    data = _read_job_meta(_REPO_ROOT, job_id)
    if not data:
        return None
    try:
        job = TrainJob(**data)
    except Exception:
        return None
    with _lock:
        _jobs[job_id] = job
    return job


def load_jobs_from_disk(repo_root: str) -> dict:
    """
    Restore job metadata from artifacts/jobs/train_*.json.
    If a job was running and its pid is no longer alive, mark it failed as stale.
    """
    jobs_dir = os.path.join(repo_root, "artifacts", "jobs")
    loaded = 0
    stale_failed = 0
    if not os.path.isdir(jobs_dir):
        return {"loaded": 0, "stale_failed": 0, "jobs_dir": jobs_dir}

    rows = []
    for name in sorted(os.listdir(jobs_dir)):
        if not (name.startswith("train_") and name.endswith(".json")):
            continue
        path = os.path.join(jobs_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
        except Exception:
            continue
        rows.append(data)

    with _lock:
        for data in rows:
            try:
                job = TrainJob(**data)
            except Exception:
                continue
            # Repair status for previously-running jobs after server restart.
            if job.status in {"queued", "running"}:
                if job.pid and _pid_is_alive(int(job.pid)):
                    job.status = "running"
                else:
                    job.status = "failed"
                    job.finished_at = job.finished_at or time.time()
                    job.error = job.error or "stale_job_after_server_restart"
                    stale_failed += 1
            _jobs[job.job_id] = job
            loaded += 1

    _configure_repo_root(repo_root)
    return {"loaded": loaded, "stale_failed": stale_failed, "jobs_dir": jobs_dir}


def get_job(job_id: str) -> Optional[TrainJob]:
    # Refresh from disk to reflect runner updates even while server is running.
    _refresh_job_from_disk(job_id)
    with _lock:
        return _jobs.get(job_id)


def list_jobs(limit: int = 20) -> list[TrainJob]:
    # Best-effort refresh (cheap): keep memory snapshot; disk refresh happens on get_job/status endpoints.
    with _lock:
        items = list(_jobs.values())
    items.sort(key=lambda j: j.created_at, reverse=True)
    return items[:limit]


def has_active_job_for_run_id(run_id: str) -> bool:
    rid = (run_id or "").strip()
    if not rid:
        return False
    with _lock:
        for job in _jobs.values():
            if str(getattr(job, "run_id", "")) != rid:
                continue
            if job.status in {"queued", "running"}:
                return True
    return False


def purge_jobs_for_run_id(run_id: str, *, remove_logs: bool = False) -> dict:
    rid = (run_id or "").strip()
    if not rid:
        return {"removed_jobs": 0, "removed_logs": 0, "skipped_active_jobs": 0}

    removed_jobs = []
    skipped_active = 0
    with _lock:
        delete_ids = []
        for job_id, job in _jobs.items():
            if str(getattr(job, "run_id", "")) != rid:
                continue
            if job.status in {"queued", "running"}:
                skipped_active += 1
                continue
            delete_ids.append(job_id)
        for job_id in delete_ids:
            job = _jobs.pop(job_id, None)
            if job is not None:
                removed_jobs.append(job)

    removed_logs = 0
    if remove_logs:
        for job in removed_jobs:
            p = getattr(job, "log_path", None)
            if not p:
                continue
            try:
                if os.path.exists(p):
                    os.remove(p)
                    removed_logs += 1
            except Exception:
                pass
            # meta json is also a log-like artifact
            try:
                if _REPO_ROOT:
                    mp = _job_meta_path(_REPO_ROOT, job.job_id)
                    if os.path.exists(mp):
                        os.remove(mp)
            except Exception:
                pass

    return {
        "removed_jobs": len(removed_jobs),
        "removed_logs": removed_logs,
        "skipped_active_jobs": skipped_active,
    }


def start_training_job(
    *,
    repo_root: str,
    input_path: str,
    run_id: str,
    python_path: str = ".venv/bin/python",
    scripts_train_path: str = "scripts/train.py",
    caffeinate: bool = True,
) -> TrainJob:
    job_id = uuid.uuid4().hex[:12]
    log_path = os.path.join(repo_root, "artifacts", "jobs", f"train_{job_id}.log")
    meta_path = _job_meta_path(repo_root, job_id)

    # Required by user request: run via caffeinate -dimsu ... scripts/train.py --input ...
    # Quote user/path inputs so spaces/special chars in uploaded filenames do not break execution.
    cmd = " ".join(
        [
            shlex.quote(str(python_path)),
            shlex.quote(str(scripts_train_path)),
            "--input",
            shlex.quote(str(input_path)),
            "--run_id",
            shlex.quote(str(run_id)),
        ]
    )
    if caffeinate and str(__import__("platform").system()).lower() == "darwin":
        cmd = f"caffeinate -dimsu {cmd}"

    job = TrainJob(
        job_id=job_id,
        created_at=time.time(),
        started_at=None,
        finished_at=None,
        status="queued",
        command=cmd,
        input_path=input_path,
        run_id=run_id,
        log_path=log_path,
        pid=None,
        exit_code=None,
        error=None,
    )

    with _lock:
        _jobs[job_id] = job
    _persist_job(repo_root, job)
    _configure_repo_root(repo_root)

    # Detach runner so training continues even if the API server exits.
    runner_cmd = [
        "python3",
        os.path.join(repo_root, "scripts", "run_detached_train_job.py"),
        "--meta_path",
        meta_path,
        "--log_path",
        log_path,
        "--workdir",
        repo_root,
        "--command",
        cmd,
    ]
    kwargs = {
        "cwd": repo_root,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "text": True,
    }
    system = str(__import__("platform").system()).lower()
    if system == "windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(runner_cmd, **kwargs)
        with _lock:
            job.pid = int(proc.pid) if proc.pid else None
        _persist_job(repo_root, job)
    except Exception as e:
        with _lock:
            job.status = "failed"
            job.finished_at = time.time()
            job.exit_code = None
            job.error = f"spawn_failed:{repr(e)}"
        _persist_job(repo_root, job)
    return job
