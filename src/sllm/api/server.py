import os
import subprocess
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sllm.common.commands import DEFAULT_COMMANDS, ensure_commands_file, get_commands_path, load_commands, save_commands
from sllm.common.experiments import DEFAULT_RUN_ID, build_run_paths, experiments_root_from_env, list_run_ids, make_timestamp_run_id
from sllm.common.io import write_json
from sllm.common.prompting import FAIL_SAFE_REASON
from sllm.infer.predict import LegacyActionSLLM
from sllm.api.dashboard import build_dashboard_html
from sllm.api.results import build_results_html
from sllm.api.training_jobs import (
    get_job,
    has_active_job_for_run_id,
    list_jobs,
    load_jobs_from_disk,
    purge_jobs_for_run_id,
    start_training_job,
)
from sllm.train.finalize_run import finalize_run as finalize_run_impl

MODEL_DIR = os.environ.get("SLLM_MODEL_DIR", "artifacts/model_dev")
TOKENIZER_DIR = os.environ.get("SLLM_TOKENIZER_DIR", MODEL_DIR)
COMMANDS_PATH = os.environ.get("SLLM_COMMANDS_FILE", get_commands_path(MODEL_DIR))
EXPERIMENTS_ROOT = experiments_root_from_env()

app = FastAPI(title="Legacy Action SLLM API")
engine = None
RUN_ENGINES: Dict[str, LegacyActionSLLM] = {}
RUN_ENGINE_LOAD_ERRORS: Dict[str, str] = {}
NO_STORE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

# Serve model directory artifacts (reports, tokenizer.json, etc.)
if os.path.isdir(MODEL_DIR):
    app.mount("/model-files", StaticFiles(directory=MODEL_DIR), name="model-files")

@app.on_event("startup")
def startup():
    global engine
    ensure_commands_file(COMMANDS_PATH)
    # Restore training jobs from disk so UI can reconnect after server restart.
    try:
        load_jobs_from_disk(_repo_root())
    except Exception:
        pass
    try:
        engine = LegacyActionSLLM(model_dir=MODEL_DIR, tokenizer_dir=TOKENIZER_DIR)
    except Exception:
        engine = None


def _repo_root() -> str:
    # src/sllm/api/server.py -> repo root (three levels up)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _sanitize_run_id_or_default(run_id: Optional[str]) -> str:
    text = (run_id or "").strip()
    if not text:
        return DEFAULT_RUN_ID
    if text == DEFAULT_RUN_ID:
        return DEFAULT_RUN_ID
    try:
        return build_run_paths(text, root_dir=EXPERIMENTS_ROOT).run_id
    except Exception:
        return DEFAULT_RUN_ID


def _resolve_run_dirs(run_id: Optional[str]) -> Tuple[str, str, str]:
    rid = _sanitize_run_id_or_default(run_id)
    if rid == DEFAULT_RUN_ID:
        return rid, MODEL_DIR, TOKENIZER_DIR
    paths = build_run_paths(rid, root_dir=EXPERIMENTS_ROOT)
    return rid, paths.model_dir, paths.tokenizer_dir


def _safe_fail_response(reason: str):
    return {
        "command": "NO_ACTION",
        "reason": reason,
        "accuracy": {"command_accuracy": None},
    }


def _resolve_train_python(root: str) -> str:
    train_python = os.environ.get("SLLM_TRAIN_PYTHON", "").strip()
    if train_python:
        return train_python
    venv_py_unix = os.path.join(root, ".venv", "bin", "python")
    venv_py_win = os.path.join(root, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv_py_unix):
        return venv_py_unix
    if os.path.isfile(venv_py_win):
        return venv_py_win
    return sys.executable or "python3"


def _parse_eval_valid_file_tag(tag: str) -> Tuple[Optional[str], Optional[float]]:
    text = (tag or "").strip()
    if not text:
        return None, None
    if not text.startswith("auto_split:"):
        return text, None

    rest = text[len("auto_split:") :]
    if rest.startswith("multi:"):
        # format: auto_split:multi:<base_source>:<sig>:<train_ratio>
        payload = rest[len("multi:") :]
        parts = payload.rsplit(":", 2)
        if len(parts) != 3:
            return None, None
        base_source, _sig, ratio_text = parts
        try:
            ratio = float(ratio_text)
        except Exception:
            ratio = None
        return (base_source or None), ratio

    # format: auto_split:<train_source>:<train_ratio>
    parts = rest.rsplit(":", 1)
    if len(parts) != 2:
        return (rest or None), None
    train_source, ratio_text = parts
    try:
        ratio = float(ratio_text)
    except Exception:
        ratio = None
    return (train_source or None), ratio


def _resolve_run_input_spec(run_id: str, model_dir: str) -> Dict[str, Optional[float] | Optional[str]]:
    root = _repo_root()
    candidates = []
    train_ratio_from_eval: Optional[float] = None

    run_meta = _read_json(os.path.join(model_dir, "run_meta.json")) or {}
    meta_src = (run_meta.get("input_source") or "").strip() if isinstance(run_meta, dict) else ""
    if meta_src:
        candidates.append(meta_src)
        if not os.path.isabs(meta_src):
            candidates.append(os.path.abspath(os.path.join(root, meta_src)))

    quality = _read_json(os.path.join(model_dir, "data_quality_report.json")) or {}
    quality_src = (quality.get("input_source") or "").strip() if isinstance(quality, dict) else ""
    if quality_src:
        candidates.append(quality_src)
        if not os.path.isabs(quality_src):
            candidates.append(os.path.abspath(os.path.join(root, quality_src)))

    dist = _read_json(os.path.join(model_dir, "data_distribution_report.json")) or {}
    if isinstance(dist, dict):
        for key in ("train_file", "valid_file"):
            src = (dist.get(key) or "").strip()
            if not src:
                continue
            candidates.append(src)
            if not os.path.isabs(src):
                candidates.append(os.path.abspath(os.path.join(root, src)))

    eval_path = os.path.join(model_dir, "eval_results.jsonl")
    if os.path.exists(eval_path):
        try:
            import json as _json

            with open(eval_path, "r", encoding="utf-8") as f:
                for _idx, line in enumerate(f):
                    if _idx > 300:
                        break
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        row = _json.loads(text)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    tag = (row.get("valid_file") or "").strip()
                    if not tag:
                        continue
                    parsed_source, parsed_ratio = _parse_eval_valid_file_tag(tag)
                    if parsed_source:
                        candidates.append(parsed_source)
                        if not os.path.isabs(parsed_source):
                            candidates.append(os.path.abspath(os.path.join(root, parsed_source)))
                    if parsed_ratio is not None and train_ratio_from_eval is None:
                        train_ratio_from_eval = parsed_ratio
                    if parsed_source:
                        break
        except Exception:
            pass

    for job in list_jobs(limit=500):
        job_rid = _sanitize_run_id_or_default(getattr(job, "run_id", ""))
        if job_rid != run_id:
            continue
        ip = (getattr(job, "input_path", "") or "").strip()
        if not ip:
            continue
        candidates.append(ip)
        candidates.append(os.path.abspath(os.path.join(root, ip)))

    if run_id != DEFAULT_RUN_ID:
        samples_root = os.path.join(root, "data", "samples")
        run_dir = os.path.join(samples_root, run_id)
        if os.path.isdir(run_dir):
            for name in sorted(os.listdir(run_dir), reverse=True):
                if name.lower().endswith(".jsonl"):
                    candidates.append(os.path.join(run_dir, name))
        if os.path.isdir(samples_root):
            prefix = f"{run_id}_"
            for name in sorted(os.listdir(samples_root), reverse=True):
                if not name.startswith(prefix) or not name.lower().endswith(".jsonl"):
                    continue
                candidates.append(os.path.join(samples_root, name))

    seen = set()
    for c in candidates:
        p = os.path.abspath(str(c))
        if p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            return {"input_source": p, "auto_split_train_ratio": train_ratio_from_eval}
        if os.path.isdir(p):
            return {"input_source": p, "auto_split_train_ratio": train_ratio_from_eval}
    return {"input_source": None, "auto_split_train_ratio": train_ratio_from_eval}


def _resolve_run_input_source(run_id: str, model_dir: str) -> Optional[str]:
    spec = _resolve_run_input_spec(run_id, model_dir)
    source = spec.get("input_source")
    return str(source) if source else None


def _run_cmd_capture(cmd: list[str], *, cwd: str, tail_chars: int = 4000):
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "")[-tail_chars:]
        err = (proc.stderr or "")[-tail_chars:]
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout_tail": out,
            "stderr_tail": err,
        }
    except Exception as e:
        return {
            "ok": False,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "exception": repr(e),
        }


def _get_engine_for_run(run_id: Optional[str]) -> Optional[LegacyActionSLLM]:
    global engine
    rid, model_dir, tokenizer_dir = _resolve_run_dirs(run_id)

    if rid == DEFAULT_RUN_ID:
        if engine is not None:
            return engine
        if os.path.isdir(model_dir):
            try:
                engine = LegacyActionSLLM(model_dir=model_dir, tokenizer_dir=tokenizer_dir)
                RUN_ENGINE_LOAD_ERRORS.pop(rid, None)
                return engine
            except Exception as e:
                RUN_ENGINE_LOAD_ERRORS[rid] = repr(e)
                return None
        return None

    cached = RUN_ENGINES.get(rid)
    if cached is not None:
        return cached
    if not os.path.isdir(model_dir):
        return None
    tok_dir = tokenizer_dir if os.path.isdir(tokenizer_dir) else model_dir
    try:
        loaded = LegacyActionSLLM(model_dir=model_dir, tokenizer_dir=tok_dir)
        RUN_ENGINES[rid] = loaded
        RUN_ENGINE_LOAD_ERRORS.pop(rid, None)
        return loaded
    except Exception as e:
        RUN_ENGINE_LOAD_ERRORS[rid] = repr(e)
        return None


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _fmt_mtime(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _fmt_ts(epoch: Optional[float]) -> str:
    try:
        if epoch is None:
            return "-"
        return datetime.fromtimestamp(float(epoch)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _collect_files_inventory(root_dir: str, *, max_items: int = 2000):
    if not os.path.isdir(root_dir):
        return {"root_dir": os.path.abspath(root_dir), "exists": False, "truncated": False, "count": 0, "items": []}
    rows = []
    for cur_root, _dirs, files in os.walk(root_dir):
        files = sorted(files)
        for name in files:
            full = os.path.join(cur_root, name)
            rel = os.path.relpath(full, root_dir)
            try:
                size = os.path.getsize(full)
            except Exception:
                size = None
            rows.append(
                {
                    "path": rel,
                    "size_bytes": size,
                    "mtime": _fmt_mtime(full),
                }
            )
            if len(rows) >= max_items:
                return {
                    "root_dir": os.path.abspath(root_dir),
                    "exists": True,
                    "truncated": True,
                    "count": len(rows),
                    "items": rows,
                }
    return {
        "root_dir": os.path.abspath(root_dir),
        "exists": True,
        "truncated": False,
        "count": len(rows),
        "items": rows,
    }


def _releases_dir_for_target(target_model_dir: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(target_model_dir)), "releases"))


def _read_current_promoted_run_id(target_model_dir: str = MODEL_DIR) -> Optional[str]:
    marker_path = os.path.join(_releases_dir_for_target(target_model_dir), "current_run_id.txt")
    if not os.path.exists(marker_path):
        return None
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            text = (f.read() or "").strip()
    except Exception:
        return None
    if not text:
        return None
    return _sanitize_run_id_or_default(text)


def _reload_default_engine() -> Optional[str]:
    global engine
    if not os.path.isdir(MODEL_DIR):
        engine = None
        return f"default model_dir not found: {MODEL_DIR}"
    tok_dir = TOKENIZER_DIR if os.path.isdir(TOKENIZER_DIR) else MODEL_DIR
    try:
        engine = LegacyActionSLLM(model_dir=MODEL_DIR, tokenizer_dir=tok_dir)
        return None
    except Exception as e:
        engine = None
        return str(e)


def _review_file_path_for_run(run_id: Optional[str]) -> str:
    _rid, model_dir, _tok = _resolve_run_dirs(run_id)
    return os.path.join(model_dir, "review_approval.json")


def _read_review_approval(run_id: Optional[str]):
    path = _review_file_path_for_run(run_id)
    data = _read_json(path) or {}
    return {"path": os.path.abspath(path), "data": data}


def _list_runs_for_ui():
    current_run_id = _read_current_promoted_run_id()
    rows = []
    row_by_run_id: Dict[str, dict] = {}
    score_by_run_id: Dict[str, float] = {}

    def _set_score(rid: str, score: float):
        prev = score_by_run_id.get(rid)
        if prev is None or score > prev:
            score_by_run_id[rid] = score

    def _upsert_row(rid: str):
        if rid in row_by_run_id:
            row_by_run_id[rid]["is_current"] = current_run_id == rid
            return
        _rid, model_dir, _ = _resolve_run_dirs(rid)
        metrics = _read_json(os.path.join(model_dir, "metrics.json")) or {}
        quality = _read_json(os.path.join(model_dir, "data_quality_report.json")) or {}
        row_by_run_id[rid] = {
            "run_id": rid,
            "label": rid if rid != DEFAULT_RUN_ID else "default",
            "model_dir": os.path.abspath(model_dir),
            "exists": os.path.isdir(model_dir),
            "command_accuracy": metrics.get("command_accuracy"),
            "quality_status": quality.get("status"),
            "is_current": current_run_id == rid,
        }

    default_metrics = _read_json(os.path.join(MODEL_DIR, "metrics.json")) or {}
    default_quality = _read_json(os.path.join(MODEL_DIR, "data_quality_report.json")) or {}
    row_by_run_id[DEFAULT_RUN_ID] = {
        "run_id": DEFAULT_RUN_ID,
        "label": "default",
        "model_dir": os.path.abspath(MODEL_DIR),
        "exists": os.path.isdir(MODEL_DIR),
        "command_accuracy": default_metrics.get("command_accuracy"),
        "quality_status": default_quality.get("status"),
        "is_current": current_run_id == DEFAULT_RUN_ID,
    }
    if os.path.isdir(MODEL_DIR):
        marker = os.path.join(MODEL_DIR, "training_result.json")
        _set_score(DEFAULT_RUN_ID, os.path.getmtime(marker) if os.path.exists(marker) else os.path.getmtime(MODEL_DIR))

    for rid in list_run_ids(root_dir=EXPERIMENTS_ROOT):
        if rid == DEFAULT_RUN_ID:
            continue
        _upsert_row(rid)
        _, model_dir, _ = _resolve_run_dirs(rid)
        if os.path.isdir(model_dir):
            marker = os.path.join(model_dir, "training_result.json")
            _set_score(rid, os.path.getmtime(marker) if os.path.exists(marker) else os.path.getmtime(model_dir))

    # Also expose run_ids that exist as uploaded dataset buckets, even if training failed
    # or no model artifact has been produced yet.
    samples_root = os.path.join(_repo_root(), "data", "samples")
    if os.path.isdir(samples_root):
        for name in os.listdir(samples_root):
            path = os.path.join(samples_root, name)
            if os.path.isdir(path):
                try:
                    rid = build_run_paths(name, root_dir=EXPERIMENTS_ROOT).run_id
                except Exception:
                    continue
                if rid == DEFAULT_RUN_ID:
                    continue
                _upsert_row(rid)
                _set_score(rid, os.path.getmtime(path))
                continue

            # Backward-compat: older uploads may be saved as data/samples/<run_id>_<filename>.jsonl
            if os.path.isfile(path) and name.lower().endswith(".jsonl") and "_" in name:
                prefix = name.split("_", 1)[0].strip()
                if not prefix:
                    continue
                try:
                    rid = build_run_paths(prefix, root_dir=EXPERIMENTS_ROOT).run_id
                except Exception:
                    continue
                if rid == DEFAULT_RUN_ID:
                    continue
                _upsert_row(rid)
                _set_score(rid, os.path.getmtime(path))

    for job in list_jobs(limit=200):
        rid = _sanitize_run_id_or_default(getattr(job, "run_id", ""))
        if not rid or rid == DEFAULT_RUN_ID:
            continue
        _upsert_row(rid)
        try:
            _set_score(rid, float(getattr(job, "created_at", 0.0) or 0.0))
        except Exception:
            pass

    if current_run_id and current_run_id != DEFAULT_RUN_ID:
        _upsert_row(current_run_id)

    rows.append(row_by_run_id[DEFAULT_RUN_ID])
    others = [x for k, x in row_by_run_id.items() if k != DEFAULT_RUN_ID]
    others.sort(key=lambda x: score_by_run_id.get(x["run_id"], 0.0), reverse=True)
    rows.extend(others)
    return rows

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard")
def dashboard():
    return build_dashboard_html(
        model_dir=MODEL_DIR,
        tokenizer_dir=TOKENIZER_DIR,
        current_run_id=_read_current_promoted_run_id(),
        experiments_root=EXPERIMENTS_ROOT,
    )

@app.get("/results")
def results(
    run_id: str = Query(DEFAULT_RUN_ID),
    compare_run_id: Optional[str] = Query(None),
):
    rid, model_dir, _ = _resolve_run_dirs(run_id)
    compare_rid = None
    compare_model_dir = None
    if compare_run_id:
        compare_rid, compare_model_dir, _ = _resolve_run_dirs(compare_run_id)
    return build_results_html(
        model_dir=model_dir,
        run_id=rid,
        compare_model_dir=compare_model_dir,
        compare_run_id=compare_rid,
        available_runs=_list_runs_for_ui(),
        current_promoted_run_id=_read_current_promoted_run_id(),
    )

@app.get("/playground")
def playground():
    html = f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Infer Playground</title>
      <style>
        :root {{
          --bg: #f6f7fb;
          --card: #ffffff;
          --text: #121826;
          --muted: #5f6b7a;
          --border: rgba(18,24,38,.12);
          --accent: #2563eb;
        }}
        body {{
          margin: 0;
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          background: radial-gradient(1200px 600px at 15% 10%, rgba(37,99,235,.14), transparent 55%),
                      radial-gradient(900px 600px at 75% 0%, rgba(16,185,129,.10), transparent 55%),
                      var(--bg);
          color: var(--text);
        }}
        .wrap {{ max-width: 1100px; margin: 32px auto; padding: 0 16px; }}
        .top {{ display:flex; justify-content:space-between; align-items:baseline; gap:16px; }}
        h1 {{ margin: 0; font-size: 20px; }}
        .nav a {{ margin-left: 10px; }}
        .card {{
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 14px;
          box-shadow: 0 10px 30px rgba(18,24,38,.06);
          margin-top: 12px;
        }}
        .card-title {{ font-weight: 600; margin-bottom: 10px; }}
        textarea {{
          width: 100%;
          min-height: 180px;
          box-sizing: border-box;
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid var(--border);
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          font-size: 12.5px;
          background: rgba(18,24,38,.02);
        }}
        button {{
          background: var(--accent);
          color: #fff;
          border: none;
          padding: 10px 12px;
          border-radius: 10px;
          cursor: pointer;
          font-weight: 600;
        }}
        button.secondary {{
          background: rgba(18,24,38,.08);
          color: var(--text);
          border: 1px solid var(--border);
          font-weight: 600;
        }}
        .row {{ display:flex; gap:10px; align-items:center; flex-wrap: wrap; }}
        .muted {{ color: var(--muted); font-size: 13px; }}
        pre {{
          white-space: pre-wrap;
          word-break: break-word;
          background: rgba(18,24,38,.03);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 12px;
          margin: 0;
          font-size: 12.5px;
        }}
        code {{ background: rgba(18,24,38,.06); padding: 2px 6px; border-radius: 6px; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="top">
          <h1>추론 확인(Playground)</h1>
          <div class="nav">
            <a href="/dashboard">dashboard</a>
            <a href="/results">results</a>
            <a href="/wizard">wizard</a>
            <a href="/release">release</a>
            <a href="/train">train</a>
            <a href="/commands">commands</a>
          </div>
        </div>

        <div class="card">
          <div class="card-title">실험 선택(RUN_ID)</div>
          <div class="row">
            <select id="runId"></select>
            <span class="muted" id="runMeta"></span>
          </div>
        </div>

        <div class="card">
          <div class="card-title">입력(질문/로그/JSON)</div>
          <div class="muted" style="margin-bottom:8px">
            입력이 JSON으로 파싱되면 객체로 <code>/infer</code>에 전송하고, 아니면 문자열 그대로 전송합니다.
          </div>
          <textarea id="input"></textarea>
          <div class="row" style="margin-top:10px">
            <button id="run">추론 실행</button>
            <button class="secondary" id="fillJson">예시 JSON</button>
            <button class="secondary" id="fillText">예시 텍스트</button>
            <span class="muted" id="status"></span>
          </div>
        </div>

        <div class="card">
          <div class="card-title">결과(JSON)</div>
          <pre id="output">{{}}</pre>
        </div>

        <div class="card">
          <div class="card-title">전송 형태(curl 참고)</div>
          <pre id="curl">-</pre>
        </div>
      </div>

      <script>
        const input = document.getElementById('input');
        const output = document.getElementById('output');
        const status = document.getElementById('status');
        const curl = document.getElementById('curl');
        const runId = document.getElementById('runId');
        const runMeta = document.getElementById('runMeta');
        const preferredRunId = new URLSearchParams(location.search).get('run_id');

        async function loadRuns() {{
          const r = await fetch('/runs/list');
          const j = await r.json();
          const allRows = j.runs || [];
          // Playground는 실제로 테스트 가능한 RUN 위주로 표시한다.
          // (model artifact가 없는 exists=false RUN은 기본적으로 숨김)
          const rows = allRows.filter(x => x.run_id === '__default__' || !!x.exists || !!x.is_current);
          runId.innerHTML = '';
          for (const x of rows) {{
            const o = document.createElement('option');
            o.value = x.run_id;
            const live = x.is_current ? ' [LIVE]' : '';
            o.textContent = `${{x.label}}${{live}} (acc=${{x.command_accuracy ?? '-'}}, quality=${{x.quality_status ?? '-'}})`;
            runId.appendChild(o);
          }}
          const existsPreferred = preferredRunId && rows.some(x => x.run_id === preferredRunId);
          runId.value = (existsPreferred ? preferredRunId : (j.current_run_id || j.default_run_id || '__default__'));
          updateRunMeta(rows);
          runId.addEventListener('change', () => updateRunMeta(rows));
        }}

        function updateRunMeta(rows) {{
          const rid = runId.value;
          const row = rows.find(x => x.run_id === rid);
          if (!row) {{
            runMeta.textContent = '';
            return;
          }}
          runMeta.textContent = `model_dir=${{row.model_dir}}`;
        }}

        function buildPayload(text) {{
          const trimmed = text.trim();
          if (!trimmed) return null;
          try {{
            return JSON.parse(trimmed);
          }} catch (e) {{
            return trimmed;
          }}
        }}

        function escapeForCurl(s) {{
          return s.replace(/\\\\/g, '\\\\\\\\').replace(/"/g, '\\\\\"');
        }}

        function buildCurl(payload, selectedRunId) {{
          const body = JSON.stringify(payload);
          return `curl -X POST "http://127.0.0.1:8000/infer?run_id=${{encodeURIComponent(selectedRunId)}}" -H "Content-Type: application/json" -d "${{escapeForCurl(body)}}"`;
        }}

        async function runInfer() {{
          const text = input.value;
          const payload = buildPayload(text);
          const selectedRunId = runId.value || '__default__';
          if (payload === null) {{
            alert('입력을 넣어주세요');
            return;
          }}
          status.textContent = 'running...';
          output.textContent = '';
          curl.textContent = buildCurl(payload, selectedRunId);
          try {{
            const r = await fetch(`/infer?run_id=${{encodeURIComponent(selectedRunId)}}`, {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify(payload),
            }});
            const t = await r.text();
            try {{
              output.textContent = JSON.stringify(JSON.parse(t), null, 2);
            }} catch (e) {{
              output.textContent = t;
            }}
            status.textContent = `done (HTTP ${{r.status}})`;
          }} catch (e) {{
            status.textContent = 'error';
            output.textContent = String(e);
          }}
        }}

        document.getElementById('run').addEventListener('click', runInfer);
        input.addEventListener('keydown', (e) => {{
          if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {{
            runInfer();
          }}
        }});

        document.getElementById('fillJson').addEventListener('click', () => {{
          input.value = JSON.stringify({{
            "domain": "oms",
            "system": "OMS",
            "message": "User with id admin access to ip 10.0.0.225 and checked the/dw/main/mainPageSD",
            "create_date": "2026-04-09 11:04:19",
            "state": "PAGE"
          }}, null, 2);
        }});

        document.getElementById('fillText').addEventListener('click', () => {{
          input.value = "picking location not found (warehouse_id=WH01)";
        }});

        // default example
        document.getElementById('fillJson').click();
        loadRuns();
      </script>
    </body>
    </html>
    """
    # This endpoint is an f-string; normalize any doubled braces that break CSS/JS.
    html = html.replace("{{", "{").replace("}}", "}")
    return HTMLResponse(html, headers=NO_STORE_HEADERS)


@app.get("/playgroud")
def playground_typo_alias():
    return RedirectResponse(url="/playground")


@app.get("/runs/list")
def runs_list():
    return {
        "default_run_id": DEFAULT_RUN_ID,
        "current_run_id": _read_current_promoted_run_id(),
        "runs": _list_runs_for_ui(),
    }


@app.get("/runs/review")
def runs_review(run_id: str = Query(DEFAULT_RUN_ID)):
    rid = _sanitize_run_id_or_default(run_id)
    review = _read_review_approval(rid)
    return {
        "run_id": rid,
        "review_file": review["path"],
        "review": review["data"],
    }


@app.post("/runs/review/approve")
def runs_review_approve(
    run_id: str = Body(..., embed=True),
    reviewer: Optional[str] = Body(None, embed=True),
    note: Optional[str] = Body(None, embed=True),
    approved: bool = Body(True, embed=True),
):
    rid, model_dir, _tok = _resolve_run_dirs(run_id)
    if not os.path.isdir(model_dir):
        return {
            "ok": False,
            "run_id": rid,
            "error": f"model_dir not found: {model_dir}",
        }
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "approved": bool(approved),
        "reviewer": (reviewer or "").strip() or None,
        "note": (note or "").strip() or None,
        "approved_at_utc": ts,
        "run_id": rid,
    }
    path = os.path.join(model_dir, "review_approval.json")
    write_json(path, payload)
    return {
        "ok": True,
        "run_id": rid,
        "review_file": os.path.abspath(path),
        "review": payload,
    }


@app.get("/runs/current")
def runs_current():
    return {
        "current_run_id": _read_current_promoted_run_id(),
        "target_model_dir": os.path.abspath(MODEL_DIR),
        "target_tokenizer_dir": os.path.abspath(TOKENIZER_DIR),
        "releases_dir": _releases_dir_for_target(MODEL_DIR),
    }


@app.get("/runs/detail")
def runs_detail(run_id: str = Query(DEFAULT_RUN_ID)):
    rid = _sanitize_run_id_or_default(run_id)
    _rid, model_dir, tokenizer_dir = _resolve_run_dirs(rid)
    current = _read_current_promoted_run_id()
    is_current = bool(current and current == rid)

    run_meta = _read_json(os.path.join(model_dir, "run_meta.json")) or {}
    config = _read_json(os.path.join(model_dir, "config.json")) or {}
    training_result = _read_json(os.path.join(model_dir, "training_result.json")) or {}
    train_metrics = _read_json(os.path.join(model_dir, "train_metrics.json")) or {}
    metrics = _read_json(os.path.join(model_dir, "metrics.json")) or {}
    quality = _read_json(os.path.join(model_dir, "data_quality_report.json")) or {}
    reason_cov = _read_json(os.path.join(model_dir, "reason_coverage_report.json")) or {}
    distribution = _read_json(os.path.join(model_dir, "data_distribution_report.json")) or {}
    review = _read_json(os.path.join(model_dir, "review_approval.json")) or {}
    promotion_record = _read_json(os.path.join(model_dir, "promotion_record.json")) or {}

    jobs = []
    for job in list_jobs(limit=500):
        job_rid = _sanitize_run_id_or_default(getattr(job, "run_id", ""))
        if job_rid == rid:
            jobs.append(job.to_dict())

    files = {
        "training_result.json": {"path": os.path.join(model_dir, "training_result.json"), "exists": os.path.exists(os.path.join(model_dir, "training_result.json"))},
        "train_metrics.json": {"path": os.path.join(model_dir, "train_metrics.json"), "exists": os.path.exists(os.path.join(model_dir, "train_metrics.json"))},
        "metrics.json": {"path": os.path.join(model_dir, "metrics.json"), "exists": os.path.exists(os.path.join(model_dir, "metrics.json"))},
        "data_quality_report.json": {"path": os.path.join(model_dir, "data_quality_report.json"), "exists": os.path.exists(os.path.join(model_dir, "data_quality_report.json"))},
        "reason_coverage_report.json": {"path": os.path.join(model_dir, "reason_coverage_report.json"), "exists": os.path.exists(os.path.join(model_dir, "reason_coverage_report.json"))},
        "data_distribution_report.json": {"path": os.path.join(model_dir, "data_distribution_report.json"), "exists": os.path.exists(os.path.join(model_dir, "data_distribution_report.json"))},
        "review_approval.json": {"path": os.path.join(model_dir, "review_approval.json"), "exists": os.path.exists(os.path.join(model_dir, "review_approval.json"))},
        "promotion_record.json": {"path": os.path.join(model_dir, "promotion_record.json"), "exists": os.path.exists(os.path.join(model_dir, "promotion_record.json"))},
        "tokenizer.json": {"path": os.path.join(tokenizer_dir, "tokenizer.json"), "exists": os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json"))},
        "model.safetensors": {"path": os.path.join(model_dir, "model.safetensors"), "exists": os.path.exists(os.path.join(model_dir, "model.safetensors"))},
    }

    return {
        "run": {
            "run_id": rid,
            "current_live_run_id": current,
            "is_current_live": is_current,
            "is_test_only_run": rid != DEFAULT_RUN_ID and not is_current,
        },
        "paths": {
            "model_dir": os.path.abspath(model_dir),
            "tokenizer_dir": os.path.abspath(tokenizer_dir),
            "model_dir_exists": os.path.isdir(model_dir),
            "tokenizer_dir_exists": os.path.isdir(tokenizer_dir),
        },
        "run_meta": run_meta,
        "summary": (training_result.get("summary") or {}) if isinstance(training_result, dict) else {},
        "metrics": metrics,
        "train_metrics": train_metrics,
        "quality": quality,
        "reason_coverage": reason_cov,
        "distribution": distribution,
        "review_approval": review,
        "promotion_record": promotion_record,
        "jobs": jobs,
        "files": files,
        "model_files_inventory": _collect_files_inventory(model_dir),
        "tokenizer_files_inventory": _collect_files_inventory(tokenizer_dir),
        "loaded_at": datetime.now().isoformat(),
    }


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _render_kv_card(title: str, rows):
    tds = "\n".join(
        f"<tr><td class='k'>{_escape_html(k)}</td><td class='v'>{_escape_html(_fmt(v))}</td></tr>"
        for k, v in rows
    )
    return f"""
    <div class="card">
      <div class="card-title">{_escape_html(title)}</div>
      <table class="kv">
        {tds}
      </table>
    </div>
    """


def _top_items(d: dict, k: int = 12):
    items = list((d or {}).items())
    try:
        items.sort(key=lambda x: x[1], reverse=True)
    except Exception:
        items.sort(key=lambda x: str(x[0]))
    shown = items[:k]
    more = max(0, len(items) - len(shown))
    return shown, more


def _render_dist_compact(dist: dict, label: str) -> str:
    if not isinstance(dist, dict) or not dist:
        return f"<div class='muted'>분포 리포트 없음 ({_escape_html(label)})</div>"
    base_rows = [
        ("rows", dist.get("rows")),
        ("labeled_rows", dist.get("labeled_rows")),
    ]
    cmd_items, cmd_more = _top_items(dist.get("command_distribution") or {}, k=12)
    sys_items, sys_more = _top_items(dist.get("system_distribution") or {}, k=12)
    dom_items, dom_more = _top_items(dist.get("domain_distribution") or {}, k=12)
    msg_items, msg_more = _top_items(dist.get("message_length_bucket_distribution") or {}, k=12)

    def _list(title: str, items, more: int) -> str:
        lis = "\n".join(
            f"<li><span class='k'>{_escape_html(k)}</span><span class='v'>{_escape_html(_fmt(v))}</span></li>"
            for k, v in items
        )
        more_html = f"<li class='more'>… 외 {more}개</li>" if more else ""
        return f"""
        <div class="dist-card">
          <div class="dist-title">{_escape_html(title)}</div>
          <ul class="dist-list">
            {lis}
            {more_html}
          </ul>
        </div>
        """

    base_kv = "".join(
        f"<tr><td class='k'>{_escape_html(k)}</td><td class='v'>{_escape_html(_fmt(v))}</td></tr>"
        for k, v in base_rows
    )
    return f"""
    <div class="card">
      <div class="card-title">데이터 분포 ({_escape_html(label)})</div>
      <table class="kv">{base_kv}</table>
      <div class="dist-grid">
        {_list("command", cmd_items, cmd_more)}
        {_list("system", sys_items, sys_more)}
        {_list("domain", dom_items, dom_more)}
        {_list("message_length_bucket", msg_items, msg_more)}
      </div>
    </div>
    """


@app.get("/runs/detail_view")
def runs_detail_view(run_id: str = Query(DEFAULT_RUN_ID)):
    rid = _sanitize_run_id_or_default(run_id)
    _rid, model_dir, tokenizer_dir = _resolve_run_dirs(rid)
    current = _read_current_promoted_run_id()
    is_current = bool(current and current == rid)

    run_meta = _read_json(os.path.join(model_dir, "run_meta.json")) or {}
    training_result = _read_json(os.path.join(model_dir, "training_result.json")) or {}
    train_metrics = _read_json(os.path.join(model_dir, "train_metrics.json")) or {}
    metrics = _read_json(os.path.join(model_dir, "metrics.json")) or {}
    quality = _read_json(os.path.join(model_dir, "data_quality_report.json")) or {}
    reason_cov = _read_json(os.path.join(model_dir, "reason_coverage_report.json")) or {}
    dist = _read_json(os.path.join(model_dir, "data_distribution_report.json")) or {}
    review = _read_json(os.path.join(model_dir, "review_approval.json")) or {}

    summary = (training_result.get("summary") or {}) if isinstance(training_result, dict) else {}
    rc = reason_cov.get("reason_coverage", {}) if isinstance(reason_cov, dict) else {}
    cs = reason_cov.get("command_shape", {}) if isinstance(reason_cov, dict) else {}
    train_dist = (dist.get("train") or {}) if isinstance(dist, dict) else {}
    valid_dist = (dist.get("valid") or {}) if isinstance(dist, dict) else {}

    dist_html_rel = os.path.join(model_dir, "data_distribution_report.html")
    dist_html_link = (
        f"<a class='link' href='/model-files/data_distribution_report.html' target='_blank'>분포 HTML 열기</a>"
        if os.path.exists(dist_html_rel) and os.path.abspath(model_dir) == os.path.abspath(MODEL_DIR)
        else ""
    )

    # checkpoints
    checkpoints = sorted(
        [x for x in os.listdir(model_dir) if x.startswith("checkpoint-")] if os.path.isdir(model_dir) else []
    )
    ckpt_html = (
        "<div class='muted'>체크포인트 없음</div>"
        if not checkpoints
        else "<ul class='ckpt'>"
        + "\n".join(f"<li>{_escape_html(c)}</li>" for c in checkpoints[-20:])
        + ("<li class='more'>…</li>" if len(checkpoints) > 20 else "")
        + "</ul>"
    )

    jobs = []
    for job in list_jobs(limit=200):
        if _sanitize_run_id_or_default(getattr(job, "run_id", "")) == rid:
            jobs.append(job)
    jobs_rows = "\n".join(
        f"<tr><td>{_escape_html(j.job_id)}</td><td>{_escape_html(j.status)}</td>"
        f"<td>{_escape_html(_fmt_ts(j.created_at))}</td><td>{_escape_html(_fmt_ts(j.started_at))}</td>"
        f"<td>{_escape_html(_fmt_ts(j.finished_at))}</td><td>{_escape_html(_fmt(j.exit_code))}</td>"
        f"<td><a class='link' href='/train/jobs/{_escape_html(j.job_id)}/log' target='_blank'>log</a></td></tr>"
        for j in jobs[:20]
    )
    jobs_card = f"""
    <div class="card">
      <div class="card-title">학습 잡 상태 (최근 20)</div>
      <table class="kv">
        <tr><th class='k'>job</th><th class='v'>status</th><th class='v'>created</th><th class='v'>started</th><th class='v'>finished</th><th class='v'>exit</th><th class='v'></th></tr>
        {jobs_rows or "<tr><td colspan='7' class='muted'>잡 이력 없음</td></tr>"}
      </table>
    </div>
    """

    header = f"""
    <div class="card">
      <div class="card-title">RUN 요약</div>
      <div class="muted">run_id=<code>{_escape_html(rid)}</code> &nbsp; current_live=<code>{_escape_html(current or '-')}</code>
      &nbsp; status={'LIVE' if is_current else 'test-only'}</div>
      <div class="muted" style="margin-top:6px;">model_dir=<code>{_escape_html(os.path.abspath(model_dir))}</code></div>
      <div class="muted">tokenizer_dir=<code>{_escape_html(os.path.abspath(tokenizer_dir))}</code></div>
      <div class="row" style="margin-top:8px; gap:8px; flex-wrap:wrap;">
        <a class="link" href="/results?run_id={_escape_html(rid)}" target="_blank">results 열기</a>
        <a class="link" href="/playground?run_id={_escape_html(rid)}" target="_blank">playground 열기</a>
        {dist_html_link}
      </div>
    </div>
    """

    cards = "\n".join(
        [
            header,
            _render_kv_card(
                "학습 상태 (training_result/metrics)",
                [
                    ("global_step", summary.get("global_step")),
                    ("epoch", summary.get("epoch")),
                    ("max_steps", summary.get("max_steps")),
                    ("train_loss", summary.get("train_loss")),
                    ("eval_loss", summary.get("eval_loss")),
                    ("command_accuracy", metrics.get("command_accuracy")),
                    ("evaluation_total", metrics.get("total")),
                ],
            ),
            _render_kv_card(
                "학습 속도 (train_metrics.json)",
                [
                    ("train_runtime", train_metrics.get("train_runtime")),
                    ("train_samples_per_second", train_metrics.get("train_samples_per_second")),
                    ("train_steps_per_second", train_metrics.get("train_steps_per_second")),
                    ("total_flos", train_metrics.get("total_flos")),
                ],
            ),
            _render_kv_card(
                "데이터 품질 게이트",
                [
                    ("status", quality.get("status")),
                    ("warnings", quality.get("warnings")),
                    ("fail_reasons", quality.get("fail_reasons")),
                ],
            ),
            _render_kv_card(
                "Reason/Command 형태 점검",
                [
                    ("total", reason_cov.get("total")),
                    ("reason_coverage_ratio", rc.get("coverage_ratio")),
                    ("reason_missing_key", rc.get("missing_key_count")),
                    ("reason_blank", rc.get("blank_count")),
                    ("reason_null", rc.get("null_count")),
                    ("command_malformed_ratio", cs.get("malformed_ratio")),
                    ("command_malformed_count", cs.get("malformed_count")),
                ],
            ),
            _render_kv_card(
                "RUN 메타데이터 (run_meta/review)",
                [
                    ("input_source", run_meta.get("input_source")),
                    ("device", run_meta.get("device")),
                    ("resume", run_meta.get("resume")),
                    ("review.approved", (review or {}).get("approved")),
                    ("review.reviewer", (review or {}).get("reviewer")),
                    ("review.approved_at_utc", (review or {}).get("approved_at_utc")),
                ],
            ),
            _render_dist_compact(train_dist, "train"),
            _render_dist_compact(valid_dist, "valid"),
            f"""
            <div class="card">
              <div class="card-title">체크포인트</div>
              {ckpt_html}
            </div>
            """,
            jobs_card,
        ]
    )

    # This endpoint returns a fragment (inserted into wizard modal).
    return HTMLResponse(cards)


@app.post("/runs/delete")
def runs_delete(
    run_id: str = Body(..., embed=True),
):
    rid = _sanitize_run_id_or_default(run_id)
    if rid == DEFAULT_RUN_ID:
        return {
            "ok": False,
            "run_id": rid,
            "error": "default run cannot be deleted",
        }

    current = _read_current_promoted_run_id()
    if current and current == rid:
        return {
            "ok": False,
            "run_id": rid,
            "error": "current LIVE run cannot be deleted; finalize another run first",
            "current_live_run_id": current,
        }

    if has_active_job_for_run_id(rid):
        return {
            "ok": False,
            "run_id": rid,
            "error": "active training job exists for this run_id (queued/running)",
        }

    rp = build_run_paths(rid, root_dir=EXPERIMENTS_ROOT)
    run_dir = os.path.abspath(rp.run_dir)
    root = os.path.abspath(EXPERIMENTS_ROOT)
    if not (run_dir.startswith(root + os.sep) or run_dir == root):
        return {
            "ok": False,
            "run_id": rid,
            "error": f"invalid run_dir: {run_dir}",
        }

    deleted = {
        "run_dir_deleted": False,
        "sample_dir_deleted": False,
        "sample_legacy_files_deleted": 0,
        "jobs": {"removed_jobs": 0, "removed_logs": 0, "skipped_active_jobs": 0},
    }

    run_dir_error = None
    if os.path.isdir(run_dir):
        try:
            shutil.rmtree(run_dir)
            deleted["run_dir_deleted"] = True
        except Exception as e:
            run_dir_error = str(e)

    samples_root = os.path.join(_repo_root(), "data", "samples")
    sample_dir = os.path.join(samples_root, rid)
    if os.path.isdir(sample_dir):
        try:
            shutil.rmtree(sample_dir)
            deleted["sample_dir_deleted"] = True
        except Exception:
            pass

    if os.path.isdir(samples_root):
        legacy_deleted = 0
        prefix = f"{rid}_"
        for name in os.listdir(samples_root):
            p = os.path.join(samples_root, name)
            if not os.path.isfile(p):
                continue
            if not name.lower().endswith(".jsonl"):
                continue
            if not name.startswith(prefix):
                continue
            try:
                os.remove(p)
                legacy_deleted += 1
            except Exception:
                pass
        deleted["sample_legacy_files_deleted"] = legacy_deleted

    deleted["jobs"] = purge_jobs_for_run_id(rid, remove_logs=True)

    RUN_ENGINES.pop(rid, None)

    if run_dir_error is not None:
        return {
            "ok": False,
            "run_id": rid,
            "error": f"run_dir delete failed: {run_dir_error}",
            "run_dir": run_dir,
            "deleted": deleted,
        }

    return {
        "ok": True,
        "run_id": rid,
        "deleted_run_dir": run_dir if deleted["run_dir_deleted"] else None,
        "model_dir": os.path.abspath(rp.model_dir),
        "tokenizer_dir": os.path.abspath(rp.tokenizer_dir),
        "current_live_run_id": current,
        "deleted": deleted,
    }


@app.post("/runs/finalize")
def runs_finalize(
    run_id: str = Body(..., embed=True),
    dry_run: bool = Body(False, embed=True),
    min_command_accuracy: Optional[float] = Body(None, embed=True),
    min_reason_coverage: float = Body(1.0, embed=True),
    max_command_malformed_ratio: float = Body(0.0, embed=True),
    require_review_approval: bool = Body(True, embed=True),
    target_model_dir: str = Body(MODEL_DIR, embed=True),
    target_tokenizer_dir: Optional[str] = Body(TOKENIZER_DIR, embed=True),
):
    rid = _sanitize_run_id_or_default(run_id)
    result = finalize_run_impl(
        run_id=rid,
        target_model_dir=target_model_dir,
        target_tokenizer_dir=target_tokenizer_dir,
        experiments_root=EXPERIMENTS_ROOT,
        default_model_dir=MODEL_DIR,
        default_tokenizer_dir=TOKENIZER_DIR,
        min_command_accuracy=min_command_accuracy,
        min_reason_coverage=min_reason_coverage,
        max_command_malformed_ratio=max_command_malformed_ratio,
        require_review_approval=require_review_approval,
        dry_run=dry_run,
    )

    target_model_abs = os.path.abspath(target_model_dir)
    default_model_abs = os.path.abspath(MODEL_DIR)
    if result.get("ok") and not dry_run and target_model_abs == default_model_abs:
        reload_error = _reload_default_engine()
        result["default_engine_reloaded"] = reload_error is None
        if reload_error is not None:
            result["default_engine_reload_error"] = reload_error
    return result


@app.post("/runs/prepare_finalize")
def runs_prepare_finalize(
    run_id: str = Body(..., embed=True),
    required_fields: str = Body("message,system", embed=True),
):
    rid, model_dir, _tok = _resolve_run_dirs(run_id)
    if not os.path.isdir(model_dir):
        return {
            "ok": False,
            "run_id": rid,
            "error": f"model_dir not found: {model_dir}",
        }

    input_spec = _resolve_run_input_spec(rid, model_dir)
    input_source = str(input_spec.get("input_source") or "").strip()
    auto_split_train_ratio = input_spec.get("auto_split_train_ratio")
    if not input_source:
        return {
            "ok": False,
            "run_id": rid,
            "model_dir": os.path.abspath(model_dir),
            "error": "input_source not found (run_meta/jobs/data samples)",
        }

    root = _repo_root()
    py = _resolve_train_python(root)

    quality_path = os.path.join(model_dir, "data_quality_report.json")
    reason_cov_path = os.path.join(model_dir, "reason_coverage_report.json")
    quality_missing_before = not os.path.exists(quality_path)
    reason_missing_before = not os.path.exists(reason_cov_path)

    steps = []
    errors = []

    # Generate data quality report if missing.
    if quality_missing_before:
        cmd = [
            py,
            "-m",
            "sllm.train.data_quality_gate",
            "--input_source",
            input_source,
            "--output_file",
            quality_path,
            "--required_fields",
            required_fields,
        ]
        rec = _run_cmd_capture(cmd, cwd=root)
        steps.append(
            {
                "name": "data_quality_gate",
                "executed": True,
                "cmd": cmd,
                **rec,
            }
        )
        if not rec.get("ok"):
            errors.append("data_quality_gate failed")
    else:
        steps.append({"name": "data_quality_gate", "executed": False, "reason": "already_exists"})

    # Generate reason coverage report if missing.
    if reason_missing_before:
        cmd = [py, "-m", "sllm.train.check_reason_coverage", "--model_dir", model_dir]
        try:
            ratio = float(auto_split_train_ratio) if auto_split_train_ratio is not None else None
        except Exception:
            ratio = None
        if ratio is not None:
            cmd.extend(
                [
                    "--train_file",
                    input_source,
                    "--auto_split",
                    "--train_ratio",
                    str(ratio),
                ]
            )
        else:
            cmd.extend(["--input_file", input_source])
        rec = _run_cmd_capture(cmd, cwd=root)
        steps.append(
            {
                "name": "check_reason_coverage",
                "executed": True,
                "cmd": cmd,
                **rec,
            }
        )
        if not rec.get("ok"):
            errors.append("check_reason_coverage failed")
    else:
        steps.append({"name": "check_reason_coverage", "executed": False, "reason": "already_exists"})

    quality = _read_json(quality_path) or {}
    reason_cov = _read_json(reason_cov_path) or {}
    reason_cov_ratio = ((reason_cov.get("reason_coverage") or {}).get("coverage_ratio")) if isinstance(reason_cov, dict) else None
    malformed_ratio = ((reason_cov.get("command_shape") or {}).get("malformed_ratio")) if isinstance(reason_cov, dict) else None

    quality_exists_after = os.path.exists(quality_path)
    reason_exists_after = os.path.exists(reason_cov_path)
    if not quality_exists_after:
        errors.append("data_quality_report.json still missing after prepare")
    if not reason_exists_after:
        errors.append("reason_coverage_report.json still missing after prepare")

    return {
        "ok": len(errors) == 0,
        "run_id": rid,
        "model_dir": os.path.abspath(model_dir),
        "input_source": os.path.abspath(input_source),
        "auto_split_train_ratio": auto_split_train_ratio,
        "python_path": py,
        "required_fields": required_fields,
        "before": {
            "data_quality_report_missing": quality_missing_before,
            "reason_coverage_report_missing": reason_missing_before,
        },
        "after": {
            "data_quality_report_exists": quality_exists_after,
            "reason_coverage_report_exists": reason_exists_after,
            "quality_status": quality.get("status") if isinstance(quality, dict) else None,
            "reason_coverage_ratio": reason_cov_ratio,
            "command_malformed_ratio": malformed_ratio,
        },
        "steps": steps,
        "errors": errors,
    }


@app.get("/release")
def release_page():
    html = """
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Release</title>
      <style>
        :root {
          --bg: #f6f7fb;
          --card: #ffffff;
          --text: #121826;
          --muted: #5f6b7a;
          --border: rgba(18,24,38,.12);
          --accent: #2563eb;
        }
        body {
          margin: 0;
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          background: radial-gradient(1200px 600px at 15% 10%, rgba(37,99,235,.14), transparent 55%),
                      radial-gradient(900px 600px at 75% 0%, rgba(16,185,129,.10), transparent 55%),
                      var(--bg);
          color: var(--text);
        }
        .wrap { max-width: 1100px; margin: 32px auto; padding: 0 16px; }
        .top { display:flex; justify-content:space-between; align-items:baseline; gap:16px; }
        h1 { margin: 0; font-size: 20px; }
        .nav a { margin-left: 10px; }
        .card {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 14px;
          box-shadow: 0 10px 30px rgba(18,24,38,.06);
          margin-top: 12px;
        }
        .card-title { font-weight: 600; margin-bottom: 10px; }
        .row { display:flex; gap:10px; align-items:center; flex-wrap: wrap; }
        .muted { color: var(--muted); font-size: 13px; }
        button {
          background: var(--accent);
          color: #fff;
          border: none;
          padding: 9px 12px;
          border-radius: 10px;
          cursor: pointer;
          font-weight: 600;
        }
        button.secondary {
          background: rgba(18,24,38,.08);
          color: var(--text);
          border: 1px solid var(--border);
        }
        input, select {
          min-width: 160px;
          padding: 8px;
          border-radius: 10px;
          border: 1px solid var(--border);
        }
        pre {
          white-space: pre-wrap;
          word-break: break-word;
          background: rgba(18,24,38,.03);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 12px;
          margin: 8px 0 0 0;
          font-size: 12.5px;
        }
        code { background: rgba(18,24,38,.06); padding: 2px 6px; border-radius: 6px; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="top">
          <h1>최종 반영(Release)</h1>
          <div class="nav">
            <a href="/dashboard">dashboard</a>
            <a href="/results">results</a>
            <a href="/wizard">wizard</a>
            <a href="/playground">playground</a>
            <a href="/train">train</a>
            <a href="/commands">commands</a>
          </div>
        </div>

        <div class="card">
          <div class="card-title">권장 운영 루틴</div>
          <div class="muted">1) 학습(RUN_ID 분리) → 2) 검수 승인 → 3) DRY_RUN 검증 → 4) 최종 반영 → 5) playground 테스트 → 6) 실제 API 호출</div>
        </div>

        <div class="card">
          <div class="card-title">RUN 선택</div>
          <div class="row">
            <label>run_id</label>
            <select id="runId"></select>
            <span id="runMeta" class="muted"></span>
          </div>
          <div class="row" style="margin-top:10px">
            <button class="secondary" id="openResults">결과 보기(/results)</button>
            <button class="secondary" id="openPlayground">선택 RUN 테스트(/playground)</button>
            <button class="secondary" id="openLivePlayground">LIVE 테스트(/playground)</button>
          </div>
        </div>

        <div class="card">
          <div class="card-title">2) 검수 승인</div>
          <div class="row">
            <label>reviewer</label><input id="reviewer" placeholder="예: rms"/>
            <label>note</label><input id="reviewNote" style="min-width:280px" placeholder="검수 코멘트"/>
            <button id="approve">검수 승인 저장</button>
            <button class="secondary" id="revoke">검수 승인 해제</button>
            <span id="reviewStatus" class="muted"></span>
          </div>
          <pre id="reviewOut">-</pre>
        </div>

        <div class="card">
          <div class="card-title">3~4) 최종 반영</div>
          <div class="row">
            <label>min_command_accuracy</label><input id="minAcc" placeholder="예: 0.82"/>
            <label>min_reason_coverage</label><input id="minCov" value="1.0"/>
            <label>max_command_malformed_ratio</label><input id="maxMalformed" value="0.0"/>
            <label><input type="checkbox" id="requireReview" checked/>검수 승인 필수</label>
          </div>
          <div class="row" style="margin-top:10px">
            <button class="secondary" id="dryRun">DRY_RUN 검증</button>
            <button id="finalize">최종 반영 실행</button>
            <span id="finalizeStatus" class="muted"></span>
          </div>
          <pre id="finalizeOut">-</pre>
        </div>

        <div class="card">
          <div class="card-title">CLI 참고</div>
          <pre id="cmdOut">-</pre>
        </div>
      </div>
      <script>
        const runId = document.getElementById('runId');
        const runMeta = document.getElementById('runMeta');
        const reviewOut = document.getElementById('reviewOut');
        const reviewStatus = document.getElementById('reviewStatus');
        const finalizeOut = document.getElementById('finalizeOut');
        const finalizeStatus = document.getElementById('finalizeStatus');
        const cmdOut = document.getElementById('cmdOut');
        let rows = [];
        let currentRunId = null;

        function selectedRunId() {
          return (runId.value || '').trim();
        }

        function refreshCmdPreview() {
          const rid = selectedRunId();
          if (!rid) {
            cmdOut.textContent = '-';
            return;
          }
          cmdOut.textContent =
`학습:
make train TRAIN_FILE=train_data/your_train.jsonl RESUME=never RUN_ID=${rid}

최종 반영(검증만):
make finalize FINALIZE_RUN_ID=${rid} DRY_RUN=1

최종 반영(실행):
make finalize FINALIZE_RUN_ID=${rid}
`;
        }

        function updateRunMeta() {
          const rid = selectedRunId();
          const row = rows.find(x => x.run_id === rid);
          if (!row) {
            runMeta.textContent = '';
            refreshCmdPreview();
            return;
          }
          const live = row.is_current ? 'LIVE' : 'not-live';
          runMeta.textContent = `exists=${row.exists} | quality=${row.quality_status ?? '-'} | acc=${row.command_accuracy ?? '-'} | ${live} | model_dir=${row.model_dir}`;
          refreshCmdPreview();
        }

        async function loadRuns() {
          const r = await fetch('/runs/list');
          const j = await r.json();
          rows = j.runs || [];
          currentRunId = j.current_run_id || null;
          runId.innerHTML = '';
          for (const x of rows) {
            const o = document.createElement('option');
            o.value = x.run_id;
            const live = x.is_current ? ' [LIVE]' : '';
            o.textContent = `${x.label}${live} (exists=${x.exists}, acc=${x.command_accuracy ?? '-'}, quality=${x.quality_status ?? '-'})`;
            runId.appendChild(o);
          }
          const nonDefault = rows.find(x => x.run_id !== '__default__' && x.exists);
          runId.value = currentRunId || (nonDefault ? nonDefault.run_id : (rows[0]?.run_id || '__default__'));
          runId.addEventListener('change', updateRunMeta);
          updateRunMeta();
        }

        async function saveReview(approved) {
          const rid = selectedRunId();
          if (!rid) {
            alert('run_id를 선택하세요');
            return;
          }
          reviewStatus.textContent = approved ? 'saving...' : 'revoking...';
          reviewOut.textContent = '';
          try {
            const res = await fetch('/runs/review/approve', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                run_id: rid,
                reviewer: (document.getElementById('reviewer').value || '').trim() || null,
                note: (document.getElementById('reviewNote').value || '').trim() || null,
                approved: !!approved,
              }),
            });
            const txt = await res.text();
            try {
              reviewOut.textContent = JSON.stringify(JSON.parse(txt), null, 2);
            } catch (_e) {
              reviewOut.textContent = txt;
            }
            reviewStatus.textContent = `done (HTTP ${res.status})`;
            await loadRuns();
          } catch (e) {
            reviewStatus.textContent = 'error';
            reviewOut.textContent = String(e);
          }
        }

        async function runFinalize(dryRun) {
          const rid = selectedRunId();
          if (!rid) {
            alert('run_id를 선택하세요');
            return;
          }
          const minAccRaw = (document.getElementById('minAcc').value || '').trim();
          const minCov = Number((document.getElementById('minCov').value || '1.0').trim());
          const maxMalformed = Number((document.getElementById('maxMalformed').value || '0.0').trim());
          if (!Number.isFinite(minCov) || !Number.isFinite(maxMalformed)) {
            alert('coverage/malformed 값은 숫자여야 합니다.');
            return;
          }
          const payload = {
            run_id: rid,
            dry_run: !!dryRun,
            min_reason_coverage: minCov,
            max_command_malformed_ratio: maxMalformed,
            require_review_approval: !!document.getElementById('requireReview').checked,
          };
          if (minAccRaw !== '') {
            const minAcc = Number(minAccRaw);
            if (!Number.isFinite(minAcc)) {
              alert('min_command_accuracy는 숫자여야 합니다.');
              return;
            }
            payload.min_command_accuracy = minAcc;
          }

          finalizeStatus.textContent = dryRun ? 'validating...' : 'finalizing...';
          finalizeOut.textContent = '';
          try {
            const res = await fetch('/runs/finalize', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
            });
            const txt = await res.text();
            let json = null;
            try {
              json = JSON.parse(txt);
              finalizeOut.textContent = JSON.stringify(json, null, 2);
            } catch (_e) {
              finalizeOut.textContent = txt;
            }
            finalizeStatus.textContent = `done (HTTP ${res.status})`;
            if (json && json.ok && !dryRun) {
              alert(`최종 반영 완료: LIVE run_id=${json.run_id}`);
              await loadRuns();
            }
          } catch (e) {
            finalizeStatus.textContent = 'error';
            finalizeOut.textContent = String(e);
          }
        }

        document.getElementById('approve').addEventListener('click', () => saveReview(true));
        document.getElementById('revoke').addEventListener('click', () => saveReview(false));
        document.getElementById('dryRun').addEventListener('click', () => runFinalize(true));
        document.getElementById('finalize').addEventListener('click', () => runFinalize(false));

        document.getElementById('openResults').addEventListener('click', () => {
          const rid = selectedRunId();
          if (!rid) return;
          location.href = `/results?run_id=${encodeURIComponent(rid)}`;
        });
        document.getElementById('openPlayground').addEventListener('click', () => {
          const rid = selectedRunId();
          if (!rid) return;
          location.href = `/playground?run_id=${encodeURIComponent(rid)}`;
        });
        document.getElementById('openLivePlayground').addEventListener('click', () => {
          if (currentRunId) {
            location.href = `/playground?run_id=${encodeURIComponent(currentRunId)}`;
            return;
          }
          location.href = '/playground';
        });

        loadRuns();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/wizard")
def wizard_page():
    # Server-side default so it's visible even if JS fails/cached.
    default_train_run_id = make_timestamp_run_id(prefix="run")
    run_rows = _list_runs_for_ui()
    if not run_rows:
        run_rows = [
            {
                "run_id": DEFAULT_RUN_ID,
                "label": "default",
                "exists": os.path.isdir(MODEL_DIR),
                "is_current": False,
                "command_accuracy": None,
                "quality_status": None,
            }
        ]
    selected_run_id = _read_current_promoted_run_id() or DEFAULT_RUN_ID
    option_rows = []
    for row in run_rows:
        rid = str(row.get("run_id", "")).strip() or DEFAULT_RUN_ID
        label = str(row.get("label", rid)).strip() or rid
        live = " [LIVE]" if row.get("is_current") else ""
        exists = row.get("exists")
        acc = row.get("command_accuracy")
        quality = row.get("quality_status")
        selected_attr = " selected" if rid == selected_run_id else ""
        text = f"{label}{live} (exists={exists}, acc={acc}, quality={quality})"
        option_rows.append(
            f'<option value="{_escape_html(rid)}"{selected_attr}>{_escape_html(text)}</option>'
        )
    run_options_html = "\n".join(option_rows)
    html = """
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>학습/반영 Wizard</title>
      <style>
        :root {
          --bg: #f6f7fb;
          --card: #ffffff;
          --text: #121826;
          --muted: #5f6b7a;
          --border: rgba(18,24,38,.12);
          --accent: #2563eb;
          --ok: #16a34a;
          --warn: #b45309;
        }
        body {
          margin: 0;
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          background: radial-gradient(1200px 600px at 15% 10%, rgba(37,99,235,.14), transparent 55%),
                      radial-gradient(900px 600px at 75% 0%, rgba(16,185,129,.10), transparent 55%),
                      var(--bg);
          color: var(--text);
        }
        .wrap { max-width: 1200px; margin: 28px auto; padding: 0 16px; }
        .top { display:flex; justify-content:space-between; align-items:baseline; gap:16px; }
        .top h1 { margin: 0; font-size: 22px; }
        .nav a { margin-left: 10px; }
        .card {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 14px;
          box-shadow: 0 10px 30px rgba(18,24,38,.06);
          margin-top: 12px;
        }
        .card-title { font-weight: 700; margin-bottom: 10px; }
        .muted { color: var(--muted); font-size: 13px; }
        .row { display:flex; gap:10px; align-items:center; flex-wrap: wrap; }
        .stack { display:grid; grid-template-columns: 1fr; gap: 12px; }
        .stepper { display:grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
        .step-chip {
          border: 1px solid var(--border);
          background: rgba(18,24,38,.02);
          border-radius: 999px;
          padding: 8px 10px;
          font-size: 12px;
          display:flex;
          justify-content: space-between;
          gap: 8px;
          align-items:center;
        }
        .badge {
          border-radius: 999px;
          padding: 2px 8px;
          font-size: 11px;
          font-weight: 700;
          background: rgba(18,24,38,.08);
          color: var(--muted);
        }
        .badge.done {
          background: rgba(22,163,74,.14);
          color: #166534;
        }
        textarea, input, select {
          box-sizing: border-box;
          padding: 8px 10px;
          border-radius: 10px;
          border: 1px solid var(--border);
          font-size: 13px;
        }
        textarea {
          width: 100%;
          min-height: 140px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }
        input[type="file"] { padding: 6px; background: #fff; }
        button {
          background: var(--accent);
          color: #fff;
          border: none;
          padding: 9px 12px;
          border-radius: 10px;
          cursor: pointer;
          font-weight: 600;
        }
        button.secondary {
          background: rgba(18,24,38,.08);
          color: var(--text);
          border: 1px solid var(--border);
        }
        button:disabled { opacity: .45; cursor: not-allowed; }
        pre {
          white-space: pre-wrap;
          word-break: break-word;
          background: rgba(18,24,38,.03);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 12px;
          margin: 8px 0 0 0;
          font-size: 12px;
        }
        .phase-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 8px;
          margin-top: 8px;
        }
        .phase-pill {
          border: 1px solid var(--border);
          border-radius: 999px;
          padding: 6px 10px;
          font-size: 12px;
          background: rgba(18,24,38,.04);
          color: var(--muted);
          font-weight: 600;
        }
        .phase-pill.running {
          background: rgba(37,99,235,.12);
          color: #1d4ed8;
          border-color: rgba(37,99,235,.25);
        }
        .phase-pill.done {
          background: rgba(22,163,74,.14);
          color: #166534;
          border-color: rgba(22,163,74,.24);
        }
        .phase-pill.fail {
          background: rgba(220,38,38,.12);
          color: #b91c1c;
          border-color: rgba(220,38,38,.24);
        }
        .metric-grid {
          margin-top: 8px;
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 8px;
        }
        .metric-card {
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 8px;
          background: rgba(18,24,38,.02);
        }
        .metric-k {
          color: var(--muted);
          font-size: 11px;
          margin-bottom: 4px;
        }
        .metric-v {
          color: var(--text);
          font-size: 14px;
          font-weight: 700;
          font-variant-numeric: tabular-nums;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }
        .metric-scroll {
          max-height: 180px;
          overflow: auto;
          border: 1px solid var(--border);
          border-radius: 10px;
          margin-top: 8px;
          background: rgba(18,24,38,.02);
        }
        table.metric-table { width: 100%; border-collapse: collapse; }
        table.metric-table th, table.metric-table td {
          border-top: 1px solid var(--border);
          padding: 6px 8px;
          font-size: 12px;
          text-align: left;
          font-variant-numeric: tabular-nums;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }
        table.metric-table th {
          position: sticky;
          top: 0;
          background: rgba(255,255,255,.92);
          z-index: 2;
        }
        #trainJobMeta, #trainTokOut, #trainFitOut, #trainEvalOut, #trainOut {
          max-height: 230px;
          overflow-y: auto;
          overflow-x: auto;
        }
        #trainOut { max-height: 300px; }
        .ok { color: var(--ok); font-weight: 700; }
        .warn { color: var(--warn); font-weight: 700; }
        .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .modal {
          position: fixed;
          inset: 0;
          display: none;
          align-items: center;
          justify-content: center;
          padding: 16px;
          background: rgba(18,24,38,.42);
          z-index: 9999;
        }
        .modal.open { display: flex; }
        .modal-card {
          width: min(1100px, 100%);
          max-height: 86vh;
          overflow: auto;
          background: #fff;
          border: 1px solid var(--border);
          border-radius: 14px;
          box-shadow: 0 20px 60px rgba(18,24,38,.25);
          padding: 14px;
        }
        /* detail view (HTML fragment) */
        table.kv { width: 100%; border-collapse: collapse; margin-top: 6px; }
        table.kv td, table.kv th { border-top: 1px solid var(--border); padding: 8px; font-size: 12px; vertical-align: top; }
        table.kv td.k, table.kv th.k { width: 220px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
        table.kv td.v, table.kv th.v { color: var(--text); }
        .dist-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
        .dist-card { border: 1px solid var(--border); border-radius: 12px; padding: 10px; background: rgba(18,24,38,.02); }
        .dist-title { font-weight: 700; font-size: 12px; margin-bottom: 8px; color: var(--text); }
        .dist-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 6px; }
        .dist-list li { display:flex; justify-content: space-between; gap: 10px; font-size: 12px; }
        .dist-list .k { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
        .dist-list .v { color: var(--text); font-variant-numeric: tabular-nums; }
        ul.ckpt { margin: 8px 0 0 18px; }
        ul.ckpt li { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }
        @media (max-width: 980px) {
          .stepper { grid-template-columns: 1fr; }
          .grid2 { grid-template-columns: 1fr; }
          .dist-grid { grid-template-columns: 1fr; }
        }
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="top">
          <h1>운영 Wizard (학습 → 테스트 → 반영 → 최종 확인)</h1>
          <div class="nav">
            <a href="/dashboard">dashboard</a>
            <a href="/results">results</a>
            <a href="/release">release</a>
            <a href="/playground">playground</a>
            <a href="/train">train</a>
            <a href="/commands">commands</a>
          </div>
        </div>

        <div class="card">
          <div class="card-title">진행 상태</div>
          <div class="muted">각 단계에서 버튼 실행 후, 활성화되는 "다음 단계" 버튼을 눌러 계속 진행하세요.</div>
          <div class="stepper">
            <div class="step-chip"><span>1. 학습</span><span id="badge1" class="badge">대기</span></div>
            <div class="step-chip"><span>2. 학습결과 테스트</span><span id="badge2" class="badge">대기</span></div>
            <div class="step-chip"><span>3. 반영</span><span id="badge3" class="badge">대기</span></div>
            <div class="step-chip"><span>4. 반영 후 테스트</span><span id="badge4" class="badge">대기</span></div>
            <div class="step-chip"><span>5. 대시보드 최종 확인</span><span id="badge5" class="badge">대기</span></div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">RUN_ID 선택</div>
          <div class="row">
            <select id="runSelect">__WIZARD_RUN_OPTIONS__</select>
            <button id="refreshRuns" class="secondary">RUN 목록 새로고침</button>
            <button id="openResults" class="secondary">선택 RUN 결과 보기</button>
            <button id="openRunPlayground" class="secondary">선택 RUN playground</button>
            <button id="openModelDetail" class="secondary">선택 RUN 모델정보(팝업)</button>
            <button id="deleteRunModel" class="secondary">선택 RUN 모델 삭제</button>
            <span id="deleteStatus" class="muted"></span>
          </div>
          <div id="runMeta" class="muted" style="margin-top:8px">-</div>
          <pre id="deleteOut">-</pre>
        </div>

        <div id="step1" class="card">
          <div class="card-title">1) 학습</div>
          <div class="muted">RUN_ID와 학습 파일을 지정해 학습을 시작합니다. 완료/실패 상태를 이 화면에서 계속 확인합니다.</div>
          <div class="row" style="margin-top:10px">
            <label>run_id</label>
            <input id="trainRunId" placeholder="run_20260414_103000" value="__DEFAULT_TRAIN_RUN_ID__"/>
            <input id="trainFile" type="file" accept=".jsonl,application/json"/>
            <button id="startTrain">학습 시작</button>
            <span id="trainStatus" class="muted"></span>
          </div>
          <div class="grid2" style="margin-top:10px">
            <div>
              <div class="muted">학습 파이프라인 단계</div>
              <div class="phase-grid">
                <div id="phaseQuality" class="phase-pill">품질게이트: 대기</div>
                <div id="phaseTokenizer" class="phase-pill">토크나이저: 대기</div>
                <div id="phaseTrain" class="phase-pill">모델학습: 대기</div>
                <div id="phaseEvaluate" class="phase-pill">평가: 대기</div>
              </div>
              <div class="muted" style="margin-top:8px">진행률: <code id="trainProgressText">-</code></div>
              <div class="muted" style="margin-top:10px">학습 상태(JSON)</div>
              <pre id="trainJobMeta">-</pre>
            </div>
            <div>
              <div class="muted">실시간 학습 지표</div>
              <div class="metric-grid">
                <div class="metric-card"><div class="metric-k">loss</div><div id="metricLoss" class="metric-v">-</div></div>
                <div class="metric-card"><div class="metric-k">grad_norm</div><div id="metricGradNorm" class="metric-v">-</div></div>
                <div class="metric-card"><div class="metric-k">learning_rate</div><div id="metricLearningRate" class="metric-v">-</div></div>
                <div class="metric-card"><div class="metric-k">epoch</div><div id="metricEpoch" class="metric-v">-</div></div>
              </div>
              <div class="muted" style="margin-top:8px">업데이트 시각: <code id="metricUpdatedAt">-</code></div>
              <div class="metric-scroll">
                <table class="metric-table">
                  <thead><tr><th>time</th><th>epoch</th><th>loss</th><th>grad_norm</th><th>learning_rate</th></tr></thead>
                  <tbody id="trainMetricRows"><tr><td colspan="5" class="muted">아직 없음</td></tr></tbody>
                </table>
              </div>
            </div>
          </div>
          <div class="grid2" style="margin-top:10px">
            <div>
              <div class="muted">토크나이저 로그</div>
              <pre id="trainTokOut">-</pre>
            </div>
            <div>
              <div class="muted">학습 로그</div>
              <pre id="trainFitOut">-</pre>
            </div>
          </div>
          <div style="margin-top:10px">
            <div class="muted">평가 로그</div>
            <pre id="trainEvalOut">-</pre>
          </div>
          <div style="margin-top:10px">
            <div class="muted">전체 로그</div>
            <pre id="trainOut">-</pre>
          </div>
          <div class="row" style="margin-top:10px">
            <button id="next1" class="secondary">2단계로 이동</button>
          </div>
        </div>

        <div id="step2" class="card">
          <div class="card-title">2) 학습결과 테스트</div>
          <div class="muted">선택 RUN_ID(<code id="preRunText">-</code>)로 `/infer?run_id=...` 테스트를 실행합니다.</div>
          <textarea id="preInput"></textarea>
          <div class="row" style="margin-top:10px">
            <button id="runPreTest">학습결과 테스트 실행</button>
            <button id="openStep2Playground" class="secondary">playground에서 같은 RUN 테스트</button>
            <span id="preStatus" class="muted"></span>
          </div>
          <pre id="preOut">-</pre>
          <div class="row" style="margin-top:10px">
            <button id="next2" class="secondary">3단계로 이동</button>
          </div>
        </div>

        <div id="step3" class="card">
          <div class="card-title">3) 반영</div>
          <div class="muted">선택 RUN_ID(<code id="promoteRunText">-</code>)에 대해 검수 승인 후 DRY_RUN 검증, 최종 반영 순서로 실행합니다.</div>
          <div class="row" style="margin-top:10px">
            <label>reviewer</label><input id="reviewer" placeholder="예: rms"/>
            <label>note</label><input id="reviewNote" style="min-width:260px" placeholder="검수 코멘트"/>
            <button id="approveReview">검수 승인 저장</button>
            <button id="revokeReview" class="secondary">검수 승인 해제</button>
            <span id="reviewStatus" class="muted"></span>
          </div>
          <pre id="reviewOut">-</pre>
          <div class="row" style="margin-top:10px">
            <label>min_command_accuracy</label><input id="minAcc" placeholder="예: 0.82"/>
            <label>min_reason_coverage</label><input id="minCov" value="1.0"/>
            <label>max_command_malformed_ratio</label><input id="maxMalformed" value="0.0"/>
            <label><input type="checkbox" id="requireReview" checked/>검수 승인 필수</label>
          </div>
          <div class="row" style="margin-top:10px">
            <button id="dryFinalize" class="secondary">DRY_RUN 검증</button>
            <button id="applyFinalize">최종 반영 실행</button>
            <span id="finalizeStatus" class="muted"></span>
          </div>
          <pre id="finalizeOut">-</pre>
          <div class="row" style="margin-top:10px">
            <button id="next3" class="secondary">4단계로 이동</button>
          </div>
        </div>

        <div id="step4" class="card">
          <div class="card-title">4) 반영 후 테스트</div>
          <div class="muted">`/infer`(run_id 없이 LIVE 모델)로 테스트해 반영된 결과를 확인합니다.</div>
          <textarea id="postInput"></textarea>
          <div class="row" style="margin-top:10px">
            <button id="runPostTest">LIVE 테스트 실행</button>
            <button id="openLivePlayground" class="secondary">LIVE playground 열기</button>
            <span id="postStatus" class="muted"></span>
          </div>
          <pre id="postOut">-</pre>
          <div class="row" style="margin-top:10px">
            <button id="next4" class="secondary">5단계로 이동</button>
          </div>
        </div>

        <div id="step5" class="card">
          <div class="card-title">5) 대시보드로 최종 반영 확인</div>
          <div class="grid2">
            <div>
              <div class="muted">선택 RUN_ID: <code id="expectedRunText">-</code></div>
              <div class="muted">현재 LIVE RUN_ID: <code id="currentRunText">-</code></div>
              <div id="liveMatchText" class="warn" style="margin-top:8px">아직 확인 전</div>
              <div class="row" style="margin-top:10px">
                <button id="refreshLiveState">LIVE 반영 상태 새로고침</button>
                <button id="openDashboard" class="secondary">dashboard 열기</button>
              </div>
            </div>
            <div>
              <pre id="liveOut">-</pre>
            </div>
          </div>
        </div>

        <div id="detailModal" class="modal">
          <div class="modal-card">
            <div class="row" style="justify-content:space-between; margin-bottom:10px;">
              <div>
                <div class="card-title" style="margin-bottom:4px;">RUN 모델 정보 (팝업)</div>
                <div class="muted">선택 RUN의 학습/품질/분포/파일/잡 이력을 포함한 상세 정보</div>
              </div>
              <div class="row">
                <button id="refreshModelDetail" class="secondary">새로고침</button>
                <button id="closeModelDetail" class="secondary">닫기</button>
              </div>
            </div>
            <div id="modelDetailOut" class="stack">-</div>
          </div>
        </div>
      </div>

      <script>
        const stepDone = { 1:false, 2:false, 3:false, 4:false, 5:false };
        let rowsCache = [];
        let currentRunId = null;

        const runSelect = document.getElementById('runSelect');
        const runMeta = document.getElementById('runMeta');
        const trainRunId = document.getElementById('trainRunId');
        const detailModal = document.getElementById('detailModal');
        const modelDetailOut = document.getElementById('modelDetailOut');
        const deleteStatus = document.getElementById('deleteStatus');
        const deleteOut = document.getElementById('deleteOut');

        function makeDefaultRunId() {
          const now = new Date();
          const pad2 = (n) => String(n).padStart(2, '0');
          const y = now.getFullYear();
          const m = pad2(now.getMonth() + 1);
          const d = pad2(now.getDate());
          const hh = pad2(now.getHours());
          const mm = pad2(now.getMinutes());
          const ss = pad2(now.getSeconds());
          return `run_${y}${m}${d}_${hh}${mm}${ss}`;
        }

        function initTrainRunIdDefault() {
          const cur = (trainRunId.value || '').trim();
          // If empty, populate a timestamp-based run_id by default.
          if (!cur) {
            trainRunId.value = makeDefaultRunId();
          }
        }

        function sleep(ms) {
          return new Promise(resolve => setTimeout(resolve, ms));
        }

        function setDone(step, done) {
          stepDone[step] = !!done;
          const badge = document.getElementById(`badge${step}`);
          badge.textContent = done ? '완료' : '대기';
          badge.className = done ? 'badge done' : 'badge';
        }

        function gotoStep(step) {
          const el = document.getElementById(`step${step}`);
          if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function selectedRunId() {
          return (runSelect.value || '').trim();
        }

        function ensureRunOption(rid) {
          if (!rid) return;
          const found = Array.from(runSelect.options).some(o => o.value === rid);
          if (found) return;
          const o = document.createElement('option');
          o.value = rid;
          o.textContent = `${rid} (new)`;
          runSelect.appendChild(o);
        }

        function syncRunMeta() {
          const rid = selectedRunId();
          const row = rowsCache.find(x => x.run_id === rid);
          if (row) {
            const live = row.is_current ? 'LIVE' : 'not-live';
            runMeta.textContent = `exists=${row.exists} | quality=${row.quality_status ?? '-'} | acc=${row.command_accuracy ?? '-'} | ${live} | model_dir=${row.model_dir}`;
          } else {
            runMeta.textContent = rid ? `run_id=${rid}` : '-';
          }
          document.getElementById('preRunText').textContent = rid || '-';
          document.getElementById('promoteRunText').textContent = rid || '-';
          document.getElementById('expectedRunText').textContent = rid || '-';
        }

        function setRunSelection(rid) {
          if (!rid) return;
          ensureRunOption(rid);
          runSelect.value = rid;
          syncRunMeta();
        }

        async function loadRuns(preferredRid = null) {
          const prev = (preferredRid || selectedRunId() || '').trim();
          let json = null;
          try {
            const res = await fetch('/runs/list', { cache: 'no-store' });
            if (!res.ok) {
              throw new Error(`/runs/list HTTP ${res.status}`);
            }
            json = await res.json();
          } catch (e) {
            // /runs/list 로드 실패 시에도 드롭다운이 비지 않도록 기본 RUN을 보장한다.
            rowsCache = [{
              run_id: '__default__',
              label: 'default',
              exists: true,
              is_current: false,
              command_accuracy: null,
              quality_status: null,
              model_dir: '-'
            }];
            currentRunId = null;
            runSelect.innerHTML = '';
            const o = document.createElement('option');
            o.value = '__default__';
            o.textContent = 'default (fallback)';
            runSelect.appendChild(o);
            const fallback = prev || '__default__';
            setRunSelection(fallback);
            runMeta.textContent = `RUN 목록 로드 실패: ${String(e)}`;
            initTrainRunIdDefault();
            try { await refreshLiveState(false); } catch (_err) {}
            return;
          }

          const allRows = json.runs || [];
          // Wizard는 전체 RUN을 항상 노출한다.
          rowsCache = Array.isArray(allRows) ? allRows.slice() : [];
          if (!rowsCache.some(x => x.run_id === '__default__')) {
            rowsCache.unshift({
              run_id: '__default__',
              label: 'default',
              exists: true,
              is_current: false,
              command_accuracy: null,
              quality_status: null,
              model_dir: '-'
            });
          }
          currentRunId = json.current_run_id || null;

          runSelect.innerHTML = '';
          for (const row of rowsCache) {
            const o = document.createElement('option');
            o.value = row.run_id;
            const live = row.is_current ? ' [LIVE]' : '';
            const label = row.label || row.run_id || 'run';
            o.textContent = `${label}${live} (exists=${row.exists}, acc=${row.command_accuracy ?? '-'}, quality=${row.quality_status ?? '-'})`;
            runSelect.appendChild(o);
          }

          const hasPrev = prev && rowsCache.some(x => x.run_id === prev);
          const fallback = hasPrev ? prev : '__default__';
          setRunSelection(fallback);
          initTrainRunIdDefault();
          await refreshLiveState(false);
        }

        function parseInput(text) {
          const trimmed = (text || '').trim();
          if (!trimmed) return null;
          try {
            return JSON.parse(trimmed);
          } catch (_e) {
            return trimmed;
          }
        }

        function stripAnsi(text) {
          return String(text || '').replace(/\\x1B\\[[0-?]*[ -/]*[@-~]/g, '');
        }

        function normalizeLogChunk(text) {
          return stripAnsi(String(text || '')).replace(/\\r+/g, '\\n');
        }

        function nowTimeText() {
          try {
            return new Date().toLocaleTimeString('ko-KR', { hour12: false });
          } catch (_e) {
            return new Date().toISOString();
          }
        }

        function metricText(v) {
          if (v === null || v === undefined || v === '') return '-';
          const n = Number(v);
          if (!Number.isFinite(n)) return String(v);
          if (Math.abs(n) >= 1000 || (Math.abs(n) > 0 && Math.abs(n) < 0.001)) return n.toExponential(3);
          return n.toFixed(6).replace(/\\.?0+$/, '');
        }

        function parsePythonDictLine(line) {
          const t = String(line || '').trim();
          if (!t.includes('{') || !t.includes('}')) return null;
          const s = t.indexOf('{');
          const e = t.lastIndexOf('}');
          if (s < 0 || e <= s) return null;
          const body = t.slice(s, e + 1);
          if (!body.includes(':')) return null;
          try {
            const jsonLike = body
              .replace(/'/g, '"')
              .replace(/\\bNone\\b/g, 'null')
              .replace(/\\bTrue\\b/g, 'true')
              .replace(/\\bFalse\\b/g, 'false');
            return JSON.parse(jsonLike);
          } catch (_e) {
            return null;
          }
        }

        function parseProgressLine(line) {
          const t = String(line || '');
          const m = t.match(/(\\d{1,3})%\\|.*?\\|\\s*([0-9]+)\\/([0-9]+)/);
          if (!m) return null;
          return {
            percent: Number(m[1]),
            current: Number(m[2]),
            total: Number(m[3]),
          };
        }

        function setPhaseState(phase, state, detail = '') {
          const phaseIdMap = {
            quality: 'phaseQuality',
            tokenizer: 'phaseTokenizer',
            train: 'phaseTrain',
            evaluate: 'phaseEvaluate',
          };
          const phaseLabelMap = {
            quality: '품질게이트',
            tokenizer: '토크나이저',
            train: '모델학습',
            evaluate: '평가',
          };
          const stateLabelMap = {
            idle: '대기',
            running: '진행중',
            done: '완료',
            fail: '실패',
          };
          const el = document.getElementById(phaseIdMap[phase] || '');
          if (!el) return;
          const statusText = stateLabelMap[state] || stateLabelMap.idle;
          const suffix = detail ? ` (${detail})` : '';
          el.textContent = `${phaseLabelMap[phase] || phase}: ${statusText}${suffix}`;
          const cls = state === 'running' ? 'running' : (state === 'done' ? 'done' : (state === 'fail' ? 'fail' : ''));
          el.className = cls ? `phase-pill ${cls}` : 'phase-pill';
        }

        function appendLogText(el, text, maxLen = 240000) {
          if (!el) return;
          const cur = (el.textContent || '') === '-' ? '' : (el.textContent || '');
          const next = (cur ? (cur + '\\n') : '') + text;
          el.textContent = next.length > maxLen ? next.slice(next.length - maxLen) : next;
          el.scrollTop = el.scrollHeight;
        }

        function resetTrainMonitor() {
          const metricRows = document.getElementById('trainMetricRows');
          const trainProgressText = document.getElementById('trainProgressText');
          const trainJobMeta = document.getElementById('trainJobMeta');
          const trainTokOut = document.getElementById('trainTokOut');
          const trainFitOut = document.getElementById('trainFitOut');
          const trainEvalOut = document.getElementById('trainEvalOut');
          const trainOut = document.getElementById('trainOut');
          document.getElementById('metricLoss').textContent = '-';
          document.getElementById('metricGradNorm').textContent = '-';
          document.getElementById('metricLearningRate').textContent = '-';
          document.getElementById('metricEpoch').textContent = '-';
          document.getElementById('metricUpdatedAt').textContent = '-';
          trainProgressText.textContent = '-';
          trainJobMeta.textContent = '-';
          trainTokOut.textContent = '-';
          trainFitOut.textContent = '-';
          trainEvalOut.textContent = '-';
          trainOut.textContent = '-';
          metricRows.innerHTML = "<tr><td colspan='5' class='muted'>아직 없음</td></tr>";
          setPhaseState('quality', 'idle');
          setPhaseState('tokenizer', 'idle');
          setPhaseState('train', 'idle');
          setPhaseState('evaluate', 'idle');
        }

        function watchTrainingJobSSE(jobId, rid) {
          const trainStatus = document.getElementById('trainStatus');
          const trainJobMeta = document.getElementById('trainJobMeta');
          const trainTokOut = document.getElementById('trainTokOut');
          const trainFitOut = document.getElementById('trainFitOut');
          const trainEvalOut = document.getElementById('trainEvalOut');
          const trainOut = document.getElementById('trainOut');
          const trainProgressText = document.getElementById('trainProgressText');
          const metricRowsEl = document.getElementById('trainMetricRows');
          trainStatus.textContent = `학습 진행 중 (job=${jobId})`;

          const started = Date.now();
          const es = new EventSource(`/train/jobs/${encodeURIComponent(jobId)}/events`);
          let activePhase = null;
          let metricRows = [];

          function pushMetricRow(row) {
            metricRows.unshift(row);
            if (metricRows.length > 30) metricRows = metricRows.slice(0, 30);
            metricRowsEl.innerHTML = metricRows
              .map((x) => `<tr><td>${x.time}</td><td>${x.epoch}</td><td>${x.loss}</td><td>${x.grad_norm}</td><td>${x.learning_rate}</td></tr>`)
              .join('');
          }

          function applyMetricUpdate(obj) {
            if (!obj || typeof obj !== 'object') return;
            const lossVal = (obj.loss !== undefined) ? obj.loss : obj.train_loss;
            if (lossVal !== undefined) document.getElementById('metricLoss').textContent = metricText(lossVal);
            if (obj.grad_norm !== undefined) document.getElementById('metricGradNorm').textContent = metricText(obj.grad_norm);
            if (obj.learning_rate !== undefined) document.getElementById('metricLearningRate').textContent = metricText(obj.learning_rate);
            if (obj.epoch !== undefined) document.getElementById('metricEpoch').textContent = metricText(obj.epoch);

            const hasMainMetric = (lossVal !== undefined) || (obj.grad_norm !== undefined) || (obj.learning_rate !== undefined) || (obj.epoch !== undefined);
            if (hasMainMetric) {
              document.getElementById('metricUpdatedAt').textContent = nowTimeText();
            }
            if (obj.loss !== undefined || obj.train_loss !== undefined) {
              pushMetricRow({
                time: nowTimeText(),
                epoch: metricText(obj.epoch),
                loss: metricText(lossVal),
                grad_norm: metricText(obj.grad_norm),
                learning_rate: metricText(obj.learning_rate),
              });
            }
            if (obj.eval_loss !== undefined && activePhase === 'evaluate') {
              setPhaseState('evaluate', 'running', `eval_loss=${metricText(obj.eval_loss)}`);
            }
          }

          function applyRunCommandPhase(line) {
            if (!line.startsWith('[run]')) return;
            if (line.includes('sllm.train.data_quality_gate')) {
              activePhase = 'quality';
              setPhaseState('quality', 'running');
              return;
            }
            if (line.includes('sllm.tokenizer.train_tokenizer')) {
              if (activePhase === 'quality') setPhaseState('quality', 'done');
              activePhase = 'tokenizer';
              setPhaseState('tokenizer', 'running');
              return;
            }
            if (line.includes('sllm.train.train_decoder')) {
              if (activePhase === 'tokenizer') setPhaseState('tokenizer', 'done');
              activePhase = 'train';
              setPhaseState('train', 'running');
              return;
            }
            if (line.includes('sllm.train.evaluate')) {
              if (activePhase === 'train') setPhaseState('train', 'done');
              activePhase = 'evaluate';
              setPhaseState('evaluate', 'running');
              return;
            }
          }

          function applyLine(lineRaw) {
            const line = String(lineRaw || '').trim();
            if (!line) return;
            applyRunCommandPhase(line);

            if (line.startsWith('status:')) {
              if (line.includes('PASS')) setPhaseState('quality', 'done');
              if (line.includes('FAIL')) setPhaseState('quality', 'fail');
            }
            if (line.includes('saved:') && line.includes('tokenizer.json')) {
              setPhaseState('tokenizer', 'done');
            }
            if (line.includes('saved:') && line.includes('train_metrics.json')) {
              setPhaseState('train', 'done');
            }
            if (line.includes('saved:') && line.includes('/metrics.json')) {
              setPhaseState('evaluate', 'done');
            }
            if (line.startsWith('Traceback') || line.includes('CalledProcessError')) {
              if (activePhase) setPhaseState(activePhase, 'fail');
            }

            const prog = parseProgressLine(line);
            if (prog) {
              if (Number.isFinite(prog.percent) && Number.isFinite(prog.current) && Number.isFinite(prog.total)) {
                trainProgressText.textContent = `${prog.percent}% (${prog.current}/${prog.total})`;
              }
              return;
            }

            const parsedMetric = parsePythonDictLine(line);
            if (parsedMetric) {
              applyMetricUpdate(parsedMetric);
            }

            appendLogText(trainOut, line, 320000);
            if (activePhase === 'tokenizer') appendLogText(trainTokOut, line, 120000);
            if (activePhase === 'train') appendLogText(trainFitOut, line, 180000);
            if (activePhase === 'evaluate') appendLogText(trainEvalOut, line, 120000);
          }

          function finalize(ok) {
            try { es.close(); } catch (_) {}
            if (ok) {
              if (activePhase === 'evaluate') setPhaseState('evaluate', 'done');
              trainStatus.textContent = '학습 완료';
              setDone(1, true);
              if (rid) {
                setRunSelection(rid);
                trainRunId.value = rid;
              }
              loadRuns(rid).then(() => gotoStep(2));
            } else {
              if (activePhase) setPhaseState(activePhase, 'fail');
              trainStatus.textContent = '학습 실패';
              setDone(1, false);
            }
          }

          es.addEventListener('status', (e) => {
            try {
              const j = JSON.parse(e.data);
              trainJobMeta.textContent = JSON.stringify(j, null, 2);
            } catch (_) {}
          });
          es.addEventListener('log', (e) => {
            const cleaned = normalizeLogChunk(e.data || '');
            const lines = cleaned.split('\\n');
            for (const line of lines) {
              applyLine(line);
            }
          });
          es.addEventListener('done', (e) => {
            const status = (e.data || '').trim();
            finalize(status === 'succeeded');
          });
          es.onerror = () => {
            // If SSE connection drops, fall back to lightweight polling with backoff.
            es.close();
            let delay = 1000;
            (async function pollFallback() {
              while (true) {
                try {
                  const res = await fetch(`/train/jobs/${encodeURIComponent(jobId)}`);
                  const j = await res.json();
                  trainJobMeta.textContent = JSON.stringify(j, null, 2);
                  if (j.status === 'succeeded') return finalize(true);
                  if (j.status === 'failed') return finalize(false);
                } catch (_) {}
                if (Date.now() - started > 1000 * 60 * 60 * 12) {
                  trainStatus.textContent = '학습 상태 조회 타임아웃(12h)';
                  return;
                }
                await sleep(delay);
                delay = Math.min(30000, Math.floor(delay * 1.6));
              }
            })();
          };
        }

        async function startTraining() {
          const trainStatus = document.getElementById('trainStatus');
          const trainJobMeta = document.getElementById('trainJobMeta');
          const trainOut = document.getElementById('trainOut');
          const fileEl = document.getElementById('trainFile');
          const file = fileEl.files && fileEl.files[0] ? fileEl.files[0] : null;
          if (!file) {
            alert('학습 파일(.jsonl)을 선택하세요.');
            return;
          }
          const ridRaw = (trainRunId.value || '').trim();
          trainStatus.textContent = '업로드/학습 시작 중...';
          resetTrainMonitor();
          try {
            const fd = new FormData();
            fd.append('file', file);
            if (ridRaw) fd.append('run_id', ridRaw);
            const res = await fetch('/train/upload', { method: 'POST', body: fd });
            const text = await res.text();
            let json = null;
            try {
              json = JSON.parse(text);
              trainJobMeta.textContent = JSON.stringify(json, null, 2);
            } catch (_e) {
              trainJobMeta.textContent = text;
            }
            if (!json || !json.job_id) {
              trainStatus.textContent = `학습 시작 실패 (HTTP ${res.status})`;
              return;
            }
            const rid = (json.run_id || ridRaw || '').trim();
            if (rid) {
              setRunSelection(rid);
              trainRunId.value = rid;
            }
            trainStatus.textContent = `학습 진행 중 (job=${json.job_id})`;
            try { localStorage.setItem('last_train_job_id', String(json.job_id)); } catch (_) {}
            watchTrainingJobSSE(json.job_id, rid);
          } catch (e) {
            trainStatus.textContent = 'error';
            trainOut.textContent = String(e);
          }
        }

        async function runPreTest() {
          const rid = selectedRunId();
          const preStatus = document.getElementById('preStatus');
          const preOut = document.getElementById('preOut');
          const inputText = document.getElementById('preInput').value || '';
          const payload = parseInput(inputText);
          if (!rid) {
            alert('run_id를 선택하세요.');
            return;
          }
          if (payload === null) {
            alert('테스트 입력을 넣어주세요.');
            return;
          }
          preStatus.textContent = '테스트 실행 중...';
          preOut.textContent = '';
          try {
            const res = await fetch(`/infer?run_id=${encodeURIComponent(rid)}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
            });
            const text = await res.text();
            try {
              preOut.textContent = JSON.stringify(JSON.parse(text), null, 2);
            } catch (_e) {
              preOut.textContent = text;
            }
            preStatus.textContent = `완료 (HTTP ${res.status})`;
            if (res.ok) {
              setDone(2, true);
              document.getElementById('postInput').value = inputText;
            }
          } catch (e) {
            preStatus.textContent = 'error';
            preOut.textContent = String(e);
          }
        }

        async function saveReview(approved) {
          const rid = selectedRunId();
          const reviewStatus = document.getElementById('reviewStatus');
          const reviewOut = document.getElementById('reviewOut');
          if (!rid) {
            alert('run_id를 선택하세요.');
            return;
          }
          reviewStatus.textContent = approved ? '승인 저장 중...' : '승인 해제 중...';
          reviewOut.textContent = '';
          try {
            const res = await fetch('/runs/review/approve', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                run_id: rid,
                reviewer: (document.getElementById('reviewer').value || '').trim() || null,
                note: (document.getElementById('reviewNote').value || '').trim() || null,
                approved: !!approved,
              }),
            });
            const text = await res.text();
            try {
              reviewOut.textContent = JSON.stringify(JSON.parse(text), null, 2);
            } catch (_e) {
              reviewOut.textContent = text;
            }
            reviewStatus.textContent = `완료 (HTTP ${res.status})`;
            await loadRuns(rid);
          } catch (e) {
            reviewStatus.textContent = 'error';
            reviewOut.textContent = String(e);
          }
        }

        async function runFinalize(dryRun, opts = {}) {
          const rid = selectedRunId();
          const finalizeStatus = document.getElementById('finalizeStatus');
          const finalizeOut = document.getElementById('finalizeOut');
          const autoFromStep4 = !!(opts && opts.autoFromStep4);
          if (!rid) {
            alert('run_id를 선택하세요.');
            return { ok: false, status: null, response: null, error: 'run_id_required' };
          }
          const minAccRaw = (document.getElementById('minAcc').value || '').trim();
          const minCov = Number((document.getElementById('minCov').value || '1.0').trim());
          const maxMalformed = Number((document.getElementById('maxMalformed').value || '0.0').trim());
          if (!Number.isFinite(minCov) || !Number.isFinite(maxMalformed)) {
            alert('min_reason_coverage / max_command_malformed_ratio는 숫자여야 합니다.');
            return { ok: false, status: null, response: null, error: 'invalid_threshold' };
          }
          const payload = {
            run_id: rid,
            dry_run: !!dryRun,
            min_reason_coverage: minCov,
            max_command_malformed_ratio: maxMalformed,
            require_review_approval: !!document.getElementById('requireReview').checked,
          };
          if (minAccRaw) {
            const minAcc = Number(minAccRaw);
            if (!Number.isFinite(minAcc)) {
              alert('min_command_accuracy는 숫자여야 합니다.');
              return { ok: false, status: null, response: null, error: 'invalid_min_command_accuracy' };
            }
            payload.min_command_accuracy = minAcc;
          }

          finalizeStatus.textContent = dryRun
            ? 'DRY_RUN 검증 중...'
            : (autoFromStep4 ? 'LIVE 자동 반영 중...' : '최종 반영 중...');
          finalizeOut.textContent = '';
          try {
            const res = await fetch('/runs/finalize', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
            });
            const text = await res.text();
            let json = null;
            try {
              json = JSON.parse(text);
              finalizeOut.textContent = JSON.stringify(json, null, 2);
            } catch (_e) {
              finalizeOut.textContent = text;
            }
            finalizeStatus.textContent = `완료 (HTTP ${res.status})`;
            const ok = !!(json && json.ok && res.ok);
            if (ok && !dryRun) {
              setDone(3, true);
              await loadRuns(rid);
              await refreshLiveState(true);
              if (!autoFromStep4) {
                gotoStep(4);
              }
            }
            return { ok, status: res.status, response: json, error: null };
          } catch (e) {
            finalizeStatus.textContent = 'error';
            finalizeOut.textContent = String(e);
            return { ok: false, status: null, response: null, error: String(e) };
          }
        }

        async function runPostTest() {
          const postStatus = document.getElementById('postStatus');
          const postOut = document.getElementById('postOut');
          const payload = parseInput(document.getElementById('postInput').value || '');
          if (payload === null) {
            alert('테스트 입력을 넣어주세요.');
            return;
          }

          async function callLiveInferOnce() {
            const res = await fetch('/infer', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
            });
            const text = await res.text();
            let json = null;
            try {
              json = JSON.parse(text);
              postOut.textContent = JSON.stringify(json, null, 2);
            } catch (_e) {
              postOut.textContent = text;
            }
            return { res, json };
          }

          function isDefaultLiveLoadFail(json) {
            if (!json || typeof json !== 'object') return false;
            const cmd = String(json.command || '');
            const reason = String(json.reason || '');
            if (cmd !== 'NO_ACTION') return false;
            return (
              reason.includes('run_id model load failed: __default__') ||
              reason.includes('run_id model not found: __default__')
            );
          }

          postStatus.textContent = 'LIVE 테스트 실행 중...';
          postOut.textContent = '';
          try {
            let { res, json } = await callLiveInferOnce();
            if (res.ok && isDefaultLiveLoadFail(json)) {
              postStatus.textContent = 'LIVE 모델 로드 실패 감지: 선택 RUN 자동 반영 후 재테스트 중...';
              const promoted = await ensureLivePromotionBeforeStep5();
              if (promoted) {
                ({ res, json } = await callLiveInferOnce());
              } else {
                postStatus.textContent = 'LIVE 반영 실패: 5단계 이동 전에 반영 조건을 확인하세요.';
                return;
              }
            }
            postStatus.textContent = `완료 (HTTP ${res.status})`;
            if (res.ok) {
              setDone(4, true);
            }
          } catch (e) {
            postStatus.textContent = 'error';
            postOut.textContent = String(e);
          }
        }

        function getFinalizeErrorsFromResponse(resp) {
          const lines = [];
          if (!resp || typeof resp !== 'object') return lines;
          if (Array.isArray(resp.errors)) {
            for (const e of resp.errors) lines.push(String(e));
          }
          const v = resp.validation || {};
          if (Array.isArray(v.errors)) {
            for (const e of v.errors) lines.push(String(e));
          }
          if (resp.error) lines.push(String(resp.error));
          return Array.from(new Set(lines.filter(Boolean)));
        }

        function shouldTryPrepareArtifactsFromErrors(errs) {
          if (!Array.isArray(errs) || errs.length === 0) return false;
          const text = errs.join('\\n');
          return (
            text.includes('data_quality_report.json') ||
            text.includes('reason_coverage_report.json') ||
            text.includes('data_quality_report.status') ||
            text.includes('reason coverage ratio None') ||
            text.includes('command malformed ratio None')
          );
        }

        async function autoPrepareFinalizeArtifactsForStep5() {
          const rid = selectedRunId();
          if (!rid) return { ok: false, error: 'run_id_required', response: null };
          const finalizeStatus = document.getElementById('finalizeStatus');
          const finalizeOut = document.getElementById('finalizeOut');
          finalizeStatus.textContent = 'LIVE 반영 전 리포트 자동 준비 중...';
          try {
            const res = await fetch('/runs/prepare_finalize', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                run_id: rid,
                required_fields: 'message,system',
              }),
            });
            const text = await res.text();
            let json = null;
            try {
              json = JSON.parse(text);
              finalizeOut.textContent = JSON.stringify(json, null, 2);
            } catch (_e) {
              finalizeOut.textContent = text;
            }
            const ok = !!(res.ok && json && json.ok);
            finalizeStatus.textContent = ok
              ? `리포트 준비 완료 (HTTP ${res.status})`
              : `리포트 준비 실패 (HTTP ${res.status})`;
            return { ok, response: json, error: null };
          } catch (e) {
            finalizeStatus.textContent = '리포트 준비 error';
            return { ok: false, response: null, error: String(e) };
          }
        }

        async function autoApproveReviewForStep5() {
          const rid = selectedRunId();
          if (!rid) return false;
          const reviewerEl = document.getElementById('reviewer');
          const noteEl = document.getElementById('reviewNote');
          const reviewer = (reviewerEl.value || '').trim() || 'wizard-auto';
          const note = (noteEl.value || '').trim() || 'wizard step5 auto approval';
          reviewerEl.value = reviewer;
          noteEl.value = note;
          try {
            const res = await fetch('/runs/review/approve', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                run_id: rid,
                reviewer,
                note,
                approved: true,
              }),
            });
            const reviewStatus = document.getElementById('reviewStatus');
            reviewStatus.textContent = `자동 검수 승인 완료 (HTTP ${res.status})`;
            return res.ok;
          } catch (_e) {
            return false;
          }
        }

        async function ensureLivePromotionBeforeStep5() {
          const rid = selectedRunId();
          if (!rid) {
            alert('run_id를 선택하세요.');
            return false;
          }
          await refreshLiveState(false);
          if (currentRunId && currentRunId === rid) {
            return true;
          }
          let result = await runFinalize(false, { autoFromStep4: true });
          if (!result.ok) {
            const errs = getFinalizeErrorsFromResponse(result.response);
            if (shouldTryPrepareArtifactsFromErrors(errs)) {
              const prep = await autoPrepareFinalizeArtifactsForStep5();
              if (prep.ok) {
                result = await runFinalize(false, { autoFromStep4: true });
              } else {
                const prepErrs = getFinalizeErrorsFromResponse(prep.response);
                const prepDetail = prepErrs.length
                  ? prepErrs.map((x, i) => `${i + 1}. ${x}`).join('\\n')
                  : (prep.error || 'prepare finalize artifacts failed');
                alert(`LIVE 준비 실패 상세:\\n${prepDetail}`);
                return false;
              }
            }
          }
          if (!result.ok) {
            const errs = getFinalizeErrorsFromResponse(result.response);
            const onlyReviewMissing =
              errs.length > 0 &&
              errs.every((x) => x.includes('review approval required'));
            if (onlyReviewMissing) {
              const approved = await autoApproveReviewForStep5();
              if (approved) {
                result = await runFinalize(false, { autoFromStep4: true });
              }
            }
            if (!result.ok) {
              const errs2 = getFinalizeErrorsFromResponse(result.response);
              const detail = errs2.length
                ? errs2.map((x, i) => `${i + 1}. ${x}`).join('\\n')
                : (result.error || 'unknown finalize failure');
              alert(`LIVE 반영 실패 상세:\\n${detail}`);
              return false;
            }
          }
          await loadRuns(rid);
          await refreshLiveState(false);
          const matched = !!currentRunId && currentRunId === rid;
          if (!matched) {
            alert(`LIVE 반영 상태 확인 실패:\\n선택 RUN_ID=${rid}\\n현재 LIVE RUN_ID=${currentRunId || '-'}`);
          }
          return matched;
        }

        async function refreshLiveState(markDone) {
          const currentRunText = document.getElementById('currentRunText');
          const liveMatchText = document.getElementById('liveMatchText');
          const liveOut = document.getElementById('liveOut');
          const rid = selectedRunId();
          try {
            const res = await fetch('/runs/current');
            const json = await res.json();
            currentRunId = json.current_run_id || null;
            currentRunText.textContent = currentRunId || '-';
            liveOut.textContent = JSON.stringify(json, null, 2);
            const matched = !!rid && !!currentRunId && rid === currentRunId;
            if (matched) {
              liveMatchText.textContent = '선택 RUN이 LIVE로 최종 반영되었습니다.';
              liveMatchText.className = 'ok';
            } else {
              liveMatchText.textContent = '선택 RUN이 아직 LIVE가 아닙니다.';
              liveMatchText.className = 'warn';
            }
            if (markDone) {
              setDone(5, matched);
            }
          } catch (e) {
            liveOut.textContent = String(e);
          }
        }

        async function loadModelDetail() {
          const rid = selectedRunId();
          if (!rid) {
            alert('run_id를 선택하세요.');
            return;
          }
          modelDetailOut.textContent = 'loading...';
          try {
            const res = await fetch(`/runs/detail_view?run_id=${encodeURIComponent(rid)}`);
            const text = await res.text();
            modelDetailOut.innerHTML = text;
          } catch (e) {
            modelDetailOut.textContent = String(e);
          }
        }

        async function deleteSelectedRunModel() {
          const rid = selectedRunId();
          if (!rid) {
            alert('run_id를 선택하세요.');
            return;
          }
          if (rid === '__default__') {
            alert('__default__ 모델은 삭제할 수 없습니다.');
            return;
          }
          if (!confirm(`선택 RUN 모델을 삭제할까요?\\nrun_id=${rid}\\n(artifacts/experiments/${rid} 삭제)`)) {
            return;
          }
          deleteStatus.textContent = 'deleting...';
          deleteOut.textContent = '';
          try {
            const res = await fetch('/runs/delete', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ run_id: rid }),
            });
            const text = await res.text();
            let json = null;
            try {
              json = JSON.parse(text);
              deleteOut.textContent = JSON.stringify(json, null, 2);
            } catch (_e) {
              deleteOut.textContent = text;
            }
            deleteStatus.textContent = `done (HTTP ${res.status})`;
            await loadRuns();
            if (json && json.ok) {
              closeModelDetail();
            }
          } catch (e) {
            deleteStatus.textContent = 'error';
            deleteOut.textContent = String(e);
          }
        }

        function openModelDetail() {
          detailModal.classList.add('open');
          loadModelDetail();
        }

        function closeModelDetail() {
          detailModal.classList.remove('open');
        }

        document.getElementById('refreshRuns').addEventListener('click', () => loadRuns());
        runSelect.addEventListener('change', () => syncRunMeta());
        document.getElementById('openResults').addEventListener('click', () => {
          const rid = selectedRunId();
          if (!rid) return;
          location.href = `/results?run_id=${encodeURIComponent(rid)}`;
        });
        document.getElementById('openRunPlayground').addEventListener('click', () => {
          const rid = selectedRunId();
          if (!rid) return;
          location.href = `/playground?run_id=${encodeURIComponent(rid)}`;
        });
        document.getElementById('openModelDetail').addEventListener('click', openModelDetail);
        document.getElementById('deleteRunModel').addEventListener('click', deleteSelectedRunModel);
        document.getElementById('startTrain').addEventListener('click', startTraining);
        document.getElementById('next1').addEventListener('click', () => gotoStep(2));

        document.getElementById('runPreTest').addEventListener('click', runPreTest);
        document.getElementById('openStep2Playground').addEventListener('click', () => {
          const rid = selectedRunId();
          if (!rid) return;
          location.href = `/playground?run_id=${encodeURIComponent(rid)}`;
        });
        document.getElementById('next2').addEventListener('click', () => gotoStep(3));

        document.getElementById('approveReview').addEventListener('click', () => saveReview(true));
        document.getElementById('revokeReview').addEventListener('click', () => saveReview(false));
        document.getElementById('dryFinalize').addEventListener('click', () => runFinalize(true));
        document.getElementById('applyFinalize').addEventListener('click', () => runFinalize(false));
        document.getElementById('next3').addEventListener('click', () => gotoStep(4));

        document.getElementById('runPostTest').addEventListener('click', runPostTest);
        document.getElementById('openLivePlayground').addEventListener('click', () => {
          if (currentRunId) {
            location.href = `/playground?run_id=${encodeURIComponent(currentRunId)}`;
            return;
          }
          location.href = '/playground';
        });
        document.getElementById('next4').addEventListener('click', async () => {
          const ok = await ensureLivePromotionBeforeStep5();
          if (!ok) {
            return;
          }
          setDone(5, true);
          gotoStep(5);
          await refreshLiveState(true);
        });

        document.getElementById('refreshLiveState').addEventListener('click', () => refreshLiveState(true));
        document.getElementById('openDashboard').addEventListener('click', () => location.href = '/dashboard');
        document.getElementById('refreshModelDetail').addEventListener('click', loadModelDetail);
        document.getElementById('closeModelDetail').addEventListener('click', closeModelDetail);
        detailModal.addEventListener('click', (e) => {
          if (e.target === detailModal) closeModelDetail();
        });
        document.addEventListener('keydown', (e) => {
          if (e.key === 'Escape' && detailModal.classList.contains('open')) closeModelDetail();
        });

        document.getElementById('preInput').value = JSON.stringify({
          domain: "oms",
          system: "OMS",
          message: "User with id admin access to ip 10.0.0.225 and checked the/dw/main/mainPageSD",
          create_date: "2026-04-09 11:04:19",
          state: "PAGE"
        }, null, 2);
        document.getElementById('postInput').value = document.getElementById('preInput').value;

        const AUTO_RECONNECT_LAST_JOB = false; // keep(auto reconnect) OFF by default
        initTrainRunIdDefault();
        resetTrainMonitor();
        loadRuns();
        if (AUTO_RECONNECT_LAST_JOB) {
          // Auto reconnect to last job if page was reopened.
          try {
            const lastJob = (localStorage.getItem('last_train_job_id') || '').trim();
            if (lastJob) {
              // Only reconnect if job is still active.
              fetch(`/train/jobs/${encodeURIComponent(lastJob)}`).then(r => r.json()).then(j => {
                if (j && (j.status === 'queued' || j.status === 'running')) {
                  watchTrainingJobSSE(lastJob, selectedRunId());
                }
              }).catch(_ => {});
            }
          } catch (_) {}
        }
      </script>
    </body>
    </html>
    """
    html = html.replace('value="__DEFAULT_TRAIN_RUN_ID__"', f'value="{default_train_run_id}"')
    html = html.replace("__WIZARD_RUN_OPTIONS__", run_options_html)
    return HTMLResponse(html, headers=NO_STORE_HEADERS)


@app.get("/train")
def train_page():
    jobs = list_jobs(limit=20)
    rows = "\n".join(
        f"<tr><td>{j.job_id}</td><td><code>{j.run_id}</code></td><td>{j.status}</td><td><code>{j.input_path}</code></td>"
        f"<td><a href='/train/jobs/{j.job_id}'>status</a></td>"
        f"<td><a href='/results?run_id={j.run_id}'>results</a></td>"
        f"<td><a href='/train/jobs/{j.job_id}/log' target='_blank'>log</a></td></tr>"
        for j in jobs
    )
    html = f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Train</title>
      <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
        .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin-bottom: 14px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        td, th {{ border-bottom: 1px solid #eee; padding: 8px; text-align: left; }}
        code {{ background:#f6f6f6; padding: 2px 4px; border-radius: 4px; }}
        .muted {{ color:#666; }}
      </style>
    </head>
    <body>
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:baseline; gap:12px;">
          <h2 style="margin:0">학습 파일 업로드</h2>
          <div style="display:flex; gap:10px;">
            <a href="/dashboard">dashboard</a>
            <a href="/results">results</a>
            <a href="/wizard">wizard</a>
            <a href="/release">release</a>
            <a href="/playground">playground</a>
            <a href="/commands">commands</a>
          </div>
        </div>
        <p class="muted" style="margin-top:6px">업로드 후 백그라운드에서 학습이 실행되며 완료 시 알림이 뜹니다.</p>
        <form id="uploadForm">
          <label>run_id (비우면 자동 생성)</label>
          <input type="text" name="run_id" placeholder="run_20260414_103000" />
          <input type="file" name="file" accept=".jsonl,application/json" required />
          <button type="submit">업로드 & 학습 시작</button>
        </form>
        <pre id="result" style="margin-top:10px"></pre>
        <p class="muted" style="margin-top:10px; margin-bottom:0">
          권장 순서: <a href="/wizard">wizard</a>에서 1~5단계 진행 (학습 → 학습결과 테스트 → 반영 → 반영 후 테스트 → 대시보드 확인)
        </p>
      </div>

      <div class="card">
        <h3 style="margin:0 0 10px 0">최근 학습 잡</h3>
        <table>
          <thead><tr><th>job</th><th>run_id</th><th>status</th><th>input</th><th></th><th></th><th></th></tr></thead>
          <tbody>{rows or "<tr><td colspan='7' class='muted'>아직 없음</td></tr>"}</tbody>
        </table>
      </div>

      <script>
        const form = document.getElementById('uploadForm');
        const result = document.getElementById('result');

        function watch(jobId) {{
          const started = Date.now();
          result.textContent = 'connecting...';
          document.title = 'Train: running';
          const es = new EventSource(`/train/jobs/${{jobId}}/events`);
          es.addEventListener('status', (e) => {{
            try {{
              const j = JSON.parse(e.data);
              result.textContent = JSON.stringify(j, null, 2);
              if (j.status) document.title = `Train: ${{j.status}}`;
            }} catch (_e) {{}}
          }});
          es.addEventListener('log', (e) => {{
            const cur = result.textContent || '';
            const next = (cur ? (cur + \"\\n\") : \"\") + e.data;
            result.textContent = next.length > 200000 ? next.slice(next.length - 200000) : next;
          }});
          es.addEventListener('done', (e) => {{
            const status = (e.data || '').trim();
            try {{ es.close(); }} catch (_e) {{}}
            alert(`학습 완료: ${{status}} (job=${{jobId}})`);
            document.title = `Train: ${{status}}`;
          }});
          es.onerror = () => {{
            es.close();
            // Fallback: slow polling with backoff.
            let delay = 1000;
            (async function pollFallback() {{
              while (true) {{
                try {{
                  const r = await fetch(`/train/jobs/${{jobId}}`);
                  const j = await r.json();
                  result.textContent = JSON.stringify(j, null, 2);
                  if (j.status) document.title = `Train: ${{j.status}}`;
                  if (j.status === 'succeeded' || j.status === 'failed') {{
                    alert(`학습 완료: ${{j.status}} (job=${{jobId}})`);
                    return;
                  }}
                }} catch (_e) {{}}
                if (Date.now() - started > 1000 * 60 * 60 * 12) {{
                  alert('학습 상태 조회 타임아웃(12h).');
                  return;
                }}
                await new Promise(res => setTimeout(res, delay));
                delay = Math.min(30000, Math.floor(delay * 1.6));
              }}
            }})();
          }};
        }}

        form.addEventListener('submit', async (e) => {{
          e.preventDefault();
          result.textContent = 'uploading...';
          const fd = new FormData(form);
          const r = await fetch('/train/upload', {{ method: 'POST', body: fd }});
          const j = await r.json();
          result.textContent = JSON.stringify(j, null, 2);
          if (j.job_id) {{
            try {{ localStorage.setItem('last_train_job_id', String(j.job_id)); }} catch (_e) {{}}
            watch(j.job_id);
          }} else {{
            alert('학습 시작 실패');
          }}
        }});

        const AUTO_RECONNECT_LAST_JOB = false; // keep(auto reconnect) OFF by default
        if (AUTO_RECONNECT_LAST_JOB) {{
          // Auto reconnect when reopening the page.
          try {{
            const lastJob = (localStorage.getItem('last_train_job_id') || '').trim();
            if (lastJob) {{
              fetch(`/train/jobs/${{encodeURIComponent(lastJob)}}`).then(r => r.json()).then(j => {{
                if (j && (j.status === 'queued' || j.status === 'running')) {{
                  watch(lastJob);
                }}
              }}).catch(_ => {{}});
            }}
          }} catch (_e) {{}}
        }}
      </script>
    </body>
    </html>
    """
    # This endpoint is an f-string; normalize any doubled braces that break CSS/JS.
    html = html.replace("{{", "{").replace("}}", "}")
    return HTMLResponse(html, headers=NO_STORE_HEADERS)


@app.post("/train/upload")
async def train_upload(
    file: UploadFile = File(...),
    run_id: Optional[str] = Form(None),
):
    root = _repo_root()
    rid = _sanitize_run_id_or_default(run_id)
    if rid == DEFAULT_RUN_ID:
        rid = make_timestamp_run_id()
    safe_run_dir = os.path.join(root, "data", "samples", rid)
    os.makedirs(safe_run_dir, exist_ok=True)

    # Save uploaded file (streaming) to avoid loading large datasets fully in memory.
    safe_name = os.path.basename(file.filename or "uploaded.jsonl")
    ts = str(int(time.time()))
    out_rel = os.path.join("data", "samples", rid, f"{ts}_{safe_name}")
    out_abs = os.path.join(root, out_rel)
    total_bytes = 0
    try:
        with open(out_abs, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                f.write(chunk)
                total_bytes += len(chunk)
    finally:
        try:
            await file.close()
        except Exception:
            pass
    if total_bytes <= 0:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    train_python = _resolve_train_python(root)

    job = start_training_job(
        repo_root=root,
        input_path=out_rel,
        run_id=rid,
        python_path=train_python,
        scripts_train_path="scripts/train.py",
        caffeinate=True,
    )
    data = job.to_dict()
    data["uploaded_bytes"] = total_bytes
    return data


@app.get("/train/jobs/{job_id}")
def train_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        return {"error": "not_found", "job_id": job_id}
    return job.to_dict()


def _sse_pack(event: str, data: str, *, event_id: str | None = None) -> str:
    # Basic SSE framing: event + data + blank line.
    # data must be a single string; caller should JSON-encode if needed.
    data_lines = (data or "").splitlines() or [""]
    id_line = f"id: {event_id}\n" if event_id else ""
    payload = id_line + f"event: {event}\n" + "\n".join(f"data: {line}" for line in data_lines) + "\n\n"
    return payload


@app.get("/train/jobs/{job_id}/events")
def train_job_events(job_id: str, request: Request):
    job = get_job(job_id)
    if not job:
        return {"error": "not_found", "job_id": job_id}

    def gen():
        import json

        last_status = None
        last_sent_at = 0.0
        pos = 0
        # Resume log streaming from Last-Event-ID (we set id to the byte offset).
        lei = request.headers.get("last-event-id")
        if lei:
            try:
                pos = max(0, int(str(lei).strip()))
            except Exception:
                pos = 0

        # Initial snapshot.
        cur = get_job(job_id)
        if cur:
            last_status = cur.status
            yield _sse_pack("status", json.dumps(cur.to_dict(), ensure_ascii=False))

        # Stream updates until terminal state.
        while True:
            cur = get_job(job_id)
            if not cur:
                yield _sse_pack("status", json.dumps({"error": "not_found", "job_id": job_id}, ensure_ascii=False))
                yield _sse_pack("done", "not_found")
                return

            # Status change push.
            if cur.status != last_status:
                last_status = cur.status
                yield _sse_pack("status", json.dumps(cur.to_dict(), ensure_ascii=False))

            # Log tail push (append-only).
            try:
                if cur.log_path and os.path.exists(cur.log_path):
                    size = os.path.getsize(cur.log_path)
                    if size < pos:
                        # log rotated/truncated
                        pos = 0
                    if size > pos:
                        with open(cur.log_path, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(pos)
                            chunk = f.read(size - pos)
                            pos = f.tell()
                        if chunk:
                            yield _sse_pack("log", chunk.rstrip("\n"), event_id=str(pos))
            except Exception:
                pass

            now = time.time()
            if now - last_sent_at > 15:
                last_sent_at = now
                yield _sse_pack("ping", str(int(now)))

            if cur.status in {"succeeded", "failed"}:
                # Final status snapshot to ensure client sees terminal state.
                yield _sse_pack("status", json.dumps(cur.to_dict(), ensure_ascii=False))
                yield _sse_pack("done", cur.status)
                return

            time.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/train/jobs/{job_id}/log")
def train_job_log(job_id: str):
    job = get_job(job_id)
    if not job:
        return {"error": "not_found", "job_id": job_id}
    if not os.path.exists(job.log_path):
        return {"error": "log_not_found", "job_id": job_id, "log_path": job.log_path}
    from fastapi.responses import PlainTextResponse

    with open(job.log_path, "r", encoding="utf-8", errors="replace") as f:
        return PlainTextResponse(f.read())

@app.get("/health")
def health():
    return {"status": "ok"}


def _sanitize_infer_input(input_record: dict | str):
    if isinstance(input_record, dict):
        return {
            k: v
            for k, v in input_record.items()
            if k not in {"command", "reason", "accuracy"}
        }
    return input_record


@app.post("/infer")
def infer(
    input_record: dict | str = Body(...),
    run_id: str = Query(DEFAULT_RUN_ID),
):
    rid = _sanitize_run_id_or_default(run_id)
    try:
        target_engine = _get_engine_for_run(rid)
    except Exception as e:
        RUN_ENGINE_LOAD_ERRORS[rid] = repr(e)
        return _safe_fail_response(f"run_id model load failed: {rid} ({repr(e)})")
    if target_engine is None:
        load_err = RUN_ENGINE_LOAD_ERRORS.get(rid)
        if load_err:
            return _safe_fail_response(f"run_id model load failed: {rid} ({load_err})")
        return _safe_fail_response(f"run_id model not found: {rid}")
    try:
        return target_engine.predict(_sanitize_infer_input(input_record))
    except Exception:
        return _safe_fail_response(FAIL_SAFE_REASON)


@app.get("/run-files/{run_id}/{name}")
def run_files(run_id: str, name: str):
    rid, model_dir, _ = _resolve_run_dirs(run_id)
    safe_name = os.path.basename(name)
    full = os.path.abspath(os.path.join(model_dir, safe_name))
    model_abs = os.path.abspath(model_dir)
    if not full.startswith(model_abs + os.sep) and full != model_abs:
        return {"error": "invalid_path"}
    if not os.path.exists(full):
        return {"error": "not_found", "run_id": rid, "name": safe_name}
    return FileResponse(full)


@app.get("/commands")
def commands_page():
    commands = load_commands(COMMANDS_PATH)
    rows = "\n".join(
        f"<tr><td><code>{c.get('command')}</code></td>"
        f"<td>{c.get('reason_template','')}</td>"
        f"<td>{', '.join(c.get('aliases', []) or [])}</td>"
        f"<td><button onclick=\"deleteCmd('{c.get('command')}')\">delete</button></td></tr>"
        for c in commands
    )
    html = f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Command Catalog</title>
      <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
        .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin-bottom: 14px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        td, th {{ border-bottom: 1px solid #eee; padding: 8px; text-align: left; vertical-align: top; }}
        code {{ background:#f6f6f6; padding: 2px 4px; border-radius: 4px; }}
        input {{ width: 100%; box-sizing: border-box; padding: 8px; margin-bottom: 8px; }}
        .muted {{ color:#666; }}
      </style>
    </head>
    <body>
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:baseline; gap:12px;">
          <h2 style="margin:0">Command 관리</h2>
          <div style="display:flex; gap:10px;">
            <a href="/dashboard">dashboard</a>
            <a href="/wizard">wizard</a>
            <a href="/train">train</a>
          </div>
        </div>
        <p class="muted">입력에는 command를 넣지 않고, 예측 시 이 목록 중 하나의 command가 출력됩니다.</p>
      </div>

      <div class="card">
        <h3 style="margin:0 0 10px 0">Command 추가/수정</h3>
        <form id="upsertForm">
          <label>command (UPPER_SNAKE_CASE)</label>
          <input name="command" placeholder="TOP_IP_BY_PAGE" required />
          <label>reason_template</label>
          <input name="reason_template" placeholder="identify the most frequent IP for the requested page" required />
          <label>aliases (comma separated)</label>
          <input name="aliases_csv" placeholder="top ip by page, most viewed ip" />
          <button type="submit">save</button>
        </form>
        <pre id="result"></pre>
      </div>

      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h3 style="margin:0 0 10px 0">현재 Command 목록 ({len(commands)}개)</h3>
          <button onclick="resetDefaults()">reset default 8</button>
        </div>
        <table>
          <thead><tr><th>command</th><th>reason_template</th><th>aliases</th><th></th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>

      <script>
        const result = document.getElementById('result');
        document.getElementById('upsertForm').addEventListener('submit', async (e) => {{
          e.preventDefault();
          const fd = new FormData(e.target);
          const r = await fetch('/commands/upsert', {{ method: 'POST', body: fd }});
          const j = await r.json();
          result.textContent = JSON.stringify(j, null, 2);
          if (j.ok) location.reload();
        }});

        async function deleteCmd(command) {{
          if (!confirm(`delete ${{command}} ?`)) return;
          const fd = new FormData();
          fd.append('command', command);
          const r = await fetch('/commands/delete', {{ method: 'POST', body: fd }});
          const j = await r.json();
          result.textContent = JSON.stringify(j, null, 2);
          if (j.ok) location.reload();
        }}

        async function resetDefaults() {{
          if (!confirm('reset to default 8 commands?')) return;
          const r = await fetch('/commands/reset', {{ method: 'POST' }});
          const j = await r.json();
          result.textContent = JSON.stringify(j, null, 2);
          if (j.ok) location.reload();
        }}
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/commands/list")
def commands_list():
    return {"commands": load_commands(COMMANDS_PATH)}


@app.post("/commands/upsert")
def commands_upsert(
    command: str = Form(...),
    reason_template: str = Form(""),
    aliases_csv: str = Form(""),
):
    rows = load_commands(COMMANDS_PATH)
    aliases = [x.strip() for x in aliases_csv.split(",") if x.strip()]
    entry = {
        "command": command,
        "reason_template": reason_template,
        "aliases": aliases,
    }
    cmd_upper = command.strip().upper()
    updated = False
    for i, row in enumerate(rows):
        if str(row.get("command", "")).upper() == cmd_upper:
            rows[i] = entry
            updated = True
            break
    if not updated:
        rows.append(entry)
    save_commands(COMMANDS_PATH, rows)
    return {"ok": True, "updated": updated, "saved_path": COMMANDS_PATH, "count": len(load_commands(COMMANDS_PATH))}


@app.post("/commands/delete")
def commands_delete(command: str = Form(...)):
    rows = load_commands(COMMANDS_PATH)
    cmd_upper = command.strip().upper()
    new_rows = [x for x in rows if str(x.get("command", "")).upper() != cmd_upper]
    save_commands(COMMANDS_PATH, new_rows)
    return {
        "ok": True,
        "deleted": len(rows) - len(new_rows),
        "saved_path": COMMANDS_PATH,
        "count": len(load_commands(COMMANDS_PATH)),
    }


@app.post("/commands/reset")
def commands_reset():
    save_commands(COMMANDS_PATH, DEFAULT_COMMANDS)
    return {"ok": True, "saved_path": COMMANDS_PATH, "count": len(load_commands(COMMANDS_PATH))}
