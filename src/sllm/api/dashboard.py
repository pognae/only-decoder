import json
import math
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from fastapi.responses import HTMLResponse

from sllm.api.training_jobs import list_jobs
from sllm.common.experiments import DEFAULT_RUN_ID, build_run_paths


def _read_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_last_jsonl(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        last = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if not last:
            return None
        return json.loads(last)
    except Exception:
        return None


def _read_text_if_exists(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = (f.read() or "").strip()
        return text or None
    except Exception:
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        if v.is_integer():
            return f"{int(v):,}"
        abs_v = abs(v)
        if abs_v >= 1:
            s = f"{v:,.6f}".rstrip("0").rstrip(".")
            return "0" if s == "-0" else s
        s = f"{v:.12f}".rstrip("0").rstrip(".")
        if s in {"", "-0"}:
            return "0"
        return s
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(val)}" for k, val in v.items()) + "}"
    if isinstance(v, (list, tuple, set)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _fmt_ts(epoch: Any) -> str:
    try:
        if epoch is None:
            return "-"
        return datetime.fromtimestamp(float(epoch)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _mtime(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fsize(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        n = os.path.getsize(path)
    except Exception:
        return None
    if n < 1024:
        return f"{n:,} B"
    if n < 1024 * 1024:
        return f"{n / 1024:,.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):,.1f} MB"
    return f"{n / (1024 * 1024 * 1024):,.2f} GB"


def _format_param_count(n: Optional[int]) -> str:
    if n is None:
        return "-"
    raw = f"{int(n):,}"
    val = float(n)
    if val >= 1_000_000_000:
        short = f"{val / 1_000_000_000:.2f}B"
    elif val >= 1_000_000:
        short = f"{val / 1_000_000:.2f}M"
    elif val >= 1_000:
        short = f"{val / 1_000:.2f}K"
    else:
        short = str(int(n))
    return f"{raw} ({short})"


def _numel_from_shape(shape: Any) -> int:
    n = 1
    for d in shape or []:
        n *= int(d)
    return int(n)


def _safetensors_param_count(path: str) -> Optional[int]:
    if not path or not os.path.exists(path):
        return None
    try:
        from safetensors import safe_open  # type: ignore

        total = 0
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                try:
                    shape = f.get_slice(key).get_shape()
                    total += _numel_from_shape(shape)
                except Exception:
                    tensor = f.get_tensor(key)
                    total += int(getattr(tensor, "numel", lambda: 0)())
        return total
    except Exception:
        return None


def _estimate_llama_param_count_from_config(config: Dict[str, Any]) -> Optional[int]:
    try:
        vocab_size = int(config.get("vocab_size"))
        hidden_size = int(config.get("hidden_size"))
        intermediate_size = int(config.get("intermediate_size"))
        num_hidden_layers = int(config.get("num_hidden_layers"))
        num_attention_heads = int(config.get("num_attention_heads"))
        num_key_value_heads = int(config.get("num_key_value_heads", num_attention_heads))
    except Exception:
        return None

    if num_attention_heads <= 0 or hidden_size <= 0 or num_hidden_layers <= 0:
        return None

    if hidden_size % num_attention_heads == 0:
        head_dim = hidden_size // num_attention_heads
    else:
        head_dim = None

    # Llama-style attention/MLP layout with optional bias terms.
    q_proj = hidden_size * hidden_size
    if head_dim is not None:
        kv_dim = num_key_value_heads * head_dim
        k_proj = hidden_size * kv_dim
        v_proj = hidden_size * kv_dim
    else:
        k_proj = hidden_size * hidden_size
        v_proj = hidden_size * hidden_size
        kv_dim = hidden_size
    o_proj = hidden_size * hidden_size
    attn = q_proj + k_proj + v_proj + o_proj
    mlp = (hidden_size * intermediate_size) * 3  # gate, up, down
    norms = hidden_size * 2  # input + post_attention
    per_layer = attn + mlp + norms

    attention_bias = bool(config.get("attention_bias", False))
    if attention_bias:
        per_layer += (hidden_size * 2) + (kv_dim * 2)  # q,o + k,v biases

    mlp_bias = bool(config.get("mlp_bias", False))
    if mlp_bias:
        per_layer += (intermediate_size * 2) + hidden_size  # gate,up,down biases

    embed = vocab_size * hidden_size
    final_norm = hidden_size
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
    lm_head = 0 if tie_word_embeddings else (hidden_size * vocab_size)

    return int(embed + (num_hidden_layers * per_layer) + final_norm + lm_head)


def _model_param_count(model_dir: str, config: Dict[str, Any]) -> Tuple[Optional[int], str]:
    model_dir_abs = os.path.abspath(model_dir)
    single = os.path.join(model_dir_abs, "model.safetensors")
    n = _safetensors_param_count(single)
    if n is not None:
        return n, "model.safetensors"

    index_path = os.path.join(model_dir_abs, "model.safetensors.index.json")
    index = _read_json_if_exists(index_path) or {}
    weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
    if isinstance(weight_map, dict) and weight_map:
        shard_files = sorted({str(v) for v in weight_map.values() if str(v).strip()})
        total = 0
        used = 0
        for shard in shard_files:
            shard_path = os.path.join(model_dir_abs, shard)
            s = _safetensors_param_count(shard_path)
            if s is None:
                continue
            total += s
            used += 1
        if used > 0:
            return total, f"model.safetensors shards ({used})"

    est = _estimate_llama_param_count_from_config(config)
    if est is not None:
        return est, "config_estimate"
    return None, "-"


def _top_k(d: Dict[str, Any], k: int = 12) -> Tuple[list[tuple[str, Any]], int]:
    items = list(d.items())
    try:
        items.sort(key=lambda x: x[1], reverse=True)
    except Exception:
        items.sort(key=lambda x: str(x[0]))
    shown = items[:k]
    remaining = max(0, len(items) - len(shown))
    return shown, remaining


def _render_kv_table(title: str, rows: list[tuple[str, Any]]) -> str:
    tds = "\n".join(
        f"<tr><td class='k'>{k}</td><td class='v'>{_fmt(v)}</td></tr>" for k, v in rows
    )
    return f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <table class="kv">
        {tds}
      </table>
    </div>
    """


def _render_dist_section(label: str, section: Dict[str, Any]) -> str:
    rows = [
        ("rows", section.get("rows")),
        ("labeled_rows", section.get("labeled_rows")),
    ]
    cmd_items, cmd_more = _top_k(section.get("command_distribution") or {}, k=12)
    sys_items, sys_more = _top_k(section.get("system_distribution") or {}, k=12)
    state_items, state_more = _top_k(section.get("state_distribution") or {}, k=12)
    domain_items, domain_more = _top_k(section.get("domain_distribution") or {}, k=12)
    msglen_items, msglen_more = _top_k(section.get("message_length_bucket_distribution") or {}, k=12)

    def _render_list(title: str, items: list[tuple[str, Any]], more: int) -> str:
        lis = "\n".join(f"<li><span class='k'>{k}</span><span class='v'>{_fmt(v)}</span></li>" for k, v in items)
        more_html = f"<li class='more'>… 외 {more}개</li>" if more else ""
        return f"""
        <div class="dist-card">
          <div class="dist-title">{title}</div>
          <ul class="dist-list">
            {lis}
            {more_html}
          </ul>
        </div>
        """

    return f"""
    <div class="card">
      <div class="card-title">데이터 분포 ({label})</div>
      <table class="kv">
        {"".join(f"<tr><td class='k'>{k}</td><td class='v'>{_fmt(v)}</td></tr>" for k, v in rows)}
      </table>
      <div class="dist-grid">
        {_render_list("command", cmd_items, cmd_more)}
        {_render_list("system", sys_items, sys_more)}
        {_render_list("state", state_items, state_more)}
        {_render_list("domain", domain_items, domain_more)}
        {_render_list("message_length_bucket", msglen_items, msglen_more)}
      </div>
    </div>
    """


def build_dashboard_html(
    model_dir: str,
    tokenizer_dir: Optional[str],
    *,
    current_run_id: Optional[str] = None,
    experiments_root: Optional[str] = None,
) -> HTMLResponse:
    model_dir_abs = os.path.abspath(model_dir)
    tok_dir_abs = os.path.abspath(tokenizer_dir) if tokenizer_dir else model_dir_abs

    # Live (finalized) run metadata
    releases_dir = os.path.abspath(os.path.join(os.path.dirname(model_dir_abs), "releases"))
    current_run_marker = os.path.join(releases_dir, "current_run_id.txt")
    current_run_from_marker = _read_text_if_exists(current_run_marker)
    live_run_id = (current_run_id or current_run_from_marker or DEFAULT_RUN_ID).strip()

    promotion_record_path = os.path.join(model_dir_abs, "promotion_record.json")
    promotion_record = _read_json_if_exists(promotion_record_path) or {}
    promotion_history_path = os.path.join(releases_dir, "promotion_history.jsonl")
    promotion_history_last = _read_last_jsonl(promotion_history_path) or {}
    review_approval_path = os.path.join(model_dir_abs, "review_approval.json")
    review_approval = _read_json_if_exists(review_approval_path) or {}

    source_run_model_dir = None
    source_run_tokenizer_dir = None
    if live_run_id != DEFAULT_RUN_ID and experiments_root:
        try:
            rp = build_run_paths(live_run_id, root_dir=experiments_root)
            source_run_model_dir = os.path.abspath(rp.model_dir)
            source_run_tokenizer_dir = os.path.abspath(rp.tokenizer_dir)
        except Exception:
            pass

    # Core live model files
    training_result_path = os.path.join(model_dir_abs, "training_result.json")
    train_metrics_path = os.path.join(model_dir_abs, "train_metrics.json")
    metrics_path = os.path.join(model_dir_abs, "metrics.json")
    reason_coverage_path = os.path.join(model_dir_abs, "reason_coverage_report.json")
    quality_path = os.path.join(model_dir_abs, "data_quality_report.json")
    dist_json_path = os.path.join(model_dir_abs, "data_distribution_report.json")
    dist_html_path = os.path.join(model_dir_abs, "data_distribution_report.html")
    config_path = os.path.join(model_dir_abs, "config.json")
    tokenizer_json_path = os.path.join(tok_dir_abs, "tokenizer.json")
    model_weights_path = os.path.join(model_dir_abs, "model.safetensors")
    run_meta_path = os.path.join(model_dir_abs, "run_meta.json")

    training_result = _read_json_if_exists(training_result_path) or {}
    train_metrics = _read_json_if_exists(train_metrics_path) or {}
    metrics = _read_json_if_exists(metrics_path) or {}
    reason_coverage = _read_json_if_exists(reason_coverage_path) or {}
    quality = _read_json_if_exists(quality_path) or {}
    dist = _read_json_if_exists(dist_json_path) or {}
    config = _read_json_if_exists(config_path) or {}
    run_meta = _read_json_if_exists(run_meta_path) or {}

    param_target_model_dir = source_run_model_dir if source_run_model_dir and os.path.isdir(source_run_model_dir) else model_dir_abs
    param_count, param_count_source = _model_param_count(param_target_model_dir, config if isinstance(config, dict) else {})

    summary = (training_result or {}).get("summary", {}) if isinstance(training_result, dict) else {}
    rc = reason_coverage.get("reason_coverage", {}) if isinstance(reason_coverage, dict) else {}
    cs = reason_coverage.get("command_shape", {}) if isinstance(reason_coverage, dict) else {}
    quality_checks = quality.get("checks", {}) if isinstance(quality, dict) else {}
    quality_required = quality_checks.get("required_columns", {}) if isinstance(quality_checks, dict) else {}
    quality_dups = quality_checks.get("duplicates", {}) if isinstance(quality_checks, dict) else {}
    quality_conf = quality_checks.get("conflicts", {}) if isinstance(quality_checks, dict) else {}

    # Distribution (LIVE model only, no fallback)
    train_dist = (dist.get("train") or {}) if isinstance(dist, dict) else {}
    valid_dist = (dist.get("valid") or {}) if isinstance(dist, dict) else {}
    dist_missing_hint = ""
    if not train_dist and not valid_dist:
        dist_missing_hint = (
            "<div class='card'>"
            "<div class='card-title'>데이터 분포 안내</div>"
            f"<div class='muted'>LIVE 모델 경로(<code>{model_dir_abs}</code>)에 분포 리포트가 없습니다. "
            f"`make dist TRAIN_FILE=train_data/train.jsonl MODEL_DIR={model_dir_abs}` 실행 후 새로고침하세요.</div>"
            "</div>"
        )
    dist_html = "\n".join([_render_dist_section("train", train_dist), _render_dist_section("valid", valid_dist)])

    # Checkpoints
    checkpoints = sorted(
        [x for x in os.listdir(model_dir_abs) if x.startswith("checkpoint-")] if os.path.isdir(model_dir_abs) else []
    )
    ckpt_html = (
        "<div class='muted'>체크포인트 없음</div>"
        if not checkpoints
        else "<ul class='ckpt'>"
        + "\n".join(f"<li>{c}</li>" for c in checkpoints[-12:])
        + ("<li class='more'>…</li>" if len(checkpoints) > 12 else "")
        + "</ul>"
    )

    # Runtime hardware
    cuda_available = False
    cuda_device_count = None
    cuda_devices = None
    cuda_bf16 = None
    mps_built = None
    mps_available = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available and cuda_device_count:
            cuda_devices = [torch.cuda.get_device_name(i) for i in range(cuda_device_count)]
            if hasattr(torch.cuda, "is_bf16_supported"):
                cuda_bf16 = bool(torch.cuda.is_bf16_supported())
        if hasattr(torch.backends, "mps"):
            mps_built = bool(torch.backends.mps.is_built())
            mps_available = bool(torch.backends.mps.is_available())
    except Exception:
        pass

    runtime_card = _render_kv_table(
        "런타임 디바이스 상태",
        [
            ("python", sys.version.split()[0]),
            ("platform", f"{platform.system()} {platform.release()}"),
            ("cpu", "available"),
            ("cuda_available", cuda_available),
            ("cuda_device_count", cuda_device_count),
            ("cuda_devices", ", ".join(cuda_devices) if cuda_devices else None),
            ("cuda_bf16_supported", cuda_bf16),
            ("mps_built", mps_built),
            ("mps_available", mps_available),
        ],
    )

    # Jobs/training status
    jobs = list_jobs(limit=100)
    jobs_live = [j for j in jobs if str(getattr(j, "run_id", "")) == live_run_id]
    latest_job = jobs[0] if jobs else None
    latest_live_job = jobs_live[0] if jobs_live else None
    running_cnt = sum(1 for j in jobs if j.status == "running")
    queued_cnt = sum(1 for j in jobs if j.status == "queued")
    failed_recent_cnt = sum(1 for j in jobs[:20] if j.status == "failed")

    if running_cnt > 0:
        light = ("running", "학습 실행 중", "yellow")
    elif queued_cnt > 0:
        light = ("queued", "학습 대기 중", "yellow")
    elif failed_recent_cnt > 0:
        light = ("failed", "최근 학습 실패", "red")
    else:
        light = ("idle", "학습 중 아님", "green")

    traffic_light = f"""
      <div class="status">
        <span class="dot {light[2]}"></span>
        <span class="status-text">{light[1]}</span>
        <a class="link status-link" href="/train">train</a>
      </div>
    """

    live_status_card = _render_kv_table(
        "LIVE 모델/학습 상태",
        [
            ("live_run_id", live_run_id),
            ("live_run_marker", os.path.abspath(current_run_marker)),
            ("model_dir (serving target)", model_dir_abs),
            ("tokenizer_dir (serving target)", tok_dir_abs),
            ("source_run_model_dir", source_run_model_dir),
            ("source_run_tokenizer_dir", source_run_tokenizer_dir),
            ("latest_job.status", latest_job.status if latest_job else None),
            ("latest_job.run_id", latest_job.run_id if latest_job else None),
            ("latest_live_job.status", latest_live_job.status if latest_live_job else None),
            ("latest_live_job.created_at", _fmt_ts(latest_live_job.created_at) if latest_live_job else None),
            ("jobs.running_count", running_cnt),
            ("jobs.queued_count", queued_cnt),
            ("jobs.failed_recent(20)", failed_recent_cnt),
        ],
    )

    promotion_card = _render_kv_table(
        "최종 반영 정보 (Promotion)",
        [
            ("promotion_record.run_id", promotion_record.get("run_id")),
            ("promotion_record.promoted_at_utc", promotion_record.get("promoted_at_utc")),
            ("promotion_record.source_model_dir", promotion_record.get("source_model_dir")),
            ("promotion_record.target_model_dir", promotion_record.get("target_model_dir")),
            ("promotion_history_last.run_id", promotion_history_last.get("run_id")),
            ("promotion_history_last.promoted_at_utc", promotion_history_last.get("promoted_at_utc")),
            ("review_approval.approved", review_approval.get("approved")),
            ("review_approval.reviewer", review_approval.get("reviewer")),
            ("review_approval.approved_at_utc", review_approval.get("approved_at_utc")),
        ],
    )

    config_card = _render_kv_table(
        "모델 구성 (config/run_meta)",
        [
            ("selected_run_id", live_run_id),
            ("selected_run.parameter_count", _format_param_count(param_count)),
            ("selected_run.parameter_count_source", param_count_source),
            ("selected_run.model_dir", param_target_model_dir),
            ("config.vocab_size", config.get("vocab_size")),
            ("config.hidden_size", config.get("hidden_size")),
            ("config.intermediate_size", config.get("intermediate_size")),
            ("config.num_hidden_layers", config.get("num_hidden_layers")),
            ("config.num_attention_heads", config.get("num_attention_heads")),
            ("config.max_position_embeddings", config.get("max_position_embeddings")),
            ("run_meta.run_id", run_meta.get("run_id")),
            ("run_meta.input_source", run_meta.get("input_source")),
            ("run_meta.device", run_meta.get("device")),
            ("run_meta.resume", run_meta.get("resume")),
            ("tokenizer.json.size", _fsize(tokenizer_json_path)),
            ("model.safetensors.size", _fsize(model_weights_path)),
        ],
    )

    files = [
        ("training_result.json", training_result_path),
        ("train_metrics.json", train_metrics_path),
        ("metrics.json", metrics_path),
        ("data_quality_report.json", quality_path),
        ("reason_coverage_report.json", reason_coverage_path),
        ("data_distribution_report.json", dist_json_path),
        ("data_distribution_report.html", dist_html_path),
        ("config.json", config_path),
        ("tokenizer.json", tokenizer_json_path),
        ("model.safetensors", model_weights_path),
        ("run_meta.json", run_meta_path),
        ("review_approval.json", review_approval_path),
        ("promotion_record.json", promotion_record_path),
    ]
    file_rows = "\n".join(
        f"<tr><td class='k'>{name}</td><td class='v'>{'yes' if os.path.exists(path) else 'no'}</td>"
        f"<td class='v'>{_fmt(_mtime(path))}</td><td class='v'>{_fmt(_fsize(path))}</td></tr>"
        for name, path in files
    )
    files_card = f"""
    <div class="card">
      <div class="card-title">LIVE 아티팩트 상태</div>
      <table class="kv">
        <tr><th class='k'>file</th><th class='v'>exists</th><th class='v'>mtime</th><th class='v'>size</th></tr>
        {file_rows}
      </table>
      <div class="muted" style="margin-top:8px;">
        data_distribution_report.html:
        {"<a class='link' href='/model-files/data_distribution_report.html' target='_blank'>열기</a>" if os.path.exists(dist_html_path) else "없음"}
      </div>
    </div>
    """

    metrics_cards = "\n".join(
        [
            _render_kv_table(
                "학습 요약 (training_result)",
                [
                    ("global_step", summary.get("global_step")),
                    ("epoch", summary.get("epoch")),
                    ("max_steps", summary.get("max_steps")),
                    ("train_loss", summary.get("train_loss")),
                    ("train_runtime(sec)", summary.get("train_runtime")),
                    ("eval_loss", summary.get("eval_loss")),
                    ("command_accuracy", summary.get("command_accuracy", metrics.get("command_accuracy"))),
                    ("evaluation_total", summary.get("evaluation_total", metrics.get("total"))),
                ],
            ),
            _render_kv_table(
                "학습 메트릭 (train_metrics.json)",
                [
                    ("train_runtime", train_metrics.get("train_runtime")),
                    ("train_samples_per_second", train_metrics.get("train_samples_per_second")),
                    ("train_steps_per_second", train_metrics.get("train_steps_per_second")),
                    ("total_flos", train_metrics.get("total_flos")),
                    ("train_loss", train_metrics.get("train_loss")),
                    ("epoch", train_metrics.get("epoch")),
                ],
            ),
            _render_kv_table(
                "평가 메트릭 (metrics.json)",
                [
                    ("total", metrics.get("total")),
                    ("command_accuracy", metrics.get("command_accuracy")),
                ],
            ),
        ]
    )

    quality_card = _render_kv_table(
        "데이터 품질 게이트 (LIVE)",
        [
            ("status", quality.get("status")),
            ("fail_reasons", quality.get("fail_reasons")),
            ("warnings", quality.get("warnings")),
            ("required.missing_or_blank_counts", quality_required.get("missing_or_blank_counts")),
            ("duplicates.exact_duplicate_rows", quality_dups.get("exact_duplicate_rows")),
            ("conflicts.input_label_conflicts", quality_conf.get("input_label_conflicts")),
        ],
    )

    reason_card = _render_kv_table(
        "Reason/형태 점검 (LIVE)",
        [
            ("total", reason_coverage.get("total")),
            ("coverage_ratio", rc.get("coverage_ratio")),
            ("missing_key_count", rc.get("missing_key_count")),
            ("blank_count", rc.get("blank_count")),
            ("null_count", rc.get("null_count")),
            ("command_malformed_ratio", cs.get("malformed_ratio")),
            ("command_malformed_count", cs.get("malformed_count")),
        ],
    )

    jobs_rows = "\n".join(
        f"<tr><td>{j.job_id}</td><td><code>{j.run_id}</code></td><td>{j.status}</td>"
        f"<td>{_fmt_ts(j.created_at)}</td><td>{_fmt_ts(j.started_at)}</td><td>{_fmt_ts(j.finished_at)}</td>"
        f"<td>{_fmt(j.exit_code)}</td><td><a class='link' href='/train/jobs/{j.job_id}/log' target='_blank'>log</a></td></tr>"
        for j in jobs[:20]
    )
    jobs_card = f"""
    <div class="card">
      <div class="card-title">학습 잡 상태 (최근 20)</div>
      <table class="kv">
        <tr><th class='k'>job</th><th class='v'>run_id</th><th class='v'>status</th><th class='v'>created</th><th class='v'>started</th><th class='v'>finished</th><th class='v'>exit</th><th class='v'></th></tr>
        {jobs_rows or "<tr><td colspan='8' class='muted'>잡 이력 없음</td></tr>"}
      </table>
      <div class="muted" style="margin-top:8px;">주의: 잡 이력은 서버 프로세스 메모리 기반이므로 재시작 시 초기화될 수 있습니다.</div>
    </div>
    """

    html = f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>sLLM LIVE Dashboard</title>
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
        .wrap {{ max-width: 1200px; margin: 24px auto; padding: 0 16px; }}
        .title {{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; }}
        h1 {{ margin: 0; font-size: 22px; letter-spacing: .2px; }}
        .meta {{ color: var(--muted); font-size: 13px; }}
        .grid {{ display:grid; grid-template-columns: repeat(12, 1fr); gap: 12px; margin-top: 14px; }}
        .card {{
          grid-column: span 12;
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 14px 14px;
          box-shadow: 0 10px 30px rgba(18,24,38,.06);
        }}
        .card-title {{ font-weight: 600; margin-bottom: 10px; }}
        .kv {{ width:100%; border-collapse: collapse; font-size: 13px; }}
        .kv td, .kv th {{ padding: 8px 8px; border-bottom: 1px solid var(--border); vertical-align: top; text-align:left; }}
        .kv tr:last-child td {{ border-bottom: none; }}
        .kv td.k, .kv th.k {{ width: 280px; color: var(--muted); }}
        .kv td.v, .kv th.v {{ color: var(--text); }}
        .cards-3 {{
          display:grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
        }}
        .dist-grid {{
          display:grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 10px;
          margin-top: 12px;
        }}
        .dist-card {{
          background: rgba(18,24,38,.02);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 10px;
        }}
        .dist-title {{ font-size: 12px; color: var(--muted); margin-bottom: 8px; }}
        .dist-list {{ list-style:none; margin:0; padding:0; font-size: 12px; }}
        .dist-list li {{ display:flex; justify-content:space-between; gap:10px; padding: 4px 0; border-bottom: 1px dashed rgba(18,24,38,.10); }}
        .dist-list li:last-child {{ border-bottom: none; }}
        .dist-list .k {{ color: var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width: 180px; }}
        .dist-list .v {{ color: var(--muted); }}
        .muted {{ color: var(--muted); }}
        .link {{ color: var(--accent); text-decoration: none; }}
        .link:hover {{ text-decoration: underline; }}
        .ckpt {{ margin: 8px 0 0 18px; color: var(--muted); }}
        .more {{ color: var(--muted); opacity: .8; }}
        code {{ background: rgba(18,24,38,.06); padding: 2px 6px; border-radius: 6px; }}
        .status {{ display:flex; align-items:center; gap:10px; justify-content:flex-end; }}
        .status-text {{ color: var(--muted); font-size: 13px; }}
        .status-link {{ font-size: 13px; }}
        .dot {{ width: 10px; height: 10px; border-radius: 999px; border: 1px solid rgba(18,24,38,.20); display:inline-block; }}
        .dot.green {{ background: #22c55e; }}
        .dot.yellow {{ background: #f59e0b; }}
        .dot.red {{ background: #ef4444; }}
        @media (max-width: 980px) {{
          .cards-3 {{ grid-template-columns: 1fr; }}
          .dist-grid {{ grid-template-columns: 1fr; }}
          .kv td.k, .kv th.k {{ width: 180px; }}
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="title">
          <h1>sLLM LIVE Dashboard</h1>
          <div style="display:flex; flex-direction:column; align-items:flex-end; gap:6px;">
            {traffic_light}
            <div class="meta">
              live_run_id: <code>{live_run_id}</code>
              &nbsp;&nbsp;model_dir: <code>{model_dir_abs}</code>
            </div>
          </div>
        </div>

        <div class="grid">
          {runtime_card}

          <div class="card">
            <div class="card-title">운영 흐름</div>
            <div style="margin-top:8px"><a class="link" href="/wizard">5단계 Wizard(학습→테스트→반영→확인)</a></div>
            <div style="margin-top:8px"><a class="link" href="/train">학습 파일 업로드 / 실행</a></div>
            <div style="margin-top:8px"><a class="link" href="/results">RUN 비교/검수/최종 반영</a></div>
            <div style="margin-top:8px"><a class="link" href="/release">최종 반영 전용 화면</a></div>
            <div style="margin-top:8px"><a class="link" href="/playground">LIVE 추론 테스트</a></div>
            <div style="margin-top:8px"><a class="link" href="/commands">Command 관리</a></div>
          </div>

          {live_status_card}
          {promotion_card}
          {config_card}
          {files_card}

          <div class="card">
            <div class="card-title">핵심 지표 (LIVE)</div>
            <div class="cards-3">
              {metrics_cards}
            </div>
          </div>

          {quality_card}
          {reason_card}
          <div class="card"><div class="card-title">체크포인트(최근 12개)</div>{ckpt_html}</div>
          {jobs_card}
          {dist_missing_hint}
          {dist_html}
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html)
