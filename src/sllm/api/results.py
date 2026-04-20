import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi.responses import HTMLResponse

from sllm.api.training_jobs import list_jobs
from sllm.common.experiments import DEFAULT_RUN_ID


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _mtime(path: str) -> str:
    if not os.path.exists(path):
        return "-"
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


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


def _file_link(run_id: str, name: str, path: str) -> str:
    if not os.path.exists(path):
        return f"<tr><td class='k'>{name}</td><td class='v muted'>없음</td></tr>"
    rid = quote(run_id, safe="")
    safe_name = quote(os.path.basename(name), safe="")
    return (
        f"<tr><td class='k'>{name}</td>"
        f"<td class='v'><a class='link' href='/run-files/{rid}/{safe_name}' target='_blank'>열기</a>"
        f" <span class='muted'>(mtime: {_mtime(path)})</span></td></tr>"
    )


def _collect_run_payload(model_dir: str) -> Dict[str, Any]:
    training_result_path = os.path.join(model_dir, "training_result.json")
    train_metrics_path = os.path.join(model_dir, "train_metrics.json")
    metrics_path = os.path.join(model_dir, "metrics.json")
    quality_path = os.path.join(model_dir, "data_quality_report.json")
    reason_coverage_path = os.path.join(model_dir, "reason_coverage_report.json")
    dist_json_path = os.path.join(model_dir, "data_distribution_report.json")
    dist_html_path = os.path.join(model_dir, "data_distribution_report.html")
    eval_results_path = os.path.join(model_dir, "eval_results.jsonl")
    infer_result_path = os.path.join(model_dir, "infer_result.json")
    infer_compare_path = os.path.join(model_dir, "infer_compare.json")
    infer_debug_last_path = os.path.join(model_dir, "infer_debug_last.json")
    infer_debug_history_path = os.path.join(model_dir, "infer_debug_history.jsonl")
    report_result_path = os.path.join(model_dir, "report_result.json")
    promotion_record_path = os.path.join(model_dir, "promotion_record.json")
    review_approval_path = os.path.join(model_dir, "review_approval.json")

    training_result = _read_json(training_result_path) or {}
    train_metrics = _read_json(train_metrics_path) or {}
    metrics = _read_json(metrics_path) or {}
    quality = _read_json(quality_path) or {}
    reason_cov = _read_json(reason_coverage_path) or {}
    review = _read_json(review_approval_path) or {}
    summary = training_result.get("summary", {}) if isinstance(training_result, dict) else {}

    return {
        "model_dir": model_dir,
        "summary": summary,
        "train_metrics": train_metrics,
        "metrics": metrics,
        "quality": quality,
        "reason_cov": reason_cov,
        "review": review,
        "files": {
            "training_result.json": training_result_path,
            "train_metrics.json": train_metrics_path,
            "metrics.json": metrics_path,
            "data_quality_report.json": quality_path,
            "eval_results.jsonl": eval_results_path,
            "reason_coverage_report.json": reason_coverage_path,
            "data_distribution_report.json": dist_json_path,
            "data_distribution_report.html": dist_html_path,
            "infer_result.json": infer_result_path,
            "infer_compare.json": infer_compare_path,
            "infer_debug_last.json": infer_debug_last_path,
            "infer_debug_history.jsonl": infer_debug_history_path,
            "report_result.json": report_result_path,
            "review_approval.json": review_approval_path,
            "promotion_record.json": promotion_record_path,
        },
    }


def _run_selector_options(
    rows: List[Dict[str, Any]],
    *,
    selected_run_id: str,
    include_empty: bool = False,
    empty_label: str = "(비교 안 함)",
) -> str:
    out = []
    if include_empty:
        selected = " selected" if selected_run_id == "" else ""
        out.append(f"<option value='' {selected}>{empty_label}</option>")
    for row in rows:
        rid = row.get("run_id")
        label = row.get("label") or rid
        acc = row.get("command_accuracy")
        qs = row.get("quality_status")
        live = " [LIVE]" if row.get("is_current") else ""
        selected = " selected" if rid == selected_run_id else ""
        out.append(f"<option value='{rid}'{selected}>{label}{live} | acc={_fmt(acc)} | quality={_fmt(qs)}</option>")
    return "\n".join(out)


def _compare_row(name: str, left: Any, right: Any) -> str:
    delta = "-"
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        delta = _fmt(left - right)
    return (
        f"<tr><td class='k'>{name}</td><td class='v'>{_fmt(left)}</td>"
        f"<td class='v'>{_fmt(right)}</td><td class='v'>{delta}</td></tr>"
    )


def build_results_html(
    model_dir: str,
    *,
    run_id: str = DEFAULT_RUN_ID,
    compare_model_dir: Optional[str] = None,
    compare_run_id: Optional[str] = None,
    available_runs: Optional[List[Dict[str, Any]]] = None,
    current_promoted_run_id: Optional[str] = None,
) -> HTMLResponse:
    payload = _collect_run_payload(model_dir)
    compare_payload = _collect_run_payload(compare_model_dir) if compare_model_dir else None

    jobs = list_jobs(limit=20)
    latest = jobs[0] if jobs else None
    latest_job_html = (
        "<div class='muted'>최근 학습 잡 없음</div>"
        if not latest
        else f"""
        <table class="kv">
          <tr><td class="k">job_id</td><td class="v">{latest.job_id}</td></tr>
          <tr><td class="k">run_id</td><td class="v"><code>{latest.run_id}</code></td></tr>
          <tr><td class="k">status</td><td class="v">{latest.status}</td></tr>
          <tr><td class="k">command</td><td class="v"><code>{latest.command}</code></td></tr>
          <tr><td class="k">log</td><td class="v"><a class="link" href="/train/jobs/{latest.job_id}/log" target="_blank">열기</a></td></tr>
        </table>
        """
    )

    runs = available_runs or [{"run_id": run_id, "label": run_id}]
    primary_options = _run_selector_options(runs, selected_run_id=run_id)
    compare_options = _run_selector_options(
        runs,
        selected_run_id=compare_run_id or "",
        include_empty=True,
        empty_label="(비교 안 함)",
    )

    summary = payload["summary"]
    train_metrics = payload["train_metrics"]
    metrics = payload["metrics"]
    quality = payload["quality"]
    review = payload["review"]
    quality_checks = quality.get("checks", {}) if isinstance(quality, dict) else {}
    quality_dist = quality_checks.get("distribution", {}) if isinstance(quality_checks, dict) else {}

    compare_card = ""
    if compare_payload:
        cmp_summary = compare_payload["summary"]
        cmp_metrics = compare_payload["metrics"]
        compare_card = f"""
        <div class="card">
          <div class="card-title">RUN 비교 (left={run_id}, right={compare_run_id})</div>
          <table class="kv">
            <tr><th class="k">metric</th><th class="v">left</th><th class="v">right</th><th class="v">delta(left-right)</th></tr>
            {_compare_row("summary.command_accuracy", summary.get("command_accuracy"), cmp_summary.get("command_accuracy"))}
            {_compare_row("summary.evaluation_total", summary.get("evaluation_total"), cmp_summary.get("evaluation_total"))}
            {_compare_row("metrics.command_accuracy", metrics.get("command_accuracy"), cmp_metrics.get("command_accuracy"))}
            {_compare_row("metrics.total", metrics.get("total"), cmp_metrics.get("total"))}
          </table>
        </div>
        """

    file_rows = "\n".join(
        _file_link(run_id, name, path)
        for name, path in payload["files"].items()
    )

    quality_rows = quality_checks.get("required_columns", {}) if isinstance(quality_checks, dict) else {}
    conf = quality_checks.get("conflicts", {}) if isinstance(quality_checks, dict) else {}
    dups = quality_checks.get("duplicates", {}) if isinstance(quality_checks, dict) else {}

    html = f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Training Results</title>
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
        .kv {{ width:100%; border-collapse: collapse; font-size: 13px; }}
        .kv td, .kv th {{ padding: 8px 8px; border-bottom: 1px solid var(--border); vertical-align: top; text-align: left; }}
        .kv tr:last-child td {{ border-bottom: none; }}
        .kv td.k, .kv th.k {{ width: 260px; color: var(--muted); }}
        .kv td.v, .kv th.v {{ color: var(--text); }}
        .muted {{ color: var(--muted); }}
        .link {{ color: var(--accent); text-decoration: none; }}
        .link:hover {{ text-decoration: underline; }}
        .grid2 {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
        select {{ min-width: 320px; padding: 6px; }}
        button {{ padding: 7px 12px; }}
        input {{ min-width: 120px; padding: 6px; }}
        code {{ background: rgba(18,24,38,.06); padding: 2px 6px; border-radius: 6px; }}
        @media (max-width: 980px) {{ .grid2 {{ grid-template-columns: 1fr; }} .kv td.k {{ width: 180px; }} select {{ min-width: 220px; }} }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="top">
          <h1>학습 결과</h1>
          <div class="nav">
            <a class="link" href="/dashboard">dashboard</a>
            <a class="link" href="/wizard">wizard</a>
            <a class="link" href="/playground">playground</a>
            <a class="link" href="/release">release</a>
            <a class="link" href="/train">train</a>
            <a class="link" href="/commands">commands</a>
          </div>
        </div>

        <div class="card">
          <div class="card-title">RUN 선택 / 비교</div>
          <form method="get" action="/results" style="display:flex; gap:10px; flex-wrap: wrap; align-items:center;">
            <label>run_id</label>
            <select name="run_id">{primary_options}</select>
            <label>compare_run_id</label>
            <select name="compare_run_id">{compare_options}</select>
            <button type="submit">적용</button>
          </form>
          <div class="muted" style="margin-top:8px;">model_dir=<code>{os.path.abspath(model_dir)}</code></div>
          <div class="muted" style="margin-top:4px;">current_promoted_run_id=<code>{_fmt(current_promoted_run_id)}</code></div>
          <div class="muted" style="margin-top:6px;">권장 흐름: 품질테스트 → 확인/검수 → 최종 반영 → playground 테스트 → 실제 API 출력(`/infer`)</div>
        </div>

        <div class="card">
          <div class="card-title">확인 및 검수 (Review Approval)</div>
          <table class="kv">
            <tr><td class="k">approved</td><td class="v">{_fmt(review.get("approved"))}</td></tr>
            <tr><td class="k">reviewer</td><td class="v">{_fmt(review.get("reviewer"))}</td></tr>
            <tr><td class="k">approved_at_utc</td><td class="v">{_fmt(review.get("approved_at_utc"))}</td></tr>
            <tr><td class="k">note</td><td class="v">{_fmt(review.get("note"))}</td></tr>
          </table>
          <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:10px;">
            <label>reviewer</label><input id="reviewer" placeholder="예: rms" />
            <label>note</label><input id="reviewNote" style="min-width: 280px;" placeholder="검수 코멘트" />
            <button type="button" id="approveReview">검수 승인 저장</button>
            <button type="button" id="revokeReview">검수 승인 해제</button>
            <span id="reviewStatus" class="muted"></span>
          </div>
          <pre id="reviewResult" style="margin-top:10px; white-space:pre-wrap;">-</pre>
        </div>

        <div class="card">
          <div class="card-title">최종 반영 (Finalize)</div>
          <div class="muted">품질/메트릭 + 검수 승인(기본 필수)을 통과한 RUN만 기본 모델 경로로 반영합니다.</div>
          <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:10px;">
            <label>run_id</label><input id="finalizeRunId" value="{run_id}" />
            <label>min_command_accuracy</label><input id="minCommandAccuracy" placeholder="예: 0.82" />
            <label>min_reason_coverage</label><input id="minReasonCoverage" value="1.0" />
            <label>max_command_malformed_ratio</label><input id="maxMalformedRatio" value="0.0" />
            <label><input type="checkbox" id="requireReviewApproval" checked />검수 승인 필수</label>
            <button type="button" id="dryFinalize">검증만(DRY_RUN)</button>
            <button type="button" id="applyFinalize">최종 반영</button>
            <span id="finalizeStatus" class="muted"></span>
          </div>
          <pre id="finalizeResult" style="margin-top:10px; white-space:pre-wrap;">-</pre>
        </div>

        {compare_card}

        <div class="grid2">
          <div class="card">
            <div class="card-title">요약 (training_result.json)</div>
            <table class="kv">
              <tr><td class="k">global_step</td><td class="v">{_fmt(summary.get("global_step"))}</td></tr>
              <tr><td class="k">epoch</td><td class="v">{_fmt(summary.get("epoch"))}</td></tr>
              <tr><td class="k">max_steps</td><td class="v">{_fmt(summary.get("max_steps"))}</td></tr>
              <tr><td class="k">train_loss</td><td class="v">{_fmt(summary.get("train_loss"))}</td></tr>
              <tr><td class="k">train_runtime</td><td class="v">{_fmt(summary.get("train_runtime"))}</td></tr>
              <tr><td class="k">eval_loss</td><td class="v">{_fmt(summary.get("eval_loss"))}</td></tr>
              <tr><td class="k">command_accuracy</td><td class="v">{_fmt(summary.get("command_accuracy"))}</td></tr>
              <tr><td class="k">evaluation_total</td><td class="v">{_fmt(summary.get("evaluation_total"))}</td></tr>
            </table>
          </div>

          <div class="card">
            <div class="card-title">최근 학습 잡</div>
            {latest_job_html}
          </div>
        </div>

        <div class="grid2">
          <div class="card">
            <div class="card-title">학습 메트릭 (train_metrics.json)</div>
            <table class="kv">
              <tr><td class="k">train_runtime</td><td class="v">{_fmt(train_metrics.get("train_runtime"))}</td></tr>
              <tr><td class="k">train_samples_per_second</td><td class="v">{_fmt(train_metrics.get("train_samples_per_second"))}</td></tr>
              <tr><td class="k">train_steps_per_second</td><td class="v">{_fmt(train_metrics.get("train_steps_per_second"))}</td></tr>
              <tr><td class="k">total_flos</td><td class="v">{_fmt(train_metrics.get("total_flos"))}</td></tr>
              <tr><td class="k">train_loss</td><td class="v">{_fmt(train_metrics.get("train_loss"))}</td></tr>
              <tr><td class="k">epoch</td><td class="v">{_fmt(train_metrics.get("epoch"))}</td></tr>
            </table>
          </div>

          <div class="card">
            <div class="card-title">평가 메트릭 (metrics.json)</div>
            <table class="kv">
              <tr><td class="k">total</td><td class="v">{_fmt(metrics.get("total"))}</td></tr>
              <tr><td class="k">command_accuracy</td><td class="v">{_fmt(metrics.get("command_accuracy"))}</td></tr>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-title">데이터 품질 게이트 (data_quality_report.json)</div>
          <table class="kv">
            <tr><td class="k">status</td><td class="v">{_fmt(quality.get("status"))}</td></tr>
            <tr><td class="k">fail_reasons</td><td class="v">{_fmt(quality.get("fail_reasons"))}</td></tr>
            <tr><td class="k">warnings</td><td class="v">{_fmt(quality.get("warnings"))}</td></tr>
            <tr><td class="k">summary.total_rows</td><td class="v">{_fmt((quality.get("summary") or {}).get("total_rows"))}</td></tr>
            <tr><td class="k">summary.labeled_rows</td><td class="v">{_fmt((quality.get("summary") or {}).get("labeled_rows"))}</td></tr>
            <tr><td class="k">required.missing_or_blank_counts</td><td class="v">{_fmt(quality_rows.get("missing_or_blank_counts"))}</td></tr>
            <tr><td class="k">duplicates.exact_duplicate_rows</td><td class="v">{_fmt(dups.get("exact_duplicate_rows"))}</td></tr>
            <tr><td class="k">conflicts.input_label_conflicts</td><td class="v">{_fmt(conf.get("input_label_conflicts"))}</td></tr>
            <tr><td class="k">distribution.command_top20</td><td class="v">{_fmt(quality_dist.get("command_distribution_top20"))}</td></tr>
          </table>
        </div>

        <div class="card">
          <div class="card-title">리포트/결과 파일</div>
          <table class="kv">
            {file_rows}
          </table>
        </div>
      </div>
      <script>
        const reviewStatus = document.getElementById('reviewStatus');
        const reviewResult = document.getElementById('reviewResult');
        const finalizeStatus = document.getElementById('finalizeStatus');
        const finalizeResult = document.getElementById('finalizeResult');

        async function saveReviewApproval(approved) {{
          const rid = (document.getElementById('finalizeRunId').value || '').trim();
          if (!rid) {{
            alert('run_id를 입력하세요');
            return;
          }}
          const reviewer = (document.getElementById('reviewer').value || '').trim();
          const note = (document.getElementById('reviewNote').value || '').trim();
          reviewStatus.textContent = approved ? 'saving approval...' : 'revoking approval...';
          reviewResult.textContent = '';
          try {{
            const res = await fetch('/runs/review/approve', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{
                run_id: rid,
                reviewer: reviewer || null,
                note: note || null,
                approved: !!approved,
              }}),
            }});
            const text = await res.text();
            try {{
              reviewResult.textContent = JSON.stringify(JSON.parse(text), null, 2);
            }} catch (_e) {{
              reviewResult.textContent = text;
            }}
            reviewStatus.textContent = `done (HTTP ${{res.status}})`;
            setTimeout(() => location.reload(), 700);
          }} catch (e) {{
            reviewStatus.textContent = 'error';
            reviewResult.textContent = String(e);
          }}
        }}

        async function runFinalize(dryRun) {{
          const rid = (document.getElementById('finalizeRunId').value || '').trim();
          if (!rid) {{
            alert('run_id를 입력하세요');
            return;
          }}
          const minAccRaw = (document.getElementById('minCommandAccuracy').value || '').trim();
          const minCovRaw = (document.getElementById('minReasonCoverage').value || '1.0').trim();
          const maxMalRaw = (document.getElementById('maxMalformedRatio').value || '0.0').trim();
          const minCov = Number(minCovRaw);
          const maxMal = Number(maxMalRaw);
          if (!Number.isFinite(minCov) || !Number.isFinite(maxMal)) {{
            alert('min_reason_coverage / max_command_malformed_ratio는 숫자여야 합니다.');
            return;
          }}
          const payload = {{
            run_id: rid,
            dry_run: !!dryRun,
            min_reason_coverage: minCov,
            max_command_malformed_ratio: maxMal,
            require_review_approval: !!document.getElementById('requireReviewApproval').checked,
          }};
          if (minAccRaw !== '') {{
            const minAcc = Number(minAccRaw);
            if (!Number.isFinite(minAcc)) {{
              alert('min_command_accuracy는 숫자여야 합니다.');
              return;
            }}
            payload.min_command_accuracy = minAcc;
          }}

          finalizeStatus.textContent = dryRun ? 'validating...' : 'finalizing...';
          finalizeResult.textContent = '';
          try {{
            const res = await fetch('/runs/finalize', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify(payload),
            }});
            const text = await res.text();
            let json = null;
            try {{
              json = JSON.parse(text);
              finalizeResult.textContent = JSON.stringify(json, null, 2);
            }} catch (_e) {{
              finalizeResult.textContent = text;
            }}
            finalizeStatus.textContent = `done (HTTP ${{res.status}})`;
            if (json && json.ok && !dryRun) {{
              setTimeout(() => location.reload(), 700);
            }}
          }} catch (e) {{
            finalizeStatus.textContent = 'error';
            finalizeResult.textContent = String(e);
          }}
        }}

        document.getElementById('approveReview').addEventListener('click', () => saveReviewApproval(true));
        document.getElementById('revokeReview').addEventListener('click', () => saveReviewApproval(false));
        document.getElementById('dryFinalize').addEventListener('click', () => runFinalize(true));
        document.getElementById('applyFinalize').addEventListener('click', () => runFinalize(false));
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)
