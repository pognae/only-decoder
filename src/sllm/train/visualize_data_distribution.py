import argparse
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from sllm.common.data_split import iter_bucket_rows
from sllm.common.io import iter_jsonl_source, read_jsonl, write_json


def _msg_len_bucket(length: int) -> str:
    if length < 20:
        return "0-19"
    if length < 50:
        return "20-49"
    if length < 100:
        return "50-99"
    if length < 200:
        return "100-199"
    return "200+"


def _to_sorted_items(counter: Counter, top_n: int = 20) -> List[Tuple[str, int]]:
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_n]


def _collect_stats_from_rows(row_iter) -> Dict:
    command = Counter()
    system = Counter()
    state = Counter()
    domain = Counter()
    message_bucket = Counter()
    rows = 0
    labeled_rows = 0

    for row in row_iter:
        rows += 1
        if row.get("command") is not None:
            labeled_rows += 1
            command[str(row.get("command"))] += 1
        if row.get("system") is not None:
            system[str(row.get("system"))] += 1
        if row.get("state") is not None:
            state[str(row.get("state"))] += 1
        if row.get("domain") is not None:
            domain[str(row.get("domain"))] += 1
        msg = str(row.get("message", ""))
        message_bucket[_msg_len_bucket(len(msg))] += 1

    return {
        "rows": rows,
        "labeled_rows": labeled_rows,
        "command_distribution": dict(_to_sorted_items(command, top_n=50)),
        "system_distribution": dict(_to_sorted_items(system, top_n=20)),
        "state_distribution": dict(_to_sorted_items(state, top_n=20)),
        "domain_distribution": dict(_to_sorted_items(domain, top_n=20)),
        "message_length_bucket_distribution": dict(
            sorted(message_bucket.items(), key=lambda x: x[0])
        ),
    }


def _collect_stats(source: str) -> Dict:
    """One JSONL file or merged ``*.jsonl`` under a directory."""
    return _collect_stats_from_rows(iter_jsonl_source(source))


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_bar_rows(items: Iterable[Tuple[str, int]], max_value: int) -> str:
    if max_value <= 0:
        max_value = 1
    rows = []
    for label, value in items:
        width = int((value / max_value) * 100)
        rows.append(
            "<div class='row'>"
            f"<div class='label'>{_escape_html(label)}</div>"
            "<div class='bar-wrap'>"
            f"<div class='bar' style='width:{width}%;'></div>"
            "</div>"
            f"<div class='value'>{value}</div>"
            "</div>"
        )
    return "\n".join(rows)


def _render_section(title: str, dist: Dict[str, int]) -> str:
    items = list(dist.items())
    if not items:
        return f"<h3>{_escape_html(title)}</h3><p class='empty'>No data</p>"
    max_value = max(v for _, v in items)
    return (
        f"<h3>{_escape_html(title)}</h3>"
        "<div class='chart'>"
        f"{_render_bar_rows(items, max_value)}"
        "</div>"
    )


def _render_split_card(split_name: str, stats: Dict) -> str:
    return (
        "<section class='card'>"
        f"<h2>{_escape_html(split_name)}</h2>"
        f"<p class='meta'>rows={stats['rows']} | labeled_rows={stats['labeled_rows']}</p>"
        f"{_render_section('Command', stats['command_distribution'])}"
        f"{_render_section('System', stats['system_distribution'])}"
        f"{_render_section('State', stats['state_distribution'])}"
        f"{_render_section('Domain', stats['domain_distribution'])}"
        f"{_render_section('Message Length Bucket', stats['message_length_bucket_distribution'])}"
        "</section>"
    )


def _split_subtitle(report: Dict) -> str:
    if report.get("split_mode") == "auto":
        r = report.get("train_ratio", 0.8)
        return f" | auto_split train_ratio={r} (stable hash, ~{(1 - r) * 100:.0f}% valid)"
    if report.get("valid_file"):
        return " | valid_file=" + _escape_html(report["valid_file"])
    return ""


def _render_html(report: Dict) -> str:
    train_stats = report["train"]
    valid_stats = report.get("valid")
    valid_card = _render_split_card("VALID", valid_stats) if valid_stats else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Data Distribution Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 20px;
      color: #222;
      background: #f7fafc;
    }}
    h1 {{ margin: 0 0 6px 0; }}
    .sub {{ color: #4a5568; margin-bottom: 16px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 14px;
    }}
    .meta {{ color: #4a5568; margin: 0 0 12px 0; }}
    .chart {{ margin-bottom: 18px; }}
    .row {{
      display: grid;
      grid-template-columns: 180px 1fr 60px;
      align-items: center;
      gap: 8px;
      margin: 4px 0;
    }}
    .label {{
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 12px;
      color: #2d3748;
    }}
    .bar-wrap {{
      background: #edf2f7;
      border-radius: 4px;
      height: 12px;
      overflow: hidden;
    }}
    .bar {{
      background: linear-gradient(90deg, #2b6cb0, #2c5282);
      height: 12px;
    }}
    .value {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      color: #2d3748;
    }}
    .empty {{ color: #718096; }}
  </style>
</head>
<body>
  <h1>Data Distribution Report</h1>
  <div class="sub">train_file={_escape_html(report['train_file'])}{_split_subtitle(report)}</div>
  <div class="grid">
    {_render_split_card("TRAIN", train_stats)}
    {valid_card}
  </div>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_file",
        required=True,
        help="JSONL file or directory (recursive *.jsonl)",
    )
    ap.add_argument("--valid_file", default=None)
    ap.add_argument(
        "--auto_split",
        action="store_true",
        help="build TRAIN/VALID stats from train_file using same split as training (default train_ratio=0.8)",
    )
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.auto_split:
        report = {
            "train_file": os.path.abspath(args.train_file),
            "valid_file": None,
            "split_mode": "auto",
            "train_ratio": args.train_ratio,
            "train": _collect_stats_from_rows(
                iter_bucket_rows(args.train_file, args.train_ratio, want_train=True)
            ),
            "valid": _collect_stats_from_rows(
                iter_bucket_rows(args.train_file, args.train_ratio, want_train=False)
            ),
        }
    else:
        report = {
            "train_file": os.path.abspath(args.train_file),
            "valid_file": os.path.abspath(args.valid_file) if args.valid_file else None,
            "train": _collect_stats(args.train_file),
            "valid": _collect_stats(args.valid_file) if args.valid_file else None,
        }

    json_path = os.path.join(args.output_dir, "data_distribution_report.json")
    html_path = os.path.join(args.output_dir, "data_distribution_report.html")

    write_json(json_path, report)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_html(report))

    print("saved:", json_path)
    print("saved:", html_path)


if __name__ == "__main__":
    main()
