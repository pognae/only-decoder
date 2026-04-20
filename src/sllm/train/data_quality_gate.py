import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from sllm.common.io import iter_jsonl_source, write_json

INPUT_EXCLUDE = {"record_id", "command", "reason", "accuracy"}


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _norm(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _norm(v[k]) for k in sorted(v.keys())}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, str):
        return " ".join(v.strip().split())
    return v


def _input_signature(row: Dict[str, Any]) -> str:
    payload = {k: row.get(k) for k in sorted(row.keys()) if k not in INPUT_EXCLUDE}
    return json.dumps(_norm(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _exact_signature(row: Dict[str, Any]) -> str:
    return json.dumps(_norm(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _label_signature(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("command", "")).strip(), str(row.get("reason", "")).strip())


def _top(counter: Counter, n: int = 20) -> Dict[str, int]:
    return dict(counter.most_common(n))


def run_gate(
    source: str,
    *,
    required_fields: List[str],
    require_labels: bool,
    conflict_sample_limit: int,
) -> Dict[str, Any]:
    total_rows = 0
    non_dict_rows = 0
    labeled_rows = 0
    record_id_blank = 0

    missing_or_blank = Counter()
    command_dist = Counter()
    system_dist = Counter()
    domain_dist = Counter()

    exact_seen = set()
    exact_duplicate_rows = 0

    input_first_label: Dict[str, Tuple[str, str]] = {}
    duplicate_input_same_label = 0
    input_label_conflicts = 0
    conflict_samples = []

    for row in iter_jsonl_source(source):
        total_rows += 1
        if not isinstance(row, dict):
            non_dict_rows += 1
            continue

        for field in required_fields:
            if _is_blank(row.get(field)):
                missing_or_blank[field] += 1

        if _is_blank(row.get("record_id")):
            record_id_blank += 1

        has_label = not _is_blank(row.get("command")) and not _is_blank(row.get("reason"))
        if has_label:
            labeled_rows += 1
            command_dist[str(row.get("command"))] += 1

        if not _is_blank(row.get("system")):
            system_dist[str(row.get("system"))] += 1
        if not _is_blank(row.get("domain")):
            domain_dist[str(row.get("domain"))] += 1

        exact_sig = _exact_signature(row)
        if exact_sig in exact_seen:
            exact_duplicate_rows += 1
        else:
            exact_seen.add(exact_sig)

        inp_sig = _input_signature(row)
        label_sig = _label_signature(row)
        first = input_first_label.get(inp_sig)
        if first is None:
            input_first_label[inp_sig] = label_sig
        else:
            if first == label_sig:
                duplicate_input_same_label += 1
            else:
                input_label_conflicts += 1
                if len(conflict_samples) < conflict_sample_limit:
                    conflict_samples.append(
                        {
                            "input_signature": inp_sig[:240],
                            "first_label": {"command": first[0], "reason": first[1]},
                            "current_label": {"command": label_sig[0], "reason": label_sig[1]},
                            "record_id": row.get("record_id"),
                            "message": row.get("message"),
                        }
                    )

    fail_reasons = []
    warnings = []

    if total_rows == 0:
        fail_reasons.append("dataset is empty")
    if non_dict_rows > 0:
        fail_reasons.append(f"non-dict rows detected: {non_dict_rows}")

    if require_labels and labeled_rows < total_rows:
        fail_reasons.append(f"rows missing command/reason labels: {total_rows - labeled_rows}")

    required_failed = {k: v for k, v in missing_or_blank.items() if v > 0}
    if required_failed:
        fail_reasons.append(f"required fields missing/blank: {required_failed}")

    if input_label_conflicts > 0:
        fail_reasons.append(f"input-label conflicts detected: {input_label_conflicts}")

    if labeled_rows > 0:
        top_cmd, top_n = command_dist.most_common(1)[0]
        top_ratio = top_n / labeled_rows
        if top_ratio >= 0.90:
            warnings.append(
                f"command distribution is highly skewed: {top_cmd}={top_ratio:.3f} ({top_n}/{labeled_rows})"
            )
        if len(command_dist) < 2:
            warnings.append("only one command class found in labels")
        else:
            min_n = min(command_dist.values())
            if min_n > 0 and (top_n / min_n) >= 20:
                warnings.append(f"command imbalance ratio is high: max/min={top_n/min_n:.2f}")

    for name, dist in (("system", system_dist), ("domain", domain_dist)):
        if dist and total_rows:
            v, n = dist.most_common(1)[0]
            ratio = n / total_rows
            if ratio >= 0.95:
                warnings.append(f"{name} is near single-source: {v}={ratio:.3f} ({n}/{total_rows})")

    if total_rows > 0 and exact_duplicate_rows / total_rows >= 0.30:
        warnings.append(f"exact duplicate row ratio is high: {exact_duplicate_rows/total_rows:.3f}")

    if total_rows > 0 and record_id_blank / total_rows >= 0.30:
        warnings.append(f"record_id blank ratio is high: {record_id_blank/total_rows:.3f}")

    status = "PASS" if not fail_reasons else "FAIL"

    return {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_source": os.path.abspath(source),
        "summary": {
            "total_rows": total_rows,
            "labeled_rows": labeled_rows,
            "non_dict_rows": non_dict_rows,
            "unique_exact_rows": len(exact_seen),
            "unique_input_signatures": len(input_first_label),
        },
        "checks": {
            "required_columns": {
                "required_fields": required_fields,
                "missing_or_blank_counts": dict(missing_or_blank),
                "record_id_blank_count": record_id_blank,
            },
            "duplicates": {
                "exact_duplicate_rows": exact_duplicate_rows,
                "duplicate_input_same_label": duplicate_input_same_label,
            },
            "conflicts": {
                "input_label_conflicts": input_label_conflicts,
                "samples": conflict_samples,
            },
            "distribution": {
                "command_distribution_top20": _top(command_dist, 20),
                "system_distribution_top20": _top(system_dist, 20),
                "domain_distribution_top20": _top(domain_dist, 20),
            },
        },
        "warnings": warnings,
        "fail_reasons": fail_reasons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_source",
        required=True,
        help="JSONL file or directory (recursive *.jsonl)",
    )
    ap.add_argument("--output_file", required=True)
    ap.add_argument(
        "--required_fields",
        default="message,system",
        help="comma-separated required columns for training input",
    )
    ap.add_argument(
        "--allow_unlabeled",
        action="store_true",
        help="allow rows missing command/reason labels (default: false)",
    )
    ap.add_argument("--conflict_sample_limit", type=int, default=30)
    ap.add_argument("--fail_on_fail", action="store_true")
    args = ap.parse_args()

    required_fields = [x.strip() for x in args.required_fields.split(",") if x.strip()]
    report = run_gate(
        args.input_source,
        required_fields=required_fields,
        require_labels=not args.allow_unlabeled,
        conflict_sample_limit=args.conflict_sample_limit,
    )
    out_path = os.path.abspath(args.output_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_json(out_path, report)

    print("status:", report["status"])
    print("total_rows:", report["summary"]["total_rows"])
    print("labeled_rows:", report["summary"]["labeled_rows"])
    print("conflicts:", report["checks"]["conflicts"]["input_label_conflicts"])
    print("saved:", out_path)

    if args.fail_on_fail and report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
