import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from sllm.common.io import read_jsonl


def _normalize_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _write_jsonl(path: str, rows: Iterable[Dict], append: bool = False):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _top_ip_by_page_rule(row: Dict) -> Optional[Tuple[str, str, float, str]]:
    message = _normalize_text(row.get("message"))
    if not message:
        return None

    has_ip = "ip" in message
    has_question_intent = any(
        token in message
        for token in (
            "which ip",
            "top ip",
            "most",
            "highest",
            "frequent",
            "seen",
            "viewed",
        )
    )
    has_page_context = any(token in message for token in ("mainpagesd", "page", "/dw/main/"))
    state = _normalize_text(row.get("state"))

    score = 0.0
    if has_ip:
        score += 0.35
    if has_question_intent:
        score += 0.35
    if has_page_context:
        score += 0.2
    if state == "query":
        score += 0.1
    if "?" in str(row.get("message", "")):
        score += 0.05

    if score < 0.8:
        return None
    confidence = min(0.99, score)
    return (
        "TOP_IP_BY_PAGE",
        "identify the most frequent IP for the requested page",
        round(confidence, 4),
        "rule_top_ip_by_page_v1",
    )


def _user_by_login_rule(row: Dict) -> Optional[Tuple[str, str, float, str]]:
    message = _normalize_text(row.get("message"))
    if not message:
        return None

    pattern = re.compile(r"user\s+with\s+id.*access\s+to\s+ip.*checked\s+the")
    matched = bool(pattern.search(message))

    has_login_tokens = all(token in message for token in ("user with id", "access to ip", "checked the"))
    state = _normalize_text(row.get("state"))
    system = _normalize_text(row.get("system"))

    if not matched and not has_login_tokens:
        return None

    score = 0.85
    if matched:
        score += 0.08
    if state in {"page", "login"}:
        score += 0.03
    if system in {"oms", "erp", "mes", "tms"}:
        score += 0.02

    confidence = min(0.99, score)
    return (
        "USER_BY_LOGIN",
        "access status by user id",
        round(confidence, 4),
        "rule_user_access_log_v1",
    )


def _delivery_vehicle_rule(row: Dict) -> Optional[Tuple[str, str, float, str]]:
    message = _normalize_text(row.get("message"))
    if not message:
        return None

    system = _normalize_text(row.get("system"))
    state = _normalize_text(row.get("state"))
    has_vehicle_field = bool(_normalize_text(row.get("vehicle_name")))

    has_vehicle_token = any(token in message for token in ("vehicle", "car", "truck", "차량", "차"))
    has_delivery_token = any(
        token in message
        for token in ("delivery", "operation", "status", "progress", "state", "운행", "배송", "상태")
    )
    has_query_intent = ("?" in str(row.get("message", ""))) or any(
        token in message for token in ("check", "show", "verify", "what is", "조회", "확인", "알려", "어떻게")
    )

    if not ((has_vehicle_token or has_vehicle_field) and has_delivery_token):
        return None

    score = 0.72
    if has_vehicle_token:
        score += 0.08
    if has_vehicle_field:
        score += 0.05
    if has_query_intent:
        score += 0.08
    if system == "tms":
        score += 0.04
    if state == "query":
        score += 0.03

    confidence = min(0.99, score)
    return (
        "DELIVERY_VEHICLE",
        "vehicle operation detection",
        round(confidence, 4),
        "rule_delivery_vehicle_query_v1",
    )


RULES = (
    _delivery_vehicle_rule,
    _top_ip_by_page_rule,
    _user_by_login_rule,
)


def _infer_label(row: Dict) -> Optional[Tuple[str, str, float, str]]:
    for rule in RULES:
        result = rule(row)
        if result is not None:
            return result
    return None


def _load_existing_record_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    ids = set()
    for row in read_jsonl(path):
        rid = row.get("record_id")
        if rid is not None:
            ids.add(str(rid))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bronze_file", required=True)
    ap.add_argument("--silver_file", required=True)
    ap.add_argument("--reject_file", default=None)
    ap.add_argument("--min_confidence", type=float, default=0.8)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_rows", type=int, default=0, help="0 means all rows")
    args = ap.parse_args()

    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min_confidence must be between 0.0 and 1.0")

    existing_ids = _load_existing_record_ids(args.silver_file) if args.append else set()

    generated: List[Dict] = []
    rejected: List[Dict] = []
    skipped_existing = 0
    skipped_labeled = 0
    total = 0

    for row in read_jsonl(args.bronze_file):
        total += 1
        if args.max_rows and total > args.max_rows:
            break

        rid = row.get("record_id")
        if rid is not None and str(rid) in existing_ids:
            skipped_existing += 1
            continue

        if "command" in row or "reason" in row:
            skipped_labeled += 1
            continue

        inferred = _infer_label(row)
        if inferred is None:
            rejected.append(row)
            continue

        command, reason, confidence, source = inferred
        if confidence < args.min_confidence:
            rejected.append(row)
            continue

        out = dict(row)
        out["command"] = command
        out["reason"] = reason
        out["auto_label_source"] = source
        out["auto_label_confidence"] = confidence
        generated.append(out)

    if not args.dry_run:
        if generated:
            _write_jsonl(args.silver_file, generated, append=args.append)
        if args.reject_file and rejected:
            _write_jsonl(args.reject_file, rejected, append=False)

    print("bronze_to_silver summary")
    print("total_rows:", total if not args.max_rows else min(total, args.max_rows))
    print("generated_silver_rows:", len(generated))
    print("rejected_rows:", len(rejected))
    print("skipped_existing_rows:", skipped_existing)
    print("skipped_already_labeled_rows:", skipped_labeled)
    print("dry_run:", args.dry_run)
    if not args.dry_run and generated:
        print("saved:", args.silver_file)
    if not args.dry_run and args.reject_file and rejected:
        print("saved:", args.reject_file)


if __name__ == "__main__":
    main()
