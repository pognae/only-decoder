import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_COMMANDS: List[Dict[str, Any]] = [
    {
        "command": "USER_BY_LOGIN",
        "reason_template": "access status by user id",
        "aliases": ["access status by user id", "user login history", "login access check"],
    },
    {
        "command": "TOP_IP_BY_PAGE",
        "reason_template": "identify the most frequent IP for the requested page",
        "aliases": ["top ip by page", "most viewed ip", "highest-frequency ip"],
    },
    {
        "command": "DELIVERY_VEHICLE",
        "reason_template": "vehicle operation detection",
        "aliases": ["vehicle operation detection", "delivery vehicle status", "vehicle dispatch status"],
    },
    {
        "command": "RESYNC_ORDER",
        "reason_template": "legacy OMS sync exception",
        "aliases": ["order resync", "resync order", "sync failure recovery"],
    },
    {
        "command": "UPDATE_LOCATION",
        "reason_template": "location mapping invalid",
        "aliases": ["update location", "fix location mapping", "invalid picking location"],
    },
    {
        "command": "SYNC_INVENTORY",
        "reason_template": "stock quantity inconsistency detected",
        "aliases": ["sync inventory", "inventory mismatch", "stock inconsistency"],
    },
    {
        "command": "RESCHEDULE_SHIPMENT",
        "reason_template": "carrier schedule exception detected",
        "aliases": ["reschedule shipment", "shipment delayed", "carrier schedule issue"],
    },
    {
        "command": "NO_ACTION",
        "reason_template": "no action required based on current input",
        "aliases": ["no action", "unknown", "none"],
    },
]


def get_commands_path(model_dir: str) -> str:
    return os.path.join(model_dir, "commands.json")


def _sanitize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    command = str(entry.get("command", "")).strip().upper()
    command = re.sub(r"[^A-Z0-9_]", "_", command)
    reason_template = str(entry.get("reason_template", "")).strip()
    aliases_raw = entry.get("aliases", [])
    aliases: List[str] = []
    if isinstance(aliases_raw, list):
        aliases = [str(x).strip() for x in aliases_raw if str(x).strip()]
    elif isinstance(aliases_raw, str):
        aliases = [x.strip() for x in aliases_raw.split(",") if x.strip()]
    return {
        "command": command,
        "reason_template": reason_template,
        "aliases": aliases,
    }


def _dedupe_commands(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        command = row.get("command", "")
        if not command or command in seen:
            continue
        seen.add(command)
        out.append(row)
    return out


def ensure_commands_file(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_COMMANDS, f, ensure_ascii=False, indent=2)


def load_commands(path: str) -> List[Dict[str, Any]]:
    ensure_commands_file(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = DEFAULT_COMMANDS
    if not isinstance(raw, list):
        raw = DEFAULT_COMMANDS
    sanitized = [_sanitize_entry(x) for x in raw if isinstance(x, dict)]
    deduped = _dedupe_commands(sanitized)
    if not deduped:
        deduped = [_sanitize_entry(x) for x in DEFAULT_COMMANDS]
    return deduped


def save_commands(path: str, commands: List[Dict[str, Any]]) -> None:
    rows = _dedupe_commands([_sanitize_entry(x) for x in commands if isinstance(x, dict)])
    if not rows:
        rows = [_sanitize_entry(x) for x in DEFAULT_COMMANDS]
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _normalize_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.upper().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^A-Z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).lower().split())


def _candidate_texts(row: Dict[str, Any]) -> List[str]:
    values = [row.get("command"), row.get("reason_template")]
    values.extend(row.get("aliases", []) or [])
    return [x for x in (_text(v) for v in values) if x]


def resolve_command(
    predicted_command: Any,
    predicted_reason: Any,
    commands: List[Dict[str, Any]],
) -> Tuple[str, str]:
    if not commands:
        return "NO_ACTION", "no_catalog"

    by_norm = {_normalize_token(row["command"]): row["command"] for row in commands}
    by_phrase: Dict[str, str] = {}
    for row in commands:
        command = row["command"]
        for phrase in _candidate_texts(row):
            by_phrase[phrase] = command

    # 1) strict normalized command match
    pc_norm = _normalize_token(predicted_command)
    if pc_norm and pc_norm in by_norm:
        return by_norm[pc_norm], "direct_command"

    # 2) exact phrase match against reason/aliases
    pc_text = _text(predicted_command)
    pr_text = _text(predicted_reason)
    for text in (pc_text, pr_text):
        if text and text in by_phrase:
            return by_phrase[text], "direct_phrase"

    # 3) fuzzy match on combined text
    combined = " ".join(x for x in [pc_text, pr_text] if x).strip()
    if combined:
        best_cmd = None
        best_score = -1.0
        for row in commands:
            cmd = row["command"]
            phrases = _candidate_texts(row)
            if not phrases:
                continue
            score = max(SequenceMatcher(None, combined, p).ratio() for p in phrases)
            if score > best_score:
                best_score = score
                best_cmd = cmd
        if best_cmd and best_score >= 0.35:
            return best_cmd, "fuzzy"

    # 4) fallback
    if "NO_ACTION" in by_norm.values():
        return "NO_ACTION", "fallback_no_action"
    return commands[0]["command"], "fallback_first"


def _message_from_input_record(input_record: Any) -> str:
    if isinstance(input_record, dict):
        return str(input_record.get("message", "")).strip()
    if isinstance(input_record, str):
        return input_record.strip()
    return ""


def _find_first(pattern: str, text: str, flags: int = 0) -> str:
    m = re.search(pattern, text, flags)
    if not m:
        return ""
    if m.lastindex:
        return (m.group(1) or "").strip()
    return (m.group(0) or "").strip()


def _extract_intent_slots(message: str) -> Dict[str, str]:
    slots: Dict[str, str] = {}
    msg = message.strip()

    page = _find_first(r"(/[A-Za-z0-9_./-]+(?:\?[A-Za-z0-9_=&.-]+)?)", msg)
    ip = _find_first(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", msg)
    user = _find_first(r"user\s+with\s+id\s*\(?([A-Za-z0-9_.@-]+)\)?", msg, re.I)
    if not user:
        user = _find_first(r"\bid\s*\(?([A-Za-z0-9_.@-]{3,})\)?", msg, re.I)
    vehicle = _find_first(r"vehicle\s+([A-Za-z0-9_.\-가-힣 ]+?)(?:\s+on|\s+for|\s+date|\?|$)", msg, re.I)
    order_id = _find_first(r"\b(?:ORD|ORDER)[-_]?[A-Za-z0-9-]+\b", msg, re.I)
    wh = _find_first(r"\bWH[-_]?[A-Za-z0-9]+\b", msg, re.I)

    if page:
        slots["page"] = page
    if ip:
        slots["ip"] = ip
    if user:
        slots["user"] = user
    if vehicle:
        slots["vehicle"] = vehicle
    if order_id:
        slots["order_id"] = order_id
    if wh:
        slots["warehouse"] = wh
    return slots


def _is_template_like_reason(reason: Any, command: str, commands: List[Dict[str, Any]]) -> bool:
    text = _text(reason)
    if not text:
        return True
    if len(text) < 10:
        return True

    row = next((x for x in commands if x.get("command") == command), None)
    if not row:
        return False

    template = _text(row.get("reason_template"))
    if template and text == template:
        return True

    aliases = [_text(x) for x in (row.get("aliases") or [])]
    if text in aliases:
        return True
    return False


def _build_reason_from_intent(command: str, input_record: Any) -> str:
    message = _message_from_input_record(input_record)
    slots = _extract_intent_slots(message)
    lower = message.lower()
    query_like = "?" in message or any(
        tok in lower for tok in ["which", "what", "show", "identify", "top", "most", "find", "can you"]
    )

    if command == "TOP_IP_BY_PAGE":
        page = slots.get("page")
        if page and query_like:
            return f"query requests the highest-frequency client IP for page {page}"
        if page:
            return f"page access trend for {page} requires top IP aggregation"
        return "input asks for highest-frequency client IP analysis on a target page"

    if command == "USER_BY_LOGIN":
        user = slots.get("user")
        ip = slots.get("ip")
        page = slots.get("page")
        if user and ip and page:
            return f"user login trace is required for id {user}, ip {ip}, and page {page}"
        if user and ip:
            return f"user login/access history is required for id {user} from ip {ip}"
        if page:
            return f"login/access audit is required for user activity on page {page}"
        return "input requires user login/access trace verification"

    if command == "DELIVERY_VEHICLE":
        vehicle = slots.get("vehicle")
        if vehicle:
            return f"input asks for delivery operation status for vehicle {vehicle}"
        return "input asks for vehicle delivery operation status verification"

    if command == "RESYNC_ORDER":
        order_id = slots.get("order_id")
        if order_id:
            return f"order sync recovery is required due to sync exception on {order_id}"
        return "order synchronization failed and requires resync recovery"

    if command == "UPDATE_LOCATION":
        warehouse = slots.get("warehouse")
        if warehouse:
            return f"location mapping inconsistency detected for warehouse {warehouse}"
        return "location mapping is invalid and requires location master correction"

    if command == "SYNC_INVENTORY":
        return "input indicates inventory mismatch and requires stock synchronization"

    if command == "RESCHEDULE_SHIPMENT":
        return "shipment schedule exception requires carrier rescheduling"

    if command == "NO_ACTION":
        return "no deterministic action was derived from current input"

    return f"input intent is aligned with command {command}"


def normalize_prediction_with_catalog(
    parsed: Dict[str, Any],
    commands: List[Dict[str, Any]],
    input_record: Any = None,
) -> Dict[str, Any]:
    predicted_command = parsed.get("command")
    predicted_reason = parsed.get("reason")
    selected_command, strategy = resolve_command(predicted_command, predicted_reason, commands)

    reason = predicted_reason
    if _is_template_like_reason(reason, selected_command, commands):
        reason = _build_reason_from_intent(selected_command, input_record)
    elif isinstance(reason, str):
        reason = reason.strip()
    else:
        reason = str(reason)

    out = dict(parsed)
    out["command"] = selected_command
    out["reason"] = reason
    out["_command_resolution"] = strategy
    return out
