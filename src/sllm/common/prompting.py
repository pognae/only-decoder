import json
import re
from sllm.common.io import extract_input_fields, stable_json_dumps_message_first

SYSTEM_PROMPT = (
    "You are a legacy action sllm. "
    "Read the full input record and output only one valid JSON object. "
    "The JSON must contain command, reason, and accuracy."
)
OUTPUT_MARKER = "### OUTPUT_JSON"
FAIL_SAFE_REASON = "model output could not be parsed safely"


def _fail_safe_output() -> dict:
    return {
        "command": "NO_ACTION",
        "reason": FAIL_SAFE_REASON,
        "accuracy": {"command_accuracy": None},
        "_parse_failed": True,
    }

def build_prompt(record: dict) -> str:
    input_obj = extract_input_fields(record)
    input_text = stable_json_dumps_message_first(input_obj)
    return (
        f"{SYSTEM_PROMPT}\n"
        "Constraints:\n"
        "- Output MUST be a single JSON object only.\n"
        "- 'reason' should be a meaningful, input-grounded explanation (target ~100+ chars when possible).\n"
        "### INPUT_RECORD_JSON\n"
        f"{input_text}\n"
        "### OUTPUT_JSON\n"
    )

def build_target(record: dict) -> str:
    return json.dumps(
        {
            "command": record["command"],
            "reason": record["reason"],
            "accuracy": {"command_accuracy": None},
        },
        ensure_ascii=False,
    )

def build_inference_prompt(input_record) -> str:
    input_obj = extract_input_fields(input_record)
    input_text = stable_json_dumps_message_first(input_obj)
    inference_instruction = (
        "Return exactly one JSON object with keys command, reason, accuracy. "
        "Do not output prose or repeated keys. "
        "Use this shape: "
        '{"command":"<COMMAND>","reason":"<SHORT_REASON>","accuracy":{"command_accuracy":null}}'
    )
    return (
        f"{SYSTEM_PROMPT}\n"
        f"{inference_instruction}\n"
        "Prefer a reason that references concrete fields/entities from INPUT_RECORD_JSON; "
        "when context allows, aim for ~100+ characters.\n"
        f"### INPUT_RECORD_JSON\n{input_text}\n"
        "### OUTPUT_JSON\n"
    )


def _extract_command_accuracy(text: str):
    m = re.search(r'"command_accuracy"\s*:\s*(null|-?\d+(?:\.\d+)?)', text, re.I)
    if not m:
        return None
    raw = m.group(1).strip().lower()
    if raw == "null":
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _repair_json_like_output(text: str) -> dict:
    command = None
    reason = None

    m_cmd = re.search(r'"command"\s*:\s*"([^"]+)"', text, re.I)
    if m_cmd:
        command = (m_cmd.group(1) or "").strip()

    m_reason = re.search(r'"reason"\s*:\s*"([^"]+)"', text, re.I)
    if m_reason:
        reason = (m_reason.group(1) or "").strip()

    quoted = [m.group(1).strip() for m in re.finditer(r'"([^"]+)"', text)]
    if not command:
        for q in quoted:
            if q and q.lower() not in {"command", "reason", "accuracy", "command_accuracy"}:
                command = q
                break
    if not reason:
        skip = {"command", "reason", "accuracy", "command_accuracy", command or ""}
        for q in quoted:
            if not q or q in skip:
                continue
            if len(q) >= 8:
                reason = q
                break

    if not command and not reason:
        return _fail_safe_output()

    return {
        "command": command or "NO_ACTION",
        "reason": reason or FAIL_SAFE_REASON,
        "accuracy": {"command_accuracy": _extract_command_accuracy(text)},
        "_parse_failed": False,
        "_parse_recovered": True,
    }

def parse_json_fragment(text: str) -> dict:
    # Prefer extracting JSON after OUTPUT marker to avoid accidentally parsing
    # braces that exist in the input record itself.
    search_text = text
    marker_index = text.rfind(OUTPUT_MARKER)
    if marker_index != -1:
        search_text = text[marker_index + len(OUTPUT_MARKER) :]

    start = search_text.find("{")
    if start == -1:
        return _repair_json_like_output(search_text)

    # Scan forward to find the first balanced JSON object.
    depth = 0
    in_string = False
    escape = False
    end = None
    for i in range(start, len(search_text)):
        ch = search_text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        return _repair_json_like_output(search_text)

    chunk = search_text[start : end + 1]
    try:
        parsed = json.loads(chunk)
        if not isinstance(parsed, dict):
            return _repair_json_like_output(search_text)
        parsed["_parse_failed"] = False
        return parsed
    except Exception:
        return _repair_json_like_output(search_text)
