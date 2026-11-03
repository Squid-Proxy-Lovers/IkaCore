"""Opt-in recovery for truncated control-tool JSON objects."""

import json
import re
from typing import Any, Optional

_PARTIAL_UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{0,3}$")


def _close_truncated_json_object(raw: str) -> Optional[dict[str, Any]]:
    """Recover a JSON object whose text was cut off, e.g. by the model's output-token limit.

    Closes an unterminated string and any open brackets; returns None when the result still
    does not parse (e.g. the cut fell between a key and its value).
    """
    text = raw.strip()
    if not text.startswith("{"):
        return None
    closers: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            closers.append("}" if ch == "{" else "]")
        elif ch in "}]" and closers:
            closers.pop()
    if in_string:
        text = _PARTIAL_UNICODE_ESCAPE.sub("", text[:-1] if escaped else text)
    suffix = ('"' if in_string else "") + "".join(reversed(closers))
    try:
        parsed = json.loads(text + suffix)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None

