# pyright: strict

from __future__ import annotations

from typing import Any, cast

JsonDict = dict[str, Any]


def json_dict(value: object) -> JsonDict:
    return cast(JsonDict, value) if isinstance(value, dict) else {}


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in cast(list[object], value) if isinstance(item, str)]


def string_value(value: object, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    return str(value)


def history_section(message_history: JsonDict, key: str) -> JsonDict:
    raw_section = message_history.get(key)
    if isinstance(raw_section, dict):
        return cast(JsonDict, raw_section)
    section: JsonDict = {}
    message_history[key] = section
    return section


def history_message_text(message_history: JsonDict, key: str) -> str:
    value = history_section(message_history, key).get("message", "")
    return value if isinstance(value, str) else str(value)
