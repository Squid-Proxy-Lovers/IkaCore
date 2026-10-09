"""Explicit character limits for tool output; existing callers retain full text."""

# pyright: strict

from __future__ import annotations

import os
from typing import Optional

from .request_control import codex_tool_output_limit, tool_result_limit


def _environment_limit(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        result = int(value)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _truncate(text: str, limit: Optional[int], label: str) -> str:
    if limit is None or len(text) <= limit:
        return text
    marker = f"\n[IkaCore truncated {label}: original length {len(text)} characters]\n"
    if len(marker) >= limit:
        return text[:limit]
    budget = limit - len(marker)
    head = (budget + 1) // 2
    tail = budget - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def truncate_tool_result(text: str, tool_name: str = "tool") -> str:
    limit = tool_result_limit()
    if limit is None:
        limit = _environment_limit("IKA_TOOL_RESULT_MAX_CHARS")
    return _truncate(text, limit, tool_name)


def truncate_codex_tool_output(text: str) -> str:
    limit = codex_tool_output_limit()
    if limit is None:
        limit = _environment_limit("IKA_CODEX_TOOL_OUTPUT_MAX_CHARS")
    return _truncate(text, limit, "Codex tool output")


def codex_output_limit_enabled() -> bool:
    return codex_tool_output_limit() is not None or "IKA_CODEX_TOOL_OUTPUT_MAX_CHARS" in os.environ
