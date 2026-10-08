"""Retry policy shared by every provider request path.

Which failures are worth retrying, and how long to wait before the next
attempt. Mirrors the official OpenAI/Anthropic SDKs: retry request timeouts,
lock conflicts, rate limits and server errors (408, 409, 429, 5xx including
Anthropic's 529), fail fast on everything else, prefer the provider's own
retry hint, and otherwise back off exponentially with jitter so parallel
agents that hit the same limit do not retry in lockstep.
"""

# pyright: strict

from __future__ import annotations

import email.utils
import json
import random
import re
import time
from collections.abc import Mapping
from datetime import timezone
from typing import Any, Optional, cast

INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0
MAX_HINT_S = 300.0

_RETRYABLE_STATUS = frozenset({408, 409, 429})
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
_DURATION_SCALE = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def is_retryable_status(status: int) -> bool:
    return status in _RETRYABLE_STATUS or status >= 500


def parse_retry_after(header_value: Optional[str], default: float) -> float:
    """Parse ``Retry-After`` as seconds (int or float) or an HTTP-date (RFC 7231 §7.1.3).

    Falls back to ``default`` when absent or unparseable. Naive datetimes are
    treated as UTC so an unknown source zone cannot produce a wildly wrong wait.
    """
    if not header_value:
        return float(default)
    try:
        return max(0.0, float(header_value))
    except (TypeError, ValueError):
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(header_value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, parsed.timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        pass
    return float(default)


def _parse_duration(value: str) -> Optional[float]:
    """OpenAI-style reset durations: ``"1s"``, ``"250ms"``, ``"6m0s"``, ``"1h2m3.5s"``."""
    text = value.strip()
    parts = _DURATION_PART.findall(text)
    if not parts or "".join(num + unit for num, unit in parts) != text:
        return None
    return sum(float(num) * _DURATION_SCALE[unit] for num, unit in parts)


def _body_retry_delay(body: Any) -> Optional[float]:
    """Google APIs put the hint in the error body: ``details[].retryDelay = "25s"``."""
    error = cast(dict[str, Any], body).get("error") if isinstance(body, dict) else None
    details = cast(dict[str, Any], error).get("details") if isinstance(error, dict) else None
    if not isinstance(details, list):
        return None
    for detail in cast(list[Any], details):
        if isinstance(detail, dict) and str(cast(dict[str, Any], detail).get("@type", "")).endswith("RetryInfo"):
            delay = cast(dict[str, Any], detail).get("retryDelay")
            if isinstance(delay, str):
                return _parse_duration(delay)
    return None


def retry_hint_seconds(headers: Mapping[str, str], body: Any = None) -> Optional[float]:
    """The provider's requested wait, from headers or error body; None when it gave none."""
    lowered = {k.lower(): v for k, v in headers.items()}
    if "retry-after-ms" in lowered:
        try:
            return max(0.0, float(lowered["retry-after-ms"]) / 1000.0)
        except ValueError:
            pass
    if "retry-after" in lowered:
        parsed = parse_retry_after(lowered["retry-after"], default=-1.0)
        if parsed >= 0:
            return parsed
    resets = [
        d for key in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")
        if key in lowered and (d := _parse_duration(lowered[key])) is not None
    ]
    if resets:
        return max(resets)
    return _body_retry_delay(body)


def backoff_seconds(attempt: int, initial: float = INITIAL_BACKOFF_S) -> float:
    """Exponential backoff (initial, 2x, 4x ... capped) scaled by 0.75-1.0 jitter."""
    base = min(float(initial) * (2 ** attempt), MAX_BACKOFF_S)
    return base * (1.0 - 0.25 * random.random())


def retry_delay(attempt: int, hint: Optional[float], initial: float = INITIAL_BACKOFF_S) -> float:
    if hint is not None:
        return min(hint, MAX_HINT_S)
    return backoff_seconds(attempt, initial)


def response_body(response: Any) -> Any:
    try:
        return response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


__all__ = [
    "INITIAL_BACKOFF_S",
    "MAX_BACKOFF_S",
    "MAX_HINT_S",
    "backoff_seconds",
    "is_retryable_status",
    "parse_retry_after",
    "response_body",
    "retry_delay",
    "retry_hint_seconds",
]
