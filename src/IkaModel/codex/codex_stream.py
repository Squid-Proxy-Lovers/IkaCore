"""Codex Server-Sent Events parsing and terminal-state validation."""

# pyright: strict

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Optional, cast

import httpx

from IkaCore.agent_runtime_payloads import JsonDict, json_dict, string_value

LOG = logging.getLogger(__name__)

_RETRYABLE_STREAM_ERROR_IDENTIFIERS = frozenset({
    "engine_error",
    "internal_error",
    "model_overloaded",
    "overloaded_error",
    "rate_limit_error",
    "rate_limit_exceeded",
    "server_error",
    "server_is_overloaded",
})


class CodexRetryableStreamError(RuntimeError):
    """A transient stream failure that the request loop may retry."""


def _empty_json_dict() -> JsonDict:
    return {}


def _empty_json_list() -> list[JsonDict]:
    return []


def _empty_string_list() -> list[str]:
    return []


@dataclass
class _StreamState:
    envelope: JsonDict = field(default_factory=_empty_json_dict)
    collected_items: list[JsonDict] = field(
        default_factory=_empty_json_list
    )
    output_text_chunks: list[str] = field(
        default_factory=_empty_string_list
    )
    completed: bool = False


def _normalized_stream_error_identifier(value: object) -> str:
    return string_value(value).strip().casefold()


def classify_stream_error(err_obj: object) -> bool:
    """Return whether an error has an exact transient type or code."""
    if not isinstance(err_obj, dict):
        return False
    error_data = cast(JsonDict, err_obj)
    return any(
        _normalized_stream_error_identifier(error_data.get(field_name))
        in _RETRYABLE_STREAM_ERROR_IDENTIFIERS
        for field_name in ("type", "code")
    )


def iter_sse(
    response: httpx.Response,
) -> Iterator[tuple[Optional[str], object]]:
    """Yield an explicit event name and parsed JSON for each SSE data line."""
    current_event: Optional[str] = None
    for line in response.iter_lines():
        if not line:
            current_event = None
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield current_event, json.loads(data)
        except json.JSONDecodeError:
            LOG.debug("Skipping unparseable SSE data line: %r", data[:200])


def _stream_failure_object(obj_data: JsonDict) -> object:
    """Return the most specific error object from an SSE failure envelope."""
    top_level = obj_data.get("error")
    if top_level:
        return top_level
    nested = json_dict(obj_data.get("response")).get("error")
    if nested:
        return nested
    return obj_data


def _raise_stream_failure(event_type: str, obj_data: JsonDict) -> None:
    """Raise a typed stream failure without classifying message prose."""
    error = _stream_failure_object(obj_data)
    if classify_stream_error(error):
        raise CodexRetryableStreamError(
            f"codex stream transient failure ({event_type}): {error}"
        )
    raise RuntimeError(f"codex response failed ({event_type}): {error}")


def _event_type(event_name: Optional[str], obj_data: JsonDict) -> str:
    return (
        string_value(obj_data.get("type")).strip()
        or string_value(event_name).strip()
    )


def _incomplete_details(obj_data: JsonDict) -> object:
    return (
        json_dict(obj_data.get("response")).get("incomplete_details")
        or obj_data.get("incomplete_details")
        or _stream_failure_object(obj_data)
    )


def _completed_response(obj_data: JsonDict) -> JsonDict:
    response = json_dict(obj_data.get("response"))
    if not response:
        return {}
    status = string_value(response.get("status")).strip().casefold()
    if status in {"failed", "error", "cancelled"}:
        _raise_stream_failure("response.completed", obj_data)
    if status == "incomplete":
        details = response.get("incomplete_details") or response.get("error")
        raise RuntimeError(
            f"codex response completed with incomplete status: {details}"
        )
    if status and status != "completed":
        raise CodexRetryableStreamError(
            "codex response.completed carried non-terminal "
            f"status {status!r}"
        )
    return response


def _apply_stream_event(
    state: _StreamState,
    event_name: Optional[str],
    obj_data: JsonDict,
) -> None:
    event_type = _event_type(event_name, obj_data)
    if event_type in {"error", "response.error", "response.failed"}:
        _raise_stream_failure(event_type, obj_data)
    if event_type == "response.incomplete":
        raise RuntimeError(
            f"codex response incomplete: {_incomplete_details(obj_data)}"
        )
    if event_type in {"response.created", "response.in_progress"}:
        response = json_dict(obj_data.get("response"))
        if response:
            state.envelope = response
    elif event_type == "response.completed":
        response = _completed_response(obj_data)
        if response:
            state.envelope = response
            state.completed = True
    elif event_type == "response.output_item.done":
        item = json_dict(obj_data.get("item"))
        if item:
            state.collected_items.append(item)
    elif event_type == "response.output_text.delta":
        delta = obj_data.get("delta")
        if isinstance(delta, str):
            state.output_text_chunks.append(delta)


def _finalize_stream(state: _StreamState) -> JsonDict:
    if not state.completed:
        envelope_detail = (
            " after a response envelope" if state.envelope else ""
        )
        raise CodexRetryableStreamError(
            "codex HTTP-200 stream ended before response.completed"
            f"{envelope_detail}"
        )
    if state.collected_items:
        state.envelope["output"] = state.collected_items
    elif state.output_text_chunks and not state.envelope.get("output"):
        state.envelope["output"] = [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "".join(state.output_text_chunks),
            }],
        }]
    return state.envelope


def collect_stream(response: httpx.Response) -> JsonDict:
    """Collect one complete Codex SSE response or raise a terminal error."""
    state = _StreamState()
    for event_name, obj in iter_sse(response):
        obj_data = json_dict(obj)
        if obj_data:
            _apply_stream_event(state, event_name, obj_data)
    return _finalize_stream(state)
