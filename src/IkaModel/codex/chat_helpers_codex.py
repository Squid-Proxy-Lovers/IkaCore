"""
Codex provider request/response helpers.

Three pieces:
- ``build_codex_request`` builds (url, headers, payload) for the codex backend.
- ``request_codex`` issues the POST, walks the SSE stream until completion,
  and synthesizes a Response-shaped shim whose ``.json()`` mirrors what a
  non-streaming Responses call would have returned. The codex backend
  REQUIRES streaming (it returns 400 "Stream must be set to true" if you
  ask for stream:false), so streaming is unavoidable on the wire. Callers
  upstream of this module never see deltas — they just get the final dict.
- ``parse_codex_response`` and ``append_codex_tool_messages`` mirror the
  openai_responses helpers so the rest of IkaCore consumes codex output
  identically to any other provider.
"""

from __future__ import annotations

import email.utils
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..codex_constants import CODEX_API_URL
from ..request_interface import _apply_tools_filter_for_payload, _restore_tools_after_payload
from .codex_responses import codex_responses_fill_payload, CODEX_KNOWN_MODELS

LOG = logging.getLogger(__name__)


class _CodexRetryableStreamError(RuntimeError):
    """Stream-level codex failure that should be retried (server-side hiccup,
    transient internal error, etc.). The outer retry loop catches this
    specifically; a plain RuntimeError indicates a non-retryable error."""


# Error type/code strings that the OpenAI Responses backend uses for transient
# faults. Anything not in here is treated as a caller error and is not retried.
_RETRYABLE_STREAM_ERROR_TYPES = {
    "server_error", "internal_error", "engine_error",
    "rate_limit_exceeded", "rate_limit_error", "overloaded_error",
}
_RETRYABLE_STREAM_ERROR_CODES = {
    "server_error", "internal_error",
    "rate_limit_exceeded", "model_overloaded",
}


def _classify_stream_error(err_obj: Any) -> bool:
    """Return True if this stream-level error looks transient."""
    if not isinstance(err_obj, dict):
        # Unknown shape — be conservative: don't retry, surface to caller.
        return False
    et = str(err_obj.get("type") or "").lower()
    code = str(err_obj.get("code") or "").lower()
    return et in _RETRYABLE_STREAM_ERROR_TYPES or code in _RETRYABLE_STREAM_ERROR_CODES


def _parse_retry_after(header_value: Optional[str], default: float) -> float:
    """Parse a Retry-After header, supporting both integer-seconds and HTTP-date
    forms (RFC 7231 §7.1.3). Falls back to ``default`` on any failure.

    Naive datetimes (RFC 5322 ``-0000`` zone, meaning "local time of the source
    is unknown") are treated as UTC, otherwise ``.timestamp()`` would silently
    use the local timezone and produce a wildly wrong wait."""
    if not header_value:
        return float(default)
    # Form 1: integer seconds (common case)
    try:
        return max(0.0, float(header_value))
    except (TypeError, ValueError):
        pass
    # Form 2: HTTP-date (rare but legal — e.g. "Wed, 21 Oct 2015 07:28:00 GMT")
    try:
        from datetime import timezone
        parsed = email.utils.parsedate_to_datetime(header_value)
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, parsed.timestamp() - time.time())
    except Exception:
        pass
    return float(default)


def _sleep_before_retry(
    log_message: str,
    delay: float,
    attempt: int,
    max_retries: int,
    *log_args: Any,
) -> None:
    LOG.warning(
        f"{log_message}; retrying in %.1fs (attempt %d/%d)",
        *log_args,
        delay,
        attempt + 1,
        max_retries,
    )
    time.sleep(delay)


def is_codex_url(api_url: Optional[str]) -> bool:
    if not api_url:
        return False
    return "chatgpt.com/backend-api/codex" in api_url.lower()


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def build_codex_request(
    barebone_model: Any,
    messages: List[dict],
    message_history: dict,
) -> Tuple[str, Dict[str, str], dict]:
    """Return (url, headers, payload) for a codex Responses call.

    ``barebone_model.api_key`` is treated as a literal bearer — same convention
    as every other IkaCore provider (openai, deepseek, anthropic, ...). Callers
    that want to resolve the bearer from ``~/.codex/auth.json`` with automatic
    refresh can call the opt-in helper at ``IkaModel.codex.auth.get_bearer()``
    and pass its return as ``api_key``. This module never reads files or env
    vars on its own.
    """
    _apply_tools_filter_for_payload(barebone_model)
    payload = codex_responses_fill_payload(barebone_model, messages, message_history)
    _restore_tools_after_payload(barebone_model)

    api_key = barebone_model.api_key
    if not api_key:
        raise ValueError(
            "codex provider requires a bearer in BareBoneModel.api_key. "
            "Pass the value from a CODEX_BEARER env var, or call "
            "IkaModel.codex.auth.get_bearer() to source it from ~/.codex/auth.json."
        )

    api_url = barebone_model.api_url or CODEX_API_URL
    if not is_codex_url(api_url):
        # Defensive: if the caller pointed at a generic openai URL but routed
        # here anyway, normalize to the codex endpoint.
        api_url = CODEX_API_URL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    if getattr(barebone_model, "model_id", None) and barebone_model.model_id not in CODEX_KNOWN_MODELS:
        LOG.warning(
            "Codex provider invoked with model_id=%r which is not in the known "
            "codex slug list %s. The backend will reject it with a 400 if it's "
            "truly unsupported.",
            barebone_model.model_id, sorted(CODEX_KNOWN_MODELS),
        )

    return api_url, headers, payload


# ---------------------------------------------------------------------------
# SSE collection → synthesized non-streaming response
# ---------------------------------------------------------------------------

class _CodexResponseShim:
    """
    httpx.Response-compatible facade returned by request_codex.

    The codex backend only speaks SSE; ``request_codex`` walks the stream
    and accumulates output_text deltas plus the final ``response.completed``
    event's payload. The rest of IkaCore expects a httpx.Response-like object
    it can call ``.json()`` on, so we wrap the collected dict here.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        text_fallback: str = "",
    ):
        self._data = data
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self.text = text_fallback or json.dumps(data)
        self.content = self.text.encode("utf-8", "replace")

    def json(self) -> Dict[str, Any]:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"codex backend returned {self.status_code}",
                request=None,  # type: ignore[arg-type]
                response=self,  # type: ignore[arg-type]
            )


def _iter_sse(response: httpx.Response):
    """
    Yield (event_name, parsed_json) from a Server-Sent Events stream.

    SSE frames are separated by blank lines. Each frame is a sequence of
    ``event:``/``data:`` lines. We only care about the data payload (JSON);
    the event name is mirrored from the ``type`` field of the JSON itself
    on the codex backend, but we honor an explicit ``event:`` line too.
    """
    current_event = None
    for line in response.iter_lines():
        if line is None:
            continue
        # httpx strips the trailing newline but leaves the line as-is.
        if not line:
            current_event = None
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                yield current_event, json.loads(data)
            except json.JSONDecodeError:
                LOG.debug("Skipping unparseable SSE data line: %r", data[:200])
                continue


def _collect_stream(response: httpx.Response) -> Dict[str, Any]:
    """
    Walk the SSE stream and return the final response dict.

    Codex SSE delivers the response in pieces:
    - ``response.created`` / ``response.in_progress`` carry the envelope
      (id, model, usage, status, ...) with an empty ``output: []``.
    - Each output item (assistant message, function_call, reasoning) arrives
      in its own ``response.output_item.added`` → ... → ``response.output_item.done``
      sequence. The ``done`` event holds the final item shape.
    - ``response.completed`` arrives last, but its ``response.output`` is
      typically still empty on the codex backend. We use it for its envelope
      (especially the final ``usage``) and graft the collected items in.

    Text deltas via ``response.output_text.delta`` are tracked only as a
    fallback for cases where ``output_item.done`` doesn't arrive (shouldn't
    happen in practice but keeps us defensive).
    """
    envelope: Dict[str, Any] = {}
    collected_items: List[Dict[str, Any]] = []
    output_text_chunks: List[str] = []

    for _event, obj in _iter_sse(response):
        if not isinstance(obj, dict):
            continue
        t = obj.get("type", "")
        if t == "response.failed" or t == "response.error":
            err = obj.get("error") or obj.get("response", {}).get("error")
            if _classify_stream_error(err):
                raise _CodexRetryableStreamError(f"codex stream transient failure: {err}")
            raise RuntimeError(f"codex response failed: {err}")
        elif t in ("response.created", "response.in_progress"):
            resp = obj.get("response")
            if isinstance(resp, dict):
                envelope = resp
        elif t == "response.completed":
            resp = obj.get("response")
            if isinstance(resp, dict):
                # Keep the latest envelope (it has the final ``usage`` and
                # any updated status fields), but its output[] is empty so
                # we'll fill it from collected_items below.
                envelope = resp
        elif t == "response.output_item.done":
            item = obj.get("item")
            if isinstance(item, dict):
                collected_items.append(item)
        elif t == "response.output_text.delta":
            delta = obj.get("delta")
            if isinstance(delta, str):
                output_text_chunks.append(delta)

    if not envelope:
        raise RuntimeError("codex stream ended without a response envelope")

    # Replace the (typically empty) envelope output with the items we
    # collected from output_item.done events.
    if collected_items:
        envelope["output"] = collected_items
    elif output_text_chunks and not envelope.get("output"):
        # Last-resort fallback: synthesize a message item from raw text deltas.
        envelope["output"] = [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "".join(output_text_chunks)}],
        }]

    return envelope


def request_codex(
    api_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float = 900.0,
    max_retries: int = 3,
    wait_seconds: int = 10,
) -> _CodexResponseShim:
    """
    POST to the codex Responses endpoint, drain the SSE stream, return a
    Response-shaped object whose ``.json()`` is the final response dict.

    Streaming is forced by the codex backend; we don't expose deltas to
    callers — the rest of IkaCore is built around single-shot JSON
    responses, and streaming the UI is a future enhancement.

    Retry policy:
      - 200 → succeed
      - 401 → no retry; caller refreshes bearer and retries
      - 429 → retry honoring Retry-After (with defensive parsing)
      - 5xx → retry with exponential backoff (transient backend)
      - Other 4xx → no retry; caller bug (bad payload, unknown model, etc.)
      - Network timeouts / httpx.HTTPError → retry with backoff
      - Stream-level ``response.failed`` events → retried if classified as
        transient (server_error / internal_error / overloaded), else bubble.
    """
    # Defensive: the codex backend rejects anything but stream:true. Make this
    # an invariant here even if a caller pre-built the payload differently.
    payload = dict(payload)
    payload["stream"] = True

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        is_last_attempt = attempt >= max_retries - 1
        backoff = wait_seconds * (2 ** attempt)
        try:
            with httpx.stream(
                "POST", api_url, headers=headers, json=payload, timeout=timeout
            ) as resp:
                rate_headers = {
                    k: v for k, v in resp.headers.items()
                    if k.lower().startswith("x-codex-") or "ratelimit" in k.lower()
                }
                status = resp.status_code

                if status == 200:
                    # Stream the SSE body. _collect_stream may raise either
                    # _CodexRetryableStreamError (transient — retry below) or
                    # plain RuntimeError (caller error — bubble immediately).
                    data = _collect_stream(resp)
                    return _CodexResponseShim(
                        data=data, status_code=200, headers=rate_headers,
                    )

                # All non-200 paths share a body read for error reporting.
                body = resp.read().decode("utf-8", "replace")

                if status == 401:
                    raise RuntimeError(
                        f"codex backend returned 401 (bearer rejected): {body[:500]}. "
                        "Caller may need to refresh the token and retry."
                    )

                if status == 429:
                    if is_last_attempt:
                        raise RuntimeError(
                            f"codex backend returned 429 after {max_retries} attempts: {body[:500]}"
                        )
                    retry_after = _parse_retry_after(
                        resp.headers.get("retry-after"), wait_seconds
                    )
                    LOG.warning(
                        "codex backend 429 (rate-limited); sleeping %.1fs (attempt %d/%d)",
                        retry_after, attempt + 1, max_retries,
                    )
                    time.sleep(retry_after)
                    continue

                if 500 <= status < 600:
                    if is_last_attempt:
                        raise RuntimeError(
                            f"codex backend returned {status} after {max_retries} attempts: {body[:500]}"
                        )
                    _sleep_before_retry(
                        "codex backend %d (transient)",
                        backoff, attempt, max_retries, status,
                    )
                    continue

                # Other 4xx — caller-side error (bad payload, unknown model,
                # missing field). Not retryable.
                raise RuntimeError(
                    f"codex backend returned {status}: {body[:2000]}"
                )

        except _CodexRetryableStreamError as e:
            last_exc = e
            if is_last_attempt:
                raise
            _sleep_before_retry(
                "codex stream transient failure: %s",
                backoff, attempt, max_retries, e,
            )
            continue
        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_exc = e
            if is_last_attempt:
                raise
            _sleep_before_retry(
                "codex request timeout: %s",
                backoff, attempt, max_retries, e,
            )
            continue
        except httpx.HTTPError as e:
            last_exc = e
            if is_last_attempt:
                raise
            _sleep_before_retry(
                "codex http error: %s",
                backoff, attempt, max_retries, e,
            )
            continue

    # Loop exhausted without a return — surface the last seen exception.
    if last_exc:
        raise last_exc
    raise RuntimeError("codex request failed without specific exception")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_codex_response(
    data: dict,
    model_id: str,
) -> Tuple[str, Optional[str], List[dict], int]:
    """
    Parse a final codex response dict (the ``response.completed`` event's
    ``response`` payload) into the internal (content, reasoning, tool_calls,
    tokens) tuple used by IkaCore.
    """
    content = ""
    tool_calls: List[dict] = []
    reasoning_content: Optional[str] = None

    output = data.get("output", []) or []
    for item in output:
        item_type = item.get("type", "")
        if item_type == "message":
            for block in item.get("content", []) or []:
                if block.get("type") == "output_text":
                    content += block.get("text", "")
        elif item_type == "function_call":
            tool_calls.append({
                "id": item.get("call_id", item.get("id", "")),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })
        elif item_type == "reasoning":
            # The codex backend serves reasoning summaries (and optionally
            # encrypted_content). We only keep the summary text here; the
            # encrypted_content carry-over is a future optimization.
            summary = item.get("summary") or []
            if summary:
                parts = [s.get("text", "") for s in summary if isinstance(s, dict)]
                joined = "\n".join(p for p in parts if p)
                if joined:
                    reasoning_content = joined

    if not content and data.get("output_text"):
        content = data["output_text"]

    usage = data.get("usage", {}) or {}
    tokens = usage.get("total_tokens") or (
        (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
    )

    return content, reasoning_content, tool_calls, tokens


# ---------------------------------------------------------------------------
# Tool message recording
# ---------------------------------------------------------------------------

def append_codex_tool_messages(
    messages: List[dict],
    message_history: dict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: List[dict],
    tool_messages: List[dict],
    tokens: int,
    repeated_warning_msg: str = "",
) -> None:
    """
    Record the assistant's turn and its tool outputs in the running
    conversation. Matches the openai_responses convention: the codex
    payload builder converts role=='tool' messages into the
    ``function_call_output`` form when it next reads ``messages``.
    """
    assistant_msg: dict = {"role": "assistant", "content": content}
    if reasoning_content:
        assistant_msg["reasoning_content"] = reasoning_content
    if executed_tool_call_list:
        assistant_msg["tool_calls"] = executed_tool_call_list

    messages.append(assistant_msg)
    messages.extend(tool_messages)

    if repeated_warning_msg:
        messages.append({"role": "user", "content": repeated_warning_msg})

    if executed_tool_call_list:
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {
            "message": json.dumps(assistant_msg),
            "tokens": tokens,
            "type": "assistant_with_tools",
        }

    for tool_msg in tool_messages:
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {
            "message": json.dumps(tool_msg),
            "tokens": 0,
            "type": "tool",
        }
