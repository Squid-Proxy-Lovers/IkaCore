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

# pyright: strict
# pyright: reportUnusedFunction=false

from __future__ import annotations

import email.utils
import json
import logging
import random
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any, Optional, cast

import httpx

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value
from IkaCore.cli_output import OutputType, get_cli_output

from ..base import BareBoneModel
from ..codex_constants import CODEX_API_URL
from ..model_metadata import CODEX_KNOWN_MODELS
from ..request_interface import (
    IkaRequestCancelled,
    agent_tools_for_payload,
    register_request_abort_callback,
    request_cancelled,
)
from .codex_responses import codex_responses_fill_payload

LOG = logging.getLogger(__name__)

ProviderRequest = tuple[str, dict[str, str], JsonDict]
ProviderRound = tuple[str, Optional[str], list[JsonDict], int]
MessageList = list[JsonDict]


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


class _CodexRetryableStreamError(RuntimeError):
    """Stream-level codex failure that should be retried (server-side hiccup,
    transient internal error, etc.). The outer retry loop catches this
    specifically; a plain RuntimeError indicates a non-retryable error."""


# Error type/code strings that the OpenAI Responses backend uses for transient
# faults. Anything not in here is treated as a caller error and is not retried.
_RETRYABLE_STREAM_ERROR_TYPES = {
    "server_error", "internal_error", "engine_error",
    "rate_limit_exceeded", "rate_limit_error", "overloaded_error",
    "server_is_overloaded",
}
_RETRYABLE_STREAM_ERROR_CODES = {
    "server_error", "internal_error",
    "rate_limit_exceeded", "model_overloaded", "server_is_overloaded",
}


def _classify_stream_error(err_obj: object) -> bool:
    """Return True if this stream-level error looks transient."""
    if not isinstance(err_obj, dict):
        # Unknown shape — be conservative: don't retry, surface to caller.
        return False
    error_data = cast(JsonDict, err_obj)
    et = string_value(error_data.get("type")).lower()
    code = string_value(error_data.get("code")).lower()
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
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, parsed.timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        pass
    return float(default)


def _sleep_before_retry(
    log_message: str,
    delay: float,
    attempt: int,
    max_retries: int,
    *log_args: Any,
) -> None:
    jitter = random.uniform(0.0, min(delay * 0.25, 5.0)) if delay > 0 else 0.0
    sleep_delay = delay + jitter
    normalized_message = log_message.lower()
    if "request timeout" in normalized_message:
        retry_notice = "Codex request exceeded its local timeout"
    elif "http error" in normalized_message:
        retry_notice = "Codex transport error"
    else:
        retry_notice = "Transient Codex backend failure"
    get_cli_output().emit(
        OutputType.AGENT_RESPONSE,
        f"{retry_notice}; retrying in {sleep_delay:.1f}s "
        f"(attempt {attempt + 1}/{max_retries})",
        ["API"],
        step=0,
    )
    LOG.warning(
        f"{log_message}; retrying in %.1fs (attempt %d/%d)",
        *log_args,
        sleep_delay,
        attempt + 1,
        max_retries,
    )
    time.sleep(sleep_delay)


def is_codex_url(api_url: Optional[str]) -> bool:
    if not api_url:
        return False
    return "chatgpt.com/backend-api/codex" in api_url.lower()


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def build_codex_request(
    barebone_model: BareBoneModel,
    messages: MessageList,
    message_history: JsonDict,
) -> ProviderRequest:
    """Return (url, headers, payload) for a codex Responses call.

    ``barebone_model.api_key`` is treated as a literal bearer — same convention
    as every other IkaCore provider (openai, deepseek, anthropic, ...). Callers
    that want to resolve the bearer from ``~/.codex/auth.json`` with automatic
    refresh can call the opt-in helper at ``IkaModel.codex.auth.get_bearer()``
    and pass its return as ``api_key``. This module never reads files or env
    vars on its own.
    """
    payload = codex_responses_fill_payload(
        barebone_model,
        messages,
        message_history,
        agent_tools=agent_tools_for_payload(barebone_model),
    )

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
        data: JsonDict,
        status_code: int = 200,
        headers: Optional[dict[str, str]] = None,
        text_fallback: str = "",
    ) -> None:
        self._data = data
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self.text = text_fallback or json.dumps(data)
        self.content = self.text.encode("utf-8", "replace")

    def json(self) -> JsonDict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"codex backend returned {self.status_code}",
                request=None,  # type: ignore[arg-type]
                response=self,  # type: ignore[arg-type]
            )


def _iter_sse(response: httpx.Response, absolute_deadline: Optional[float] = None) -> Iterator[tuple[Optional[str], object]]:
    """
    Yield (event_name, parsed_json) from a Server-Sent Events stream.

    SSE frames are separated by blank lines. Each frame is a sequence of
    ``event:``/``data:`` lines. We only care about the data payload (JSON);
    the event name is mirrored from the ``type`` field of the JSON itself
    on the codex backend, but we honor an explicit ``event:`` line too.
    """
    current_event: Optional[str] = None
    for line in response.iter_lines():
        if absolute_deadline is not None and time.monotonic() >= absolute_deadline:
            raise httpx.ReadTimeout("codex request exceeded absolute stream deadline")
        if request_cancelled():
            raise IkaRequestCancelled("codex request cancelled")
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


def _collect_stream(response: httpx.Response, absolute_deadline: Optional[float] = None) -> JsonDict:
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
    envelope: JsonDict = {}
    collected_items: list[JsonDict] = []
    output_text_chunks: list[str] = []
    completed = False

    for _event, obj in _iter_sse(response, absolute_deadline=absolute_deadline):
        obj_data = json_dict(obj)
        if not obj_data:
            continue
        t = obj_data.get("type", "")
        if t == "response.failed" or t == "response.error":
            err = obj_data.get("error") or json_dict(obj_data.get("response")).get("error")
            if _classify_stream_error(err):
                raise _CodexRetryableStreamError(f"codex stream transient failure: {err}")
            raise RuntimeError(f"codex response failed: {err}")
        elif t in ("response.created", "response.in_progress"):
            resp = json_dict(obj_data.get("response"))
            if resp:
                envelope = resp
        elif t == "response.completed":
            completed = True
            resp = json_dict(obj_data.get("response"))
            if resp:
                # Keep the latest envelope (it has the final ``usage`` and
                # any updated status fields), but its output[] is empty so
                # we'll fill it from collected_items below.
                envelope = resp
        elif t == "response.output_item.done":
            item = json_dict(obj_data.get("item"))
            if item:
                collected_items.append(item)
        elif t == "response.output_text.delta":
            delta = obj_data.get("delta")
            if isinstance(delta, str):
                output_text_chunks.append(delta)

    if not completed:
        raise _CodexRetryableStreamError("codex stream ended before response.completed")
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


def _codex_rate_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value for key, value in response.headers.items()
        if key.lower().startswith("x-codex-") or "ratelimit" in key.lower()
    }


def _handle_codex_429(response: httpx.Response, body: str, attempt: int, max_retries: int, wait_seconds: int) -> None:
    if attempt >= max_retries - 1:
        raise RuntimeError(f"codex backend returned 429 after {max_retries} attempts: {body[:500]}")
    retry_after = _parse_retry_after(response.headers.get("retry-after"), wait_seconds)
    LOG.warning(
        "codex backend 429 (rate-limited); sleeping %.1fs (attempt %d/%d)",
        retry_after, attempt + 1, max_retries,
    )
    time.sleep(retry_after)


def _handle_codex_5xx(status: int, body: str, backoff: float, attempt: int, max_retries: int) -> None:
    if attempt >= max_retries - 1:
        raise RuntimeError(f"codex backend returned {status} after {max_retries} attempts: {body[:500]}")
    _sleep_before_retry(
        "codex backend %d (transient)",
        backoff, attempt, max_retries, status,
    )


def _handle_codex_non_200(
    response: httpx.Response,
    body: str,
    backoff: float,
    attempt: int,
    max_retries: int,
    wait_seconds: int,
) -> None:
    status = response.status_code
    if status == 401:
        raise RuntimeError(
            f"codex backend returned 401 (bearer rejected): {body[:500]}. "
            "Caller may need to refresh the token and retry."
        )
    if status == 429:
        _handle_codex_429(response, body, attempt, max_retries, wait_seconds)
        return
    if 500 <= status < 600:
        _handle_codex_5xx(status, body, backoff, attempt, max_retries)
        return
    raise RuntimeError(f"codex backend returned {status}: {body[:2000]}")


def _request_codex_once(
    api_url: str,
    headers: dict[str, str],
    payload: JsonDict,
    timeout: float,
    backoff: float,
    attempt: int,
    max_retries: int,
    wait_seconds: int,
) -> Optional[_CodexResponseShim]:
    request_started = time.monotonic()
    with httpx.stream("POST", api_url, headers=headers, json=payload, timeout=timeout) as response:
        close_response = getattr(response, "close", None)
        unregister_abort = register_request_abort_callback(close_response) if callable(close_response) else None
        request_deadline_reached = threading.Event()
        deadline_timer: Optional[threading.Timer] = None

        def _expire_request() -> None:
            request_deadline_reached.set()
            if callable(close_response):
                close_response()

        try:
            if request_cancelled():
                raise IkaRequestCancelled("codex request cancelled")

            rate_headers = _codex_rate_headers(response)
            if response.status_code == 200:
                absolute_deadline = request_started + float(timeout) if timeout > 0 else None
                if timeout > 0 and callable(close_response):
                    remaining = max(0.0, absolute_deadline - time.monotonic())
                    if remaining <= 0:
                        raise httpx.ReadTimeout(
                            f"codex request exceeded absolute {timeout}s deadline before streaming"
                        )
                    deadline_timer = threading.Timer(remaining, _expire_request)
                    deadline_timer.daemon = True
                    deadline_timer.start()
                try:
                    data = _collect_stream(response, absolute_deadline=absolute_deadline)
                except (httpx.HTTPError, RuntimeError, ValueError, TypeError, OSError) as error:
                    if request_cancelled():
                        raise IkaRequestCancelled("codex request cancelled") from error
                    if request_deadline_reached.is_set():
                        raise httpx.ReadTimeout(
                            f"codex request exceeded absolute {timeout}s deadline"
                        ) from error
                    raise
                if request_cancelled():
                    raise IkaRequestCancelled("codex request cancelled")
                if request_deadline_reached.is_set():
                    raise httpx.ReadTimeout(
                        f"codex request exceeded absolute {timeout}s deadline"
                    )
                return _CodexResponseShim(data=data, status_code=200, headers=rate_headers)

            body = response.read().decode("utf-8", "replace")
            if request_cancelled():
                raise IkaRequestCancelled("codex request cancelled")
            _handle_codex_non_200(response, body, backoff, attempt, max_retries, wait_seconds)
        finally:
            if deadline_timer is not None:
                deadline_timer.cancel()
            if unregister_abort is not None:
                try:
                    unregister_abort()
                except Exception:
                    pass
    return None


def _retry_codex_exception(message: str, backoff: float, attempt: int, max_retries: int, error: Exception) -> None:
    if attempt >= max_retries - 1:
        raise error
    _sleep_before_retry(message, backoff, attempt, max_retries, error)


def _streaming_codex_payload(payload: JsonDict) -> JsonDict:
    # Defensive: the codex backend rejects anything but stream:true. Make this
    # an invariant here even if a caller pre-built the payload differently.
    normalized: JsonDict = dict(payload)
    normalized["stream"] = True
    return normalized


def _request_codex_with_retries(
    api_url: str,
    headers: dict[str, str],
    payload: JsonDict,
    timeout: float,
    max_retries: int,
    wait_seconds: int,
) -> _CodexResponseShim:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        backoff = wait_seconds * (2 ** attempt)
        try:
            if request_cancelled():
                raise IkaRequestCancelled("codex request cancelled")

            response = _request_codex_once(
                api_url,
                headers,
                payload,
                timeout,
                backoff,
                attempt,
                max_retries,
                wait_seconds,
            )
            if response is not None:
                return response
            continue

        except IkaRequestCancelled:
            raise
        except _CodexRetryableStreamError as error:
            last_exc = error
            _retry_codex_exception(
                "codex stream transient failure: %s",
                backoff, attempt, max_retries, error,
            )
        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as error:
            if request_cancelled():
                raise IkaRequestCancelled("codex request cancelled") from error
            last_exc = error
            _retry_codex_exception(
                "codex request timeout: %s",
                backoff, attempt, max_retries, error,
            )
        except httpx.HTTPError as error:
            if request_cancelled():
                raise IkaRequestCancelled("codex request cancelled") from error
            last_exc = error
            _retry_codex_exception(
                "codex http error: %s",
                backoff, attempt, max_retries, error,
            )

    # Loop exhausted without a return — surface the last seen exception.
    if last_exc:
        raise last_exc
    raise RuntimeError("codex request failed without specific exception")


def request_codex(
    api_url: str,
    headers: dict[str, str],
    payload: JsonDict,
    timeout: float = 900.0,
    max_retries: int = 3,
    wait_seconds: int = 10,
) -> _CodexResponseShim:
    """POST to the Codex Responses endpoint and return a Response-shaped shim."""
    return _request_codex_with_retries(
        api_url,
        headers,
        _streaming_codex_payload(payload),
        timeout,
        max_retries,
        wait_seconds,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_codex_response(
    data: JsonDict,
    model_id: str,
) -> ProviderRound:
    """
    Parse a final codex response dict (the ``response.completed`` event's
    ``response`` payload) into the internal (content, reasoning, tool_calls,
    tokens) tuple used by IkaCore.
    """
    content = ""
    tool_calls: list[JsonDict] = []
    reasoning_content: Optional[str] = None

    raw_output: object = data.get("output", []) or []
    output = cast(list[object], raw_output) if isinstance(raw_output, list) else []
    for item in output:
        item_data = json_dict(item)
        item_type = item_data.get("type", "")
        if item_type == "message":
            raw_blocks: object = item_data.get("content", []) or []
            blocks = cast(list[object], raw_blocks) if isinstance(raw_blocks, list) else []
            for block in blocks:
                block_data = json_dict(block)
                if block_data.get("type") == "output_text":
                    content += string_value(block_data.get("text"))
        elif item_type == "function_call":
            tool_calls.append({
                "id": item_data.get("call_id", item_data.get("id", "")),
                "type": "function",
                "function": {
                    "name": item_data.get("name", ""),
                    "arguments": item_data.get("arguments", "{}"),
                },
            })
        elif item_type == "reasoning":
            # The codex backend serves reasoning summaries (and optionally
            # encrypted_content). We only keep the summary text here; the
            # encrypted_content carry-over is a future optimization.
            raw_summary: object = item_data.get("summary") or []
            summary = cast(list[object], raw_summary) if isinstance(raw_summary, list) else []
            if summary:
                parts = [string_value(cast(JsonDict, s).get("text")) for s in summary if isinstance(s, dict)]
                joined = "\n".join(p for p in parts if p)
                if joined:
                    reasoning_content = joined

    if not content and data.get("output_text"):
        content = string_value(data.get("output_text"))

    usage = json_dict(data.get("usage"))
    fallback_tokens = _token_count(usage.get("input_tokens")) + _token_count(usage.get("output_tokens"))
    tokens = _token_count(usage.get("total_tokens")) or fallback_tokens

    return content, reasoning_content, tool_calls, tokens


# ---------------------------------------------------------------------------
# Tool message recording
# ---------------------------------------------------------------------------

def append_codex_tool_messages(
    messages: MessageList,
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: list[JsonDict],
    tool_messages: MessageList,
    tokens: int,
    repeated_warning_msg: str = "",
) -> None:
    """
    Record the assistant's turn and its tool outputs in the running
    conversation. Matches the openai_responses convention: the codex
    payload builder converts role=='tool' messages into the
    ``function_call_output`` form when it next reads ``messages``.
    """
    assistant_msg: JsonDict = {"role": "assistant", "content": content}
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
        history_section(message_history, "messages")[msg_id] = {
            "message": json.dumps(assistant_msg),
            "tokens": tokens,
            "type": "assistant_with_tools",
        }

    for tool_msg in tool_messages:
        msg_id = str(uuid.uuid4())
        history_section(message_history, "messages")[msg_id] = {
            "message": json.dumps(tool_msg),
            "tokens": 0,
            "type": "tool",
        }
