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

import json
import logging
import time
import uuid
from typing import Any, Optional, cast

import httpx

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value

from ..base import BareBoneModel
from ..codex_constants import CODEX_API_URL
from ..http_config import shared_ssl_context
from ..model_metadata import CODEX_KNOWN_MODELS
from ..request_interface import agent_tools_for_payload
from ..retry_policy import backoff_seconds, is_retryable_status, parse_retry_after, retry_delay, retry_hint_seconds
from .codex_responses import codex_responses_fill_payload
from .codex_stream import CodexRetryableStreamError as _CodexRetryableStreamError
from .codex_stream import classify_stream_error as _classify_stream_error
from .codex_stream import collect_stream as _collect_stream
from .codex_stream import iter_sse as _iter_sse

LOG = logging.getLogger(__name__)

ProviderRequest = tuple[str, dict[str, str], JsonDict]
ProviderRound = tuple[str, Optional[str], list[JsonDict], int]
MessageList = list[JsonDict]

__all__ = [
    "_CodexRetryableStreamError",
    "_classify_stream_error",
    "_collect_stream",
    "_iter_sse",
]


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


# Kept under its original name for callers/tests; the parser now lives in retry_policy.
_parse_retry_after = parse_retry_after


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
    return "/backend-api/codex/" in api_url.lower()


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


def _codex_rate_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value for key, value in response.headers.items()
        if key.lower().startswith("x-codex-") or "ratelimit" in key.lower()
    }


def _json_or_none(body: str) -> object:
    try:
        return json.loads(body)
    except ValueError:
        return None


def _handle_codex_429(response: httpx.Response, body: str, attempt: int, max_retries: int, wait_seconds: float) -> None:
    if attempt >= max_retries - 1:
        raise RuntimeError(f"codex backend returned 429 after {max_retries} attempts: {body[:500]}")
    retry_after = retry_delay(attempt, retry_hint_seconds(response.headers, _json_or_none(body)), wait_seconds)
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
    wait_seconds: float,
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
    if is_retryable_status(status):  # 408, 409, 5xx (429 handled above)
        _handle_codex_5xx(status, body, backoff, attempt, max_retries)
        return
    raise RuntimeError(f"codex backend returned {status}: {body[:2000]}")


def _open_codex_stream(
    client: Optional[httpx.Client], api_url: str, headers: dict[str, str], payload: JsonDict, timeout: float
) -> Any:
    # A caller-provided client reuses pooled keep-alive connections; otherwise open a one-off stream.
    if client is not None:
        return client.stream("POST", api_url, headers=headers, json=payload, timeout=timeout)
    return httpx.stream("POST", api_url, headers=headers, json=payload, timeout=timeout, verify=shared_ssl_context())


def _request_codex_once(
    api_url: str,
    headers: dict[str, str],
    payload: JsonDict,
    timeout: float,
    backoff: float,
    attempt: int,
    max_retries: int,
    wait_seconds: float,
    client: Optional[httpx.Client] = None,
) -> Optional[_CodexResponseShim]:
    with _open_codex_stream(client, api_url, headers, payload, timeout) as response:
        rate_headers = _codex_rate_headers(response)
        if response.status_code == 200:
            data = _collect_stream(response)
            return _CodexResponseShim(data=data, status_code=200, headers=rate_headers)

        body = response.read().decode("utf-8", "replace")
        _handle_codex_non_200(response, body, backoff, attempt, max_retries, wait_seconds)
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
    wait_seconds: float,
    client: Optional[httpx.Client] = None,
) -> _CodexResponseShim:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        backoff = backoff_seconds(attempt, wait_seconds)
        try:
            response = _request_codex_once(
                api_url,
                headers,
                payload,
                timeout,
                backoff,
                attempt,
                max_retries,
                wait_seconds,
                client,
            )
            if response is not None:
                return response
            continue

        except _CodexRetryableStreamError as error:
            last_exc = error
            _retry_codex_exception(
                "codex stream transient failure: %s",
                backoff, attempt, max_retries, error,
            )
        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as error:
            last_exc = error
            _retry_codex_exception(
                "codex request timeout: %s",
                backoff, attempt, max_retries, error,
            )
        except httpx.HTTPError as error:
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
    wait_seconds: float = 1.0,
    client: Optional[httpx.Client] = None,
) -> _CodexResponseShim:
    """POST to the Codex Responses endpoint and return a Response-shaped shim."""
    return _request_codex_with_retries(
        api_url,
        headers,
        _streaming_codex_payload(payload),
        timeout,
        max_retries,
        wait_seconds,
        client,
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
