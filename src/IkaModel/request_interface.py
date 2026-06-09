# pyright: strict
# pyright: reportPrivateUsage=false, reportUnusedFunction=false

import asyncio
import copy
import json
import logging
import time
from typing import Any, Optional, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from IkaCore.agent_runtime_payloads import JsonDict, string_value
from IkaCore.cli_output import OutputType, get_cli_output

from .model_metadata import get_max_tokens_for_model, get_provider_for_model

LOG = logging.getLogger(__name__)

_SENSITIVE_HEADER_NAMES = {
    "authorization",
    "x-api-key",
    "x-goog-api-key",
    "api-key",
}
_SENSITIVE_QUERY_NAMES = {
    "access_token",
    "api_key",
    "code",
    "key",
    "refresh_token",
    "token",
}
_RETRYABLE_UNEXPECTED_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError)


class IkaAPIError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class IkaRateLimitError(IkaAPIError):
    pass


class IkaTimeoutError(IkaAPIError):
    pass


class IkaHTTPError(IkaAPIError):
    pass


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    safe_headers: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _SENSITIVE_HEADER_NAMES:
            safe_headers[key] = "Bearer ***" if key.lower() == "authorization" else "***"
        else:
            safe_headers[key] = value
    return safe_headers


def _redact_url(api_url: str) -> str:
    try:
        parts = urlsplit(api_url)
        query = urlencode(
            [
                (key, "***" if key.lower() in _SENSITIVE_QUERY_NAMES else value)
                for key, value in parse_qsl(parts.query, keep_blank_values=True)
            ],
            doseq=True,
        )
        return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))
    except (TypeError, ValueError):
        return api_url


def _response_text_preview(response: httpx.Response, limit: int = 500) -> str:
    return response.text[:limit] if hasattr(response, "text") else str(response.status_code)


def _response_json_preview(response: httpx.Response, limit: int = 1000) -> Optional[str]:
    try:
        return json.dumps(response.json(), indent=2)[:limit]
    except (TypeError, ValueError):
        return None


def _error_text_from_response(response: httpx.Response, limit: int = 500) -> str:
    return _response_json_preview(response, limit) or _response_text_preview(response, limit)


def _retry_after_seconds(response: httpx.Response) -> Optional[int]:
    if "retry-after" not in response.headers:
        return None
    try:
        return int(response.headers["retry-after"])
    except (ValueError, TypeError):
        return None


def _rate_limit_wait_time(response: httpx.Response, wait_seconds: int, attempt: int) -> tuple[int, bool]:
    retry_after = _retry_after_seconds(response)
    if retry_after:
        return retry_after, True
    return min(max(wait_seconds * (2 ** attempt), 30), 300), False


def _log_http_request(
    prefix: str,
    api_url: str,
    headers: dict[str, str],
    payload: JsonDict,
    attempt: int,
    max_retries: int,
) -> None:
    LOG.debug(f"[{prefix} REQUEST] Attempt {attempt + 1}/{max_retries}")
    LOG.debug(f"[{prefix} REQUEST] URL: {_redact_url(api_url)}")
    LOG.debug(f"[{prefix} REQUEST] Headers: {json.dumps(_redact_headers(headers), indent=2)}")
    LOG.debug(f"[{prefix} REQUEST] Payload: {json.dumps(payload, indent=2)}")


def _log_http_response(prefix: str, response: httpx.Response) -> None:
    LOG.debug(f"[{prefix} RESPONSE] Status: {response.status_code}")
    response_preview = _response_json_preview(response, 1000)
    if response_preview is not None:
        LOG.debug(f"[{prefix} RESPONSE] Body: {response_preview}")
    else:
        LOG.debug(f"[{prefix} RESPONSE] Body (text): {_response_text_preview(response, 1000)}")


def _warn_rate_limit(response: httpx.Response, wait_time: int, used_retry_after: bool, attempt: int, max_retries: int) -> None:
    if used_retry_after:
        LOG.warning(
            f"Rate limit detected (status {response.status_code}). "
            f"Server requested wait time: {wait_time} seconds. "
            f"Attempt {attempt + 1}/{max_retries}"
        )
    else:
        LOG.warning(
            f"Rate limit detected (status {response.status_code}). "
            f"Using exponential backoff: {wait_time} seconds. "
            f"Attempt {attempt + 1}/{max_retries}"
        )


def _emit_rate_limit_wait(wait_time: int, attempt: int, max_retries: int) -> None:
    cli = get_cli_output()
    cli.emit(
        OutputType.AGENT_RESPONSE,
        f"Rate limit hit. Waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})",
        ["API"],
        step=0,
    )


def _response_retry_delay_and_error(
    response: httpx.Response,
    wait_seconds: int,
    attempt: int,
    max_retries: int,
) -> tuple[int, IkaAPIError]:
    if _is_rate_limit_error(response):
        wait_time, used_retry_after = _rate_limit_wait_time(response, wait_seconds, attempt)
        _warn_rate_limit(response, wait_time, used_retry_after, attempt, max_retries)
        if attempt < max_retries - 1:
            _emit_rate_limit_wait(wait_time, attempt, max_retries)
            return wait_time, IkaRateLimitError(
                f"API error {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
            )
        error_text = _error_text_from_response(response, 1000)
        raise IkaRateLimitError(
            f"API error {response.status_code} after {max_retries} attempts: {error_text}",
            status_code=response.status_code,
        )

    error_text_preview = _error_text_from_response(response, 500)
    LOG.warning(f"API error {response.status_code}: {error_text_preview}")
    if attempt < max_retries - 1:
        LOG.warning(
            f"API request failed with status {response.status_code} (attempt {attempt + 1}/{max_retries}). "
            f"Retrying in {wait_seconds} seconds..."
        )
        return wait_seconds, IkaAPIError(
            f"API error {response.status_code}: {error_text_preview}",
            status_code=response.status_code,
        )

    error_text = _error_text_from_response(response, 1000)
    raise IkaAPIError(
        f"API error {response.status_code} after {max_retries} attempts: {error_text}",
        status_code=response.status_code,
    )


def _timeout_retry_delay_and_error(timeout: float, wait_seconds: int, attempt: int, max_retries: int) -> tuple[int, IkaTimeoutError]:
    timeout_msg = f"API request timed out after {timeout}s (attempt {attempt + 1}/{max_retries})"
    LOG.warning(timeout_msg)
    if attempt < max_retries - 1:
        return wait_seconds, IkaTimeoutError(timeout_msg)
    raise IkaTimeoutError(timeout_msg)


def _http_retry_delay_and_error(error: httpx.HTTPError, wait_seconds: int, attempt: int, max_retries: int) -> tuple[int, IkaHTTPError]:
    if attempt < max_retries - 1:
        LOG.warning(
            f"HTTP error during API request (attempt {attempt + 1}/{max_retries}): {error}. "
            f"Retrying in {wait_seconds} seconds..."
        )
        return wait_seconds, IkaHTTPError(f"HTTP error during API request: {str(error)}")
    raise IkaHTTPError(f"HTTP error after {max_retries} attempts: {str(error)}")


def _unexpected_retry_delay_and_error(error: Exception, wait_seconds: int, attempt: int, max_retries: int) -> tuple[int, Exception]:
    if attempt < max_retries - 1:
        LOG.warning(
            f"Unexpected error during API request (attempt {attempt + 1}/{max_retries}): {error}. "
            f"Retrying in {wait_seconds} seconds..."
        )
        return wait_seconds, error
    raise error


def get_provider(model_id: str, api_url: Optional[str] = None, use_responses_api: bool = True) -> str:
    """
    Determine the provider from model_id and optionally api_url.

    IMPORTANT: API URL takes precedence over model_id to handle cases like
    OpenRouter accessing Gemini models (e.g., "google/gemini-2.0-flash-exp").

    Args:
        model_id: The model identifier
        api_url: Optional API URL to help determine provider
        use_responses_api: OpenAI models default to the Responses API (True). Pass
            False to explicitly opt back in to Chat Completions.

    Returns:
        Provider name: "openai", "openai_responses", "anthropic", "gemini",
        "deepseek", "openrouter", or "codex".
    """
    return get_provider_for_model(model_id, api_url, use_responses_api)


LimitSpec = tuple[Any, str, int]


def agent_tools_for_payload(barebone_model: object) -> list[Any]:
    full_raw = cast(object, getattr(barebone_model, "agent_tools", None) or [])
    full = cast(list[Any], full_raw) if isinstance(full_raw, list) else []
    static_cached = getattr(barebone_model, "_agent_tools_limit_specs_cache", None)
    if static_cached and static_cached[0] is full and static_cached[1] == len(full):
        limited_specs = cast(tuple[LimitSpec, ...], static_cached[2])
    else:
        limited_specs = tuple(
            (tool, string_value(getattr(tool, "name", "")), _int_value(getattr(tool, "limit_calls", 0)))
            for tool in full
            if _int_value(getattr(tool, "limit_calls", 0)) > 0
        )
        setattr(barebone_model, "_agent_tools_limit_specs_cache", (full, len(full), limited_specs))

    if not limited_specs:
        return full

    counts_raw = cast(object, getattr(barebone_model, "_tool_call_counts", None) or {})
    counts = cast(dict[str, int], counts_raw) if isinstance(counts_raw, dict) else {}
    limited_state = tuple(
        (id(tool), name, limit, counts.get(name, 0))
        for tool, name, limit in limited_specs
    )
    cache_key = (id(full), len(full), limited_state)
    cached = getattr(barebone_model, "_agent_tools_payload_cache", None)
    if cached and cached[0] == cache_key:
        return cached[1]

    limited_by_name = {name: limit for _, name, limit in limited_specs}

    def keep(tool: Any) -> bool:
        name = string_value(getattr(tool, "name", ""))
        limit = limited_by_name.get(name, 0)
        return limit <= 0 or counts.get(name, 0) < limit

    filtered = [tool for tool in full if keep(tool)]
    setattr(barebone_model, "_agent_tools_payload_cache", (cache_key, filtered))
    return filtered


def _filtered_agent_tools_for_payload(barebone_model: Any) -> list[Any]:
    return agent_tools_for_payload(barebone_model)


def model_for_payload(barebone_model: Any) -> Any:
    payload_model = copy.copy(barebone_model)
    payload_model.agent_tools = agent_tools_for_payload(barebone_model)
    return payload_model


def _apply_tools_filter_for_payload(barebone_model: Any) -> None:
    """Deprecated compatibility shim; new code should use model_for_payload()."""
    barebone_model._agent_tools_saved = getattr(barebone_model, "agent_tools", None) or []
    barebone_model.agent_tools = agent_tools_for_payload(barebone_model)


def _restore_tools_after_payload(barebone_model: Any) -> None:
    """Deprecated compatibility shim; new code should use model_for_payload()."""
    if hasattr(barebone_model, "_agent_tools_saved"):
        barebone_model.agent_tools = barebone_model._agent_tools_saved
        delattr(barebone_model, "_agent_tools_saved")


def get_max_tokens(model_id: str) -> int:
    return get_max_tokens_for_model(model_id)


def _is_rate_limit_error(response: httpx.Response) -> bool:
    if response.status_code == 429:
        return True

    try:
        error_data = response.json()
        error_text = json.dumps(error_data)
        if "rate limit" in error_text.lower() or "rate_limit" in error_text.lower():
            return True
    except (TypeError, ValueError):
        pass

    response_text = response.text.lower() if hasattr(response, 'text') else ""
    if "rate limit" in response_text or "rate_limit" in response_text:
        return True

    return False


def _is_context_length_error(e: Exception) -> bool:
    s = str(e).lower()
    return (
        "maximum context length" in s
        or "context length" in s
        or "reduce the length of the messages" in s
        or "requested" in s and "tokens" in s and "maximum" in s
    )


def api_request_retry(
    api_url: str,
    headers: dict[str, str],
    payload: JsonDict,
    max_retries: int = 3,
    wait_seconds: int = 10,
    timeout: float = 900.0,
    client: Optional[httpx.Client] = None,
) -> httpx.Response:
    # codex backend forces streaming (rejects stream:false). Dispatch into the
    # codex client, which drains the SSE stream and returns a Response-shaped
    # shim so callers continue to call .json() / .status_code as usual.
    if api_url and "chatgpt.com/backend-api/codex" in api_url.lower():
        from .codex.chat_helpers_codex import request_codex
        return cast(
            httpx.Response,
            request_codex(
                api_url, headers, payload,
                timeout=timeout,
                max_retries=max_retries,
                wait_seconds=wait_seconds,
            ),
        )

    last_exception = None
    debug_enabled = LOG.isEnabledFor(logging.DEBUG)

    for attempt in range(max_retries):
        try:
            if debug_enabled:
                _log_http_request("HTTPX", api_url, headers, payload, attempt, max_retries)

            post = client.post if client is not None else httpx.post
            response = post(api_url, headers=headers, json=payload, timeout=timeout)

            if debug_enabled:
                _log_http_response("HTTPX", response)

            if response.status_code == 200:
                return response

            delay, last_exception = _response_retry_delay_and_error(response, wait_seconds, attempt, max_retries)
            time.sleep(delay)

        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout):
            delay, last_exception = _timeout_retry_delay_and_error(timeout, wait_seconds, attempt, max_retries)
            time.sleep(delay)

        except httpx.HTTPError as e:
            delay, last_exception = _http_retry_delay_and_error(e, wait_seconds, attempt, max_retries)
            time.sleep(delay)

        except _RETRYABLE_UNEXPECTED_EXCEPTIONS as e:
            delay, last_exception = _unexpected_retry_delay_and_error(e, wait_seconds, attempt, max_retries)
            time.sleep(delay)

    if last_exception:
        raise last_exception

    raise IkaAPIError(f"API request failed after {max_retries} attempts")


async def async_api_request_retry(
    api_url: str,
    headers: dict[str, str],
    payload: JsonDict,
    max_retries: int = 3,
    wait_seconds: int = 10,
    timeout: float = 900.0,
    client: Optional[httpx.AsyncClient] = None
) -> httpx.Response:
    # codex backend requires streaming — defer to the sync codex client via a
    # threadpool. We don't have an async SSE collector yet; running the sync
    # path off-loop avoids blocking the event loop in async callers.
    if api_url and "chatgpt.com/backend-api/codex" in api_url.lower():
        from .codex.chat_helpers_codex import request_codex
        return cast(
            httpx.Response,
            await asyncio.to_thread(
                request_codex, api_url, headers, payload, timeout, max_retries, wait_seconds
            ),
        )

    last_exception = None
    should_close_client = client is None
    debug_enabled = LOG.isEnabledFor(logging.DEBUG)

    if client is None:
        client = httpx.AsyncClient(timeout=timeout)

    try:
        for attempt in range(max_retries):
            try:
                if debug_enabled:
                    _log_http_request("HTTPX ASYNC", api_url, headers, payload, attempt, max_retries)

                response = await client.post(api_url, headers=headers, json=payload)

                if debug_enabled:
                    _log_http_response("HTTPX ASYNC", response)

                if response.status_code == 200:
                    return response

                delay, last_exception = _response_retry_delay_and_error(response, wait_seconds, attempt, max_retries)
                await asyncio.sleep(delay)

            except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout):
                delay, last_exception = _timeout_retry_delay_and_error(timeout, wait_seconds, attempt, max_retries)
                await asyncio.sleep(delay)

            except httpx.HTTPError as e:
                delay, last_exception = _http_retry_delay_and_error(e, wait_seconds, attempt, max_retries)
                await asyncio.sleep(delay)

            except _RETRYABLE_UNEXPECTED_EXCEPTIONS as e:
                delay, last_exception = _unexpected_retry_delay_and_error(e, wait_seconds, attempt, max_retries)
                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception

        raise IkaAPIError(f"API request failed after {max_retries} attempts")
    finally:
        if should_close_client:
            await client.aclose()
