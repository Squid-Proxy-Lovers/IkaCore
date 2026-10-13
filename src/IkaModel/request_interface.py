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

from .api_errors import IkaAPIError as IkaAPIError
from .api_errors import IkaHTTPError as IkaHTTPError
from .api_errors import IkaRateLimitError as IkaRateLimitError
from .api_errors import IkaTimeoutError as IkaTimeoutError
from .model_metadata import get_max_tokens_for_model, get_provider_for_model
from .request_control import check_request_controls, effective_max_retries
from .request_control import register_request_abort_callback as register_request_abort_callback
from .request_control import request_cancelled as request_cancelled
from .request_control import request_controls as request_controls
from .request_control import set_request_abort_registrar as set_request_abort_registrar
from .request_control import set_request_cancel_checker as set_request_cancel_checker
from .request_control import set_request_max_retries as set_request_max_retries
from .request_lifecycle import async_sleep_with_controls, sleep_with_controls
from .request_transport import controlled_async_post, controlled_post, new_async_client
from .retry_policy import backoff_seconds, response_delay
from .runtime_errors import IkaRequestCancelled as IkaRequestCancelled
from .runtime_errors import IkaRequestControlError
from .runtime_errors import IkaRequestDeadlineExceeded as IkaRequestDeadlineExceeded

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
_RETRYABLE_UNEXPECTED_EXCEPTIONS = (OSError,)


def _is_codex_endpoint(api_url: Optional[str]) -> bool:
    """Recognize both the upstream Codex URL and an audit-scoped gateway."""
    return bool(api_url and "/backend-api/codex/" in api_url.lower())


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


def _response_retry_delay_and_error(response: httpx.Response, wait_seconds: float, attempt: int,
                                    max_retries: int) -> tuple[float, IkaAPIError]:
    return response_delay(response, attempt, max_retries, wait_seconds)


def _timeout_retry_delay_and_error(timeout: float, wait_seconds: float, attempt: int, max_retries: int) -> tuple[float, IkaTimeoutError]:
    error = IkaTimeoutError(f"API request timed out after {timeout}s (attempt {attempt + 1}/{max_retries})")
    if attempt >= max_retries - 1:
        raise error
    return backoff_seconds(attempt, wait_seconds), error


def _http_retry_delay_and_error(error: httpx.HTTPError, wait_seconds: float, attempt: int, max_retries: int) -> tuple[float, IkaHTTPError]:
    failure = IkaHTTPError("provider transport failure")
    if attempt >= max_retries - 1:
        raise failure from error
    return backoff_seconds(attempt, wait_seconds), failure


def _unexpected_retry_delay_and_error(error: OSError, wait_seconds: float, attempt: int, max_retries: int) -> tuple[float, OSError]:
    if attempt >= max_retries - 1:
        raise error
    return backoff_seconds(attempt, wait_seconds), error


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


def model_for_payload(barebone_model: Any) -> Any:
    payload_model = copy.copy(barebone_model)
    payload_model.agent_tools = agent_tools_for_payload(barebone_model)
    return payload_model


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
        or ("requested" in s and "tokens" in s and "maximum" in s)
    )


def api_request_retry(
    api_url: str,
    headers: dict[str, str],
    payload: JsonDict,
    max_retries: int = 3,
    wait_seconds: float = 1.0,
    timeout: float = 900.0,
    client: Optional[httpx.Client] = None,
) -> httpx.Response:
    check_request_controls()
    max_retries = effective_max_retries(max_retries)
    # codex backend forces streaming (rejects stream:false). Dispatch into the
    # codex client, which drains the SSE stream and returns a Response-shaped
    # shim so callers continue to call .json() / .status_code as usual.
    if _is_codex_endpoint(api_url):
        from .codex.chat_helpers_codex import request_codex
        return cast(
            httpx.Response,
            request_codex(
                api_url, headers, payload,
                timeout=timeout,
                max_retries=max_retries,
                wait_seconds=wait_seconds,
                client=client,
            ),
        )

    last_exception = None
    debug_enabled = LOG.isEnabledFor(logging.DEBUG)

    for attempt in range(max_retries):
        try:
            if debug_enabled:
                _log_http_request("HTTPX", api_url, headers, payload, attempt, max_retries)

            check_request_controls()
            response = controlled_post(api_url, headers, payload, timeout, client)

            if debug_enabled:
                _log_http_response("HTTPX", response)

            if response.status_code == 200:
                return response

            delay, last_exception = _response_retry_delay_and_error(response, wait_seconds, attempt, max_retries)
            sleep_with_controls(delay, time.sleep)

        except IkaRequestControlError:
            raise

        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout):
            delay, last_exception = _timeout_retry_delay_and_error(timeout, wait_seconds, attempt, max_retries)
            check_request_controls()
            sleep_with_controls(delay, time.sleep)

        except httpx.HTTPError as e:
            delay, last_exception = _http_retry_delay_and_error(e, wait_seconds, attempt, max_retries)
            check_request_controls()
            sleep_with_controls(delay, time.sleep)

        except _RETRYABLE_UNEXPECTED_EXCEPTIONS as e:
            delay, last_exception = _unexpected_retry_delay_and_error(e, wait_seconds, attempt, max_retries)
            check_request_controls()
            sleep_with_controls(delay, time.sleep)

    if last_exception:
        raise last_exception

    raise IkaAPIError(f"API request failed after {max_retries} attempts")


async def async_api_request_retry(
    api_url: str,
    headers: dict[str, str],
    payload: JsonDict,
    max_retries: int = 3,
    wait_seconds: float = 1.0,
    timeout: float = 900.0,
    client: Optional[httpx.AsyncClient] = None
) -> httpx.Response:
    check_request_controls()
    max_retries = effective_max_retries(max_retries)
    if _is_codex_endpoint(api_url):
        from .codex.async_request import async_request_codex
        return cast(httpx.Response, await async_request_codex(
            api_url, headers, payload, timeout, max_retries, wait_seconds, client,
        ))

    should_close_client = client is None
    if client is None:
        client = new_async_client(timeout)
    try:
        return await _async_retry_loop(api_url, headers, payload, timeout, client, max_retries, wait_seconds)
    finally:
        if should_close_client:
            await client.aclose()


async def _async_retry_loop(
    api_url: str, headers: dict[str, str], payload: JsonDict, timeout: float,
    client: httpx.AsyncClient, max_retries: int, wait_seconds: float,
) -> httpx.Response:
    last_exception = None
    debug_enabled = LOG.isEnabledFor(logging.DEBUG)
    for attempt in range(max_retries):
        try:
            if debug_enabled:
                _log_http_request("HTTPX ASYNC", api_url, headers, payload, attempt, max_retries)

            check_request_controls()
            response = await controlled_async_post(api_url, headers, payload, timeout, client)

            if debug_enabled:
                _log_http_response("HTTPX ASYNC", response)

            if response.status_code == 200:
                return response

            delay, last_exception = _response_retry_delay_and_error(response, wait_seconds, attempt, max_retries)
            await async_sleep_with_controls(delay, asyncio.sleep)

        except IkaRequestControlError:
            raise

        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout):
            delay, last_exception = _timeout_retry_delay_and_error(timeout, wait_seconds, attempt, max_retries)
            check_request_controls()
            await async_sleep_with_controls(delay, asyncio.sleep)

        except httpx.HTTPError as e:
            delay, last_exception = _http_retry_delay_and_error(e, wait_seconds, attempt, max_retries)
            check_request_controls()
            await async_sleep_with_controls(delay, asyncio.sleep)

        except _RETRYABLE_UNEXPECTED_EXCEPTIONS as e:
            delay, last_exception = _unexpected_retry_delay_and_error(e, wait_seconds, attempt, max_retries)
            check_request_controls()
            await async_sleep_with_controls(delay, asyncio.sleep)

    if last_exception:
        raise last_exception

    raise IkaAPIError(f"API request failed after {max_retries} attempts")
