import asyncio
import copy
import json
import logging
import time
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

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


def _redact_headers(headers: dict) -> dict:
    safe_headers = {}
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
    except Exception:
        return api_url


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


def agent_tools_for_payload(barebone_model: Any) -> list[Any]:
    full = getattr(barebone_model, "agent_tools", None) or []
    static_cached = getattr(barebone_model, "_agent_tools_limit_specs_cache", None)
    if static_cached and static_cached[0] is full and static_cached[1] == len(full):
        limited_specs = static_cached[2]
    else:
        limited_specs = tuple(
            (tool, getattr(tool, "name", ""), getattr(tool, "limit_calls", 0) or 0)
            for tool in full
            if (getattr(tool, "limit_calls", 0) or 0) > 0
        )
        barebone_model._agent_tools_limit_specs_cache = (full, len(full), limited_specs)

    if not limited_specs:
        return full

    counts = getattr(barebone_model, "_tool_call_counts", None) or {}
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
        name = getattr(tool, "name", "")
        limit = limited_by_name.get(name, 0)
        return limit <= 0 or counts.get(name, 0) < limit

    filtered = [tool for tool in full if keep(tool)]
    barebone_model._agent_tools_payload_cache = (cache_key, filtered)
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
    except Exception:
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
    headers: dict,
    payload: dict,
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
        return request_codex(
            api_url, headers, payload,
            timeout=timeout,
            max_retries=max_retries,
            wait_seconds=wait_seconds,
        )

    last_exception = None
    debug_enabled = LOG.isEnabledFor(logging.DEBUG)

    for attempt in range(max_retries):
        try:
            if debug_enabled:
                LOG.debug(f"[HTTPX REQUEST] Attempt {attempt + 1}/{max_retries}")
                LOG.debug(f"[HTTPX REQUEST] URL: {_redact_url(api_url)}")
                LOG.debug(f"[HTTPX REQUEST] Headers: {json.dumps(_redact_headers(headers), indent=2)}")
                LOG.debug(f"[HTTPX REQUEST] Payload: {json.dumps(payload, indent=2)}")

            post = client.post if client is not None else httpx.post
            response = post(api_url, headers=headers, json=payload, timeout=timeout)

            if debug_enabled:
                LOG.debug(f"[HTTPX RESPONSE] Status: {response.status_code}")
                try:
                    response_preview = json.dumps(response.json(), indent=2)[:1000]
                    LOG.debug(f"[HTTPX RESPONSE] Body: {response_preview}")
                except Exception:
                    LOG.debug(f"[HTTPX RESPONSE] Body (text): {response.text[:1000]}")

            if response.status_code == 200:
                return response

            is_rate_limit = _is_rate_limit_error(response)

            if is_rate_limit:
                retry_after = None
                if "retry-after" in response.headers:
                    try:
                        retry_after = int(response.headers["retry-after"])
                    except (ValueError, TypeError):
                        pass

                if retry_after:
                    wait_time = retry_after
                    LOG.warning(
                        f"Rate limit detected (status {response.status_code}). "
                        f"Server requested wait time: {wait_time} seconds. "
                        f"Attempt {attempt + 1}/{max_retries}"
                    )
                else:
                    wait_time = wait_seconds * (2 ** attempt)
                    if wait_time < 30:
                        wait_time = 30
                    elif wait_time > 300:
                        wait_time = 300
                    LOG.warning(
                        f"Rate limit detected (status {response.status_code}). "
                        f"Using exponential backoff: {wait_time} seconds. "
                        f"Attempt {attempt + 1}/{max_retries}"
                    )

                if attempt < max_retries - 1:
                    cli = get_cli_output()
                    cli.emit(
                        OutputType.AGENT_RESPONSE,
                        f"Rate limit hit. Waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})",
                        ["API"],
                        step=0
                    )
                    time.sleep(wait_time)
                    last_exception = Exception(f"API error {response.status_code}: {response.text[:500]}")
                else:
                    try:
                        error_data = response.json()
                        error_text = json.dumps(error_data, indent=2)[:1000]
                    except Exception:
                        error_text = response.text[:500] if hasattr(response, 'text') else str(response.status_code)
                    raise Exception(f"API error {response.status_code} after {max_retries} attempts: {error_text}")
            else:
                try:
                    error_data = response.json()
                    error_text_preview = json.dumps(error_data, indent=2)[:500]
                    LOG.warning(f"API error {response.status_code}: {error_text_preview}")
                except Exception:
                    error_text_preview = response.text[:500] if hasattr(response, 'text') else str(response.status_code)
                    LOG.warning(f"API error {response.status_code}: {error_text_preview}")

                if attempt < max_retries - 1:
                    LOG.warning(
                        f"API request failed with status {response.status_code} (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)
                    last_exception = Exception(f"API error {response.status_code}: {error_text_preview}")
                else:
                    try:
                        error_data = response.json()
                        error_text = json.dumps(error_data, indent=2)[:1000]
                    except Exception:
                        error_text = response.text[:500] if hasattr(response, 'text') else str(response.status_code)
                    raise Exception(f"API error {response.status_code} after {max_retries} attempts: {error_text}")

        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout):
            timeout_msg = f"API request timed out after {timeout}s (attempt {attempt + 1}/{max_retries})"
            LOG.warning(timeout_msg)
            if attempt < max_retries - 1:
                time.sleep(wait_seconds)
                last_exception = Exception(timeout_msg)
            else:
                raise Exception(timeout_msg)

        except httpx.HTTPError as e:
            if attempt < max_retries - 1:
                LOG.warning(
                    f"HTTP error during API request (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
                last_exception = e
            else:
                raise Exception(f"HTTP error after {max_retries} attempts: {str(e)}")

        except Exception as e:
            if attempt < max_retries - 1:
                LOG.warning(
                    f"Unexpected error during API request (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
                last_exception = e
            else:
                raise

    if last_exception:
        raise last_exception

    raise Exception(f"API request failed after {max_retries} attempts")


async def async_api_request_retry(
    api_url: str,
    headers: dict,
    payload: dict,
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
        return await asyncio.to_thread(
            request_codex, api_url, headers, payload, timeout, max_retries, wait_seconds
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
                    LOG.debug(f"[HTTPX ASYNC REQUEST] Attempt {attempt + 1}/{max_retries}")
                    LOG.debug(f"[HTTPX ASYNC REQUEST] URL: {_redact_url(api_url)}")
                    LOG.debug(f"[HTTPX ASYNC REQUEST] Headers: {json.dumps(_redact_headers(headers), indent=2)}")
                    LOG.debug(f"[HTTPX ASYNC REQUEST] Payload: {json.dumps(payload, indent=2)}")

                response = await client.post(api_url, headers=headers, json=payload)

                if debug_enabled:
                    LOG.debug(f"[HTTPX ASYNC RESPONSE] Status: {response.status_code}")
                    try:
                        response_preview = json.dumps(response.json(), indent=2)[:1000]
                        LOG.debug(f"[HTTPX ASYNC RESPONSE] Body: {response_preview}")
                    except Exception:
                        LOG.debug(f"[HTTPX ASYNC RESPONSE] Body (text): {response.text[:1000]}")

                if response.status_code == 200:
                    return response

                is_rate_limit = _is_rate_limit_error(response)

                if is_rate_limit:
                    retry_after = None
                    if "retry-after" in response.headers:
                        try:
                            retry_after = int(response.headers["retry-after"])
                        except (ValueError, TypeError):
                            pass

                    if retry_after:
                        wait_time = retry_after
                        LOG.warning(
                            f"Rate limit detected (status {response.status_code}). "
                            f"Server requested wait time: {wait_time} seconds. "
                            f"Attempt {attempt + 1}/{max_retries}"
                        )
                    else:
                        wait_time = wait_seconds * (2 ** attempt)
                        if wait_time < 30:
                            wait_time = 30
                        elif wait_time > 300:
                            wait_time = 300
                        LOG.warning(
                            f"Rate limit detected (status {response.status_code}). "
                            f"Using exponential backoff: {wait_time} seconds. "
                            f"Attempt {attempt + 1}/{max_retries}"
                        )

                    if attempt < max_retries - 1:
                        cli = get_cli_output()
                        cli.emit(
                            OutputType.AGENT_RESPONSE,
                            f"Rate limit hit. Waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})",
                            ["API"],
                            step=0
                        )
                        await asyncio.sleep(wait_time)
                        last_exception = Exception(f"API error {response.status_code}: {response.text[:500]}")
                    else:
                        try:
                            error_data = response.json()
                            error_text = json.dumps(error_data, indent=2)[:1000]
                        except Exception:
                            error_text = response.text[:500] if hasattr(response, 'text') else str(response.status_code)
                        raise Exception(f"API error {response.status_code} after {max_retries} attempts: {error_text}")
                else:
                    try:
                        error_data = response.json()
                        error_text_preview = json.dumps(error_data, indent=2)[:500]
                        LOG.warning(f"API error {response.status_code}: {error_text_preview}")
                    except Exception:
                        error_text_preview = response.text[:500] if hasattr(response, 'text') else str(response.status_code)
                        LOG.warning(f"API error {response.status_code}: {error_text_preview}")

                    if attempt < max_retries - 1:
                        LOG.warning(
                            f"API request failed with status {response.status_code} (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {wait_seconds} seconds..."
                        )
                        await asyncio.sleep(wait_seconds)
                        last_exception = Exception(f"API error {response.status_code}: {error_text_preview}")
                    else:
                        try:
                            error_data = response.json()
                            error_text = json.dumps(error_data, indent=2)[:1000]
                        except Exception:
                            error_text = response.text[:500] if hasattr(response, 'text') else str(response.status_code)
                        raise Exception(f"API error {response.status_code} after {max_retries} attempts: {error_text}")

            except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout):
                timeout_msg = f"API request timed out after {timeout}s (attempt {attempt + 1}/{max_retries})"
                LOG.warning(timeout_msg)
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_seconds)
                    last_exception = Exception(timeout_msg)
                else:
                    raise Exception(timeout_msg)

            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    LOG.warning(
                        f"HTTP error during API request (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_seconds} seconds..."
                    )
                    await asyncio.sleep(wait_seconds)
                    last_exception = e
                else:
                    raise Exception(f"HTTP error after {max_retries} attempts: {str(e)}")

            except Exception as e:
                if attempt < max_retries - 1:
                    LOG.warning(
                        f"Unexpected error during API request (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_seconds} seconds..."
                    )
                    await asyncio.sleep(wait_seconds)
                    last_exception = e
                else:
                    raise

        if last_exception:
            raise last_exception

        raise Exception(f"API request failed after {max_retries} attempts")
    finally:
        if should_close_client:
            await client.aclose()
