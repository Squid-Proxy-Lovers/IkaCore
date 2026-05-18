import asyncio
import json
import logging
import time
from typing import Any, Optional

import httpx

from .base import TOKENMAX_MAPPING
from IkaCore.cli_output import get_cli_output, OutputType

LOG = logging.getLogger(__name__)


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
        Provider name: "openai", "openai_responses", "anthropic", "gemini", "deepseek", "openrouter"
    """
    # PRIORITY 1: Check API URL if provided (most reliable)
    if api_url:
        api_url_lower = api_url.lower()
        if "chatgpt.com/backend-api/codex" in api_url_lower:
            return "codex"
        if "openrouter.ai" in api_url_lower:
            return "openrouter"
        if "deepseek.com" in api_url_lower:
            return "deepseek"
        if api_url_lower.rstrip("/").endswith("/v1/responses"):
            return "openai_responses"
        if api_url_lower.rstrip("/").endswith("/v1/chat/completions"):
            return "openai"
        if "generativelanguage.googleapis.com" in api_url_lower:
            return "gemini"
        if "anthropic.com" in api_url_lower:
            return "anthropic"
        if "openai.com" in api_url_lower:
            return "openai_responses" if use_responses_api else "openai"

    # PRIORITY 2: Fallback to model_id detection
    model_id_lower = model_id.lower()

    # Check for OpenRouter format (has slash) BEFORE checking provider names
    # This handles cases like "google/gemini-2.0-flash-exp" on OpenRouter
    if "/" in model_id_lower:
        # OpenRouter models: "meta-llama/llama-3.1-70b-instruct", "google/gemini-pro"
        return "openrouter"

    # Unambiguous codex slugs (suffix "-codex"). Bare gpt-5.x slugs overlap
    # with the standard OpenAI Responses API and are NOT inferred as codex —
    # callers must set api_url=CODEX_API_URL explicitly for those.
    if model_id_lower.endswith("-codex"):
        return "codex"

    # Then check for specific provider names in model_id
    if "deepseek" in model_id_lower:
        return "deepseek"
    elif "gpt" in model_id_lower or "o1" in model_id_lower or "o3" in model_id_lower or "o4" in model_id_lower:
        if use_responses_api:
            return "openai_responses"
        return "openai"
    elif "claude" in model_id_lower:
        return "anthropic"
    elif "gemini" in model_id_lower:
        return "gemini"

    # Default to OpenAI-compatible
    return "openai_responses" if use_responses_api else "openai"


def _apply_tools_filter_for_payload(barebone_model: Any) -> None:
    counts = getattr(barebone_model, "_tool_call_counts", None) or {}
    full = getattr(barebone_model, "agent_tools", None) or []
    def keep(t: Any) -> bool:
        lim = getattr(t, "limit_calls", 0) or 0
        if lim <= 0:
            return True
        return counts.get(getattr(t, "name", ""), 0) < lim
    filtered = [t for t in full if keep(t)]
    barebone_model._agent_tools_saved = full
    barebone_model.agent_tools = filtered


def _restore_tools_after_payload(barebone_model: Any) -> None:
    if hasattr(barebone_model, "_agent_tools_saved"):
        barebone_model.agent_tools = barebone_model._agent_tools_saved
        delattr(barebone_model, "_agent_tools_saved")


def get_max_tokens(model_id: str) -> int:
    model_id_lower = model_id.lower()

    for key, max_tokens in TOKENMAX_MAPPING.items():
        if key.lower() in model_id_lower:
            return max_tokens

    if "gpt-4" in model_id_lower or "gpt-4o" in model_id_lower:
        return TOKENMAX_MAPPING.get("gpt-4o", 128000)
    elif "claude" in model_id_lower:
        return TOKENMAX_MAPPING.get("claude-sonnet-4", 200000)
    elif "gemini" in model_id_lower:
        return TOKENMAX_MAPPING.get("gemini-1.5-pro", 1000000)
    elif "deepseek" in model_id_lower:
        return TOKENMAX_MAPPING.get("deepseek-chat", 131072)
    elif "llama" in model_id_lower:
        return 131072  # Common Llama context window
    elif "qwen" in model_id_lower:
        return 131072

    return 128000


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
    timeout: float = 900.0
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
                safe_headers = {k: v if k.lower() != "authorization" else "Bearer ***" for k, v in headers.items()}
                LOG.debug(f"[HTTPX REQUEST] Attempt {attempt + 1}/{max_retries}")
                LOG.debug(f"[HTTPX REQUEST] URL: {api_url}")
                LOG.debug(f"[HTTPX REQUEST] Headers: {json.dumps(safe_headers, indent=2)}")
                LOG.debug(f"[HTTPX REQUEST] Payload: {json.dumps(payload, indent=2)}")

            response = httpx.post(api_url, headers=headers, json=payload, timeout=timeout)

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

        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
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
                    safe_headers = {k: v if k.lower() != "authorization" else "Bearer ***" for k, v in headers.items()}
                    LOG.debug(f"[HTTPX ASYNC REQUEST] Attempt {attempt + 1}/{max_retries}")
                    LOG.debug(f"[HTTPX ASYNC REQUEST] URL: {api_url}")
                    LOG.debug(f"[HTTPX ASYNC REQUEST] Headers: {json.dumps(safe_headers, indent=2)}")
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

            except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
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
