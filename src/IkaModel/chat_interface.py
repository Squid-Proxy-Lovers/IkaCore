import asyncio
import json
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Dict, Optional, List, Callable, Union

import httpx

from .base import BareBoneModel, TOKENMAX_MAPPING

# Import CLI output system
from IkaCore.cli_output import get_cli_output, OutputType
from pathlib import Path

# Load SUMMARY_PROMPT directly
_SUMMARY_PROMPT_PATH = Path(__file__).parent / "summary_prompt"
try:
    with open(_SUMMARY_PROMPT_PATH, "r", encoding="utf-8") as f:
        SUMMARY_PROMPT = f.read()
except FileNotFoundError:
    SUMMARY_PROMPT = "Please summarize the following conversation history concisely, preserving key information and context."
from .deepseek import deepseek_fill_payload
from .openai import openai_fill_payload
from .claude import anthropic_fill_payload
from .google import gemini_fill_payload

LOG = logging.getLogger(__name__)


def get_summary_model(provider: str) -> tuple[str, str]:
    models = {
        "deepseek": ("deepseek-chat", "https://api.deepseek.com/chat/completions"),
        "openai": ("gpt-4.1-mini-2025-04-14", "https://api.openai.com/v1/chat/completions"),
        "anthropic": ("claude-sonnet-4-20250514", "https://api.anthropic.com/v1/messages"),
        "gemini": ("gemini-flash-latest", "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"),
    }
    return models.get(provider.lower(), (None, None))


def get_provider(model_id: str) -> str:
    model_id_lower = model_id.lower()
    if "deepseek" in model_id_lower:
        return "deepseek"
    elif "gpt" in model_id_lower or "o1" in model_id_lower or "o3" in model_id_lower:
        return "openai"
    elif "claude" in model_id_lower:
        return "anthropic"
    elif "gemini" in model_id_lower:
        return "gemini"
    return "openai"


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


def create_summary_payload(provider: str, model_name: str, api_key: str, conversation_text: str) -> tuple[dict, dict]:
    headers = {
        "Content-Type": "application/json",
    }
    
    if provider == "deepseek":
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": f"Please summarize the following conversation history:\n\n{conversation_text}"}
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
            "stream": False
        }
    elif provider == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": f"Please summarize the following conversation history:\n\n{conversation_text}"}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
    elif provider == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": model_name,
            "max_tokens": 2000,
            "system": SUMMARY_PROMPT,
            "messages": [
                {"role": "user", "content": f"Please summarize the following conversation history:\n\n{conversation_text}"}
            ],
            "temperature": 0.3
        }
    elif provider == "gemini":
        headers["x-goog-api-key"] = api_key
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{SUMMARY_PROMPT}\n\nPlease summarize the following conversation history:\n\n{conversation_text}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2000
            }
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return payload, headers


def parse_summary_response(provider: str, response: httpx.Response) -> str:
    data = response.json()
    
    if provider == "deepseek" or provider == "openai":
        return data["choices"][0]["message"]["content"]
    elif provider == "anthropic":
        content_blocks = data.get("content", [])
        text_parts = [block["text"] for block in content_blocks if block.get("type") == "text"]
        return "".join(text_parts)
    elif provider == "gemini":
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def init_message_history() -> dict:
    return {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def get_total_tokens(message_history: dict) -> int:
    total = 0
    for key in ("system", "first_input", "summary"):
        d = message_history.get(key) or {}
        t = d.get("tokens", 0) or 0
        total += t if t > 0 else _estimate_tokens(d.get("message", ""))
    for msg in (message_history.get("messages") or {}).values():
        t = msg.get("tokens", 0) or 0
        total += t if t > 0 else _estimate_tokens(msg.get("message", ""))
    return total


def get_conversation_text(message_history: dict) -> str:
    parts = []
    if message_history["first_input"]["message"]:
        parts.append(message_history["first_input"]["message"])
    if message_history["summary"]["message"]:
        parts.append(message_history["summary"]["message"])
    for msg_id in message_history["messages"]:
        parts.append(message_history["messages"][msg_id]["message"])
    return "\n".join(parts)


def summarise_message_history(barebone_model: BareBoneModel, message_history: dict) -> str:
    if not message_history["first_input"]["message"] and not message_history["messages"]:
        return ""
    
    conversation_text = get_conversation_text(message_history)
    
    provider = get_provider(barebone_model.model_id)
    model_name, api_url = get_summary_model(provider)
    
    if not model_name or not api_url:
        LOG.warning(f"Could not determine low-end model for provider: {provider}")
        return ""
    
    payload, headers = create_summary_payload(provider, model_name, barebone_model.api_key, conversation_text)
    
    try:
        response = api_request_retry(api_url, headers, payload, max_retries=3, wait_seconds=10)
        response.raise_for_status()
        data = response.json()
        summary = parse_summary_response(provider, response)
        
        summary_tokens = 0
        if provider == "deepseek" or provider == "openai":
            usage = data.get("usage", {})
            summary_tokens = usage.get("total_tokens", 0)
        elif provider == "anthropic":
            usage = data.get("usage", {})
            summary_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        elif provider == "gemini":
            usage = data.get("usageMetadata", {})
            summary_tokens = usage.get("totalTokenCount", 0)
        
        message_history["summary"]["message"] = f"[SUMMARY]\n{summary}"
        message_history["summary"]["tokens"] = summary_tokens
        message_history["messages"] = {}
        
        LOG.info("Message history summarized. Kept: system prompt, first input, and summary. Cleared all other messages.")
        return summary
    except httpx.HTTPError as e:
        LOG.error(f"Failed to summarize message history: {e}")
        return ""
    except Exception as e:
        LOG.error(f"Unexpected error during summarization: {e}")
        return ""

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
    
    return 128000


def _is_rate_limit_error(response: httpx.Response) -> bool:
    if response.status_code == 429: # anthropic & openai use 429 for rate limit errors
        return True
    
    try:
        error_data = response.json()
        error_text = json.dumps(error_data)
        if "rate limit" in error_text.lower() or "rate_limit" in error_text.lower():
            return True
    except:
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
                except:
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
                    except:
                        error_text = response.text[:500] if hasattr(response, 'text') else str(response.status_code)
                    raise Exception(f"API error {response.status_code} after {max_retries} attempts: {error_text}")
            else:
                # Log the error for debugging
                try:
                    error_data = response.json()
                    error_text_preview = json.dumps(error_data, indent=2)[:500]
                    LOG.warning(f"API error {response.status_code}: {error_text_preview}")
                except:
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
                    except:
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
                    except:
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
                        except:
                            error_text = response.text[:500] if hasattr(response, 'text') else str(response.status_code)
                        raise Exception(f"API error {response.status_code} after {max_retries} attempts: {error_text}")
                else:
                    try:
                        error_data = response.json()
                        error_text_preview = json.dumps(error_data, indent=2)[:500]
                        LOG.warning(f"API error {response.status_code}: {error_text_preview}")
                    except:
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
                        except:
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


def _api_request_with_context_fallback(
    build_payload_fn: Callable[[], tuple[str, dict, dict]],
    barebone_model: BareBoneModel,
    message_history: dict,
    timeout: float = 900.0
) -> httpx.Response:
    api_url, headers, payload = build_payload_fn()
    try:
        return api_request_retry(api_url, headers, payload, timeout=timeout)
    except Exception as e:
        if not _is_context_length_error(e):
            raise
        LOG.warning("Context length exceeded. Forcing summarization and retrying.")
        get_cli_output().emit(OutputType.AGENT_RESPONSE, "Context limit exceeded. Summarized history and retrying.", ["API"], step=0)
        summarise_message_history(barebone_model, message_history)
        api_url, headers, payload = build_payload_fn()
        return api_request_retry(api_url, headers, payload, timeout=timeout)


async def _api_request_with_context_fallback_async(
    build_payload_fn: Callable[[], tuple[str, dict, dict]],
    barebone_model: BareBoneModel,
    message_history: dict,
    timeout: float = 900.0,
    client: Optional[httpx.AsyncClient] = None
) -> httpx.Response:
    api_url, headers, payload = build_payload_fn()
    try:
        return await async_api_request_retry(api_url, headers, payload, timeout=timeout, client=client)
    except Exception as e:
        if not _is_context_length_error(e):
            raise
        LOG.warning("Context length exceeded. Forcing summarization and retrying.")
        get_cli_output().emit(OutputType.AGENT_RESPONSE, "Context limit exceeded. Summarized history and retrying.", ["API"], step=0)
        await async_summarise_message_history(barebone_model, message_history, client=client)
        api_url, headers, payload = build_payload_fn()
        return await async_api_request_retry(api_url, headers, payload, timeout=timeout, client=client)


def extract_usage(provider: str, data: dict) -> Dict[str, Any]:
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "input_cached_tokens": 0}
    raw_usage = data.get("usage", {}) or {}

    if provider in ["deepseek", "openai"]:
        usage["input_tokens"] = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
        usage["output_tokens"] = raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0))
        usage["total_tokens"] = raw_usage.get("total_tokens", usage["input_tokens"] + usage["output_tokens"])
    elif provider == "anthropic":
        usage["input_tokens"] = raw_usage.get("input_tokens", 0)
        usage["output_tokens"] = raw_usage.get("output_tokens", 0)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    elif provider == "gemini":
        meta = data.get("usageMetadata", {}) or raw_usage
        total = meta.get("totalTokenCount", 0)
        usage["total_tokens"] = total
        usage["output_tokens"] = total
    else:
        usage["total_tokens"] = raw_usage.get("total_tokens", 0)

    return usage


def validate_tool_args(tool_name: str, tool_args: dict, max_size: int = 10000, max_keys: int = 50) -> dict:
    if tool_args is None:
        raise ValueError(f"Tool '{tool_name}' requires arguments, but got None")

    if not isinstance(tool_args, dict):
        raise ValueError(f"Tool '{tool_name}' arguments must be a JSON object, got {type(tool_args).__name__}")

    # if len(tool_args) > max_keys:
    #     raise ValueError(f"Tool '{tool_name}' received too many arguments ({len(tool_args)} > {max_keys})")

    try:
        serialized = json.dumps(tool_args)
    except TypeError as e:
        raise ValueError(f"Tool '{tool_name}' arguments must be JSON-serializable: {e}")

    # if len(serialized) > max_size:
    #     raise ValueError(f"Tool '{tool_name}' arguments payload too large ({len(serialized)} bytes > {max_size})")

    coerced_args: dict = {}
    for key, value in tool_args.items():
        if isinstance(value, str):
            v = value.strip()
            if v.lower() in ("true", "false"):
                coerced_args[key] = v.lower() == "true"
                continue
            try:
                coerced_args[key] = int(v)
                continue
            except ValueError:
                pass
            try:
                coerced_args[key] = float(v)
                continue
            except ValueError:
                pass
            coerced_args[key] = value
        else:
            coerced_args[key] = value

    if tool_name == "agent_end" and "input" in coerced_args and (coerced_args["input"] is None or coerced_args["input"] == ""):
        raise ValueError(f"Tool '{tool_name}' requires a non-empty 'input' argument")

    return coerced_args


def execute_tool(tool_name: str, tool_args: dict, tool_executors: Dict[str, Callable], timeout: float = 900.0, agent_hierarchy: Optional[List[str]] = None, step: int = 0) -> str:
    cli = get_cli_output()
    hierarchy = list(agent_hierarchy or []) + [tool_name]

    if tool_name not in tool_executors:
        error_msg = f"Tool '{tool_name}' not found in tool executors"
        LOG.error(error_msg)
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        return json.dumps({"error": error_msg})

    try:
        validated_args = validate_tool_args(tool_name, tool_args)
    except ValueError as e:
        error_msg = f"Validation error for tool '{tool_name}': {str(e)}"
        LOG.warning(error_msg)
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        return json.dumps({"error": error_msg})

    executor = tool_executors[tool_name]

    # Emit tool call output
    cli.tool_call(tool_name, tool_args, hierarchy, step)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor_pool:
            future = executor_pool.submit(executor, validated_args)
            result = future.result(timeout=timeout)

            # Emit tool result output
            result_str = result if isinstance(result, str) else json.dumps(result)
            cli.tool_result(tool_name, result_str, hierarchy, step)

        if isinstance(result, str):
            return result
        return json.dumps(result)
    except FutureTimeoutError:
        timeout_msg = f"Tool '{tool_name}' execution timed out after {timeout}s"
        cli.tool_result(tool_name, timeout_msg, hierarchy, step, is_timeout=True)
        LOG.warning(timeout_msg)
        return json.dumps({"error": timeout_msg})
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        LOG.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


def format_openai_results(tool_calls: List[dict], tool_results: List[str]) -> List[dict]:
    tool_messages = []
    for i, tool_call in enumerate(tool_calls):
        tool_call_id = tool_call.get("id", f"call_{i}")
        tool_messages.append({
            "role": "tool",
            "content": tool_results[i] if i < len(tool_results) else json.dumps({"error": "No result"}),
            "tool_call_id": tool_call_id
        })
    return tool_messages


def format_anthropic_results(tool_calls: List[dict], tool_results: List[str]) -> List[dict]:
    tool_messages = []
    for i, tool_call in enumerate(tool_calls):
        tool_call_id = tool_call.get("id", f"call_{i}")
        tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name", "")
        tool_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_results[i] if i < len(tool_results) else json.dumps({"error": "No result"})
                }
            ]
        })
    return tool_messages


def format_gemini_results(tool_calls: List[dict], tool_results: List[str]) -> List[dict]:
    function_responses = []
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name", "")
        try:
            result_data = json.loads(tool_results[i]) if i < len(tool_results) else {"error": "No result"}
        except Exception:
            result_data = {"result": tool_results[i]} if i < len(tool_results) else {"error": "No result"}
        if not isinstance(result_data, dict):
            result_data = {"result": result_data}
        function_responses.append({
            "functionResponse": {
                "name": tool_name,
                "response": result_data
            }
        })
    return function_responses


def execute_tool_calls(
    tool_calls: List[dict],
    tool_executors: Dict[str, Callable],
    provider: str,
    timeout: float = 900.0,
    tool_metadata: Optional[Dict[str, dict]] = None,
    agent_hierarchy: Optional[List[str]] = None,
    step: int = 0,
    tool_call_counts: Optional[Dict[str, int]] = None
) -> tuple[List[dict], List[str], Dict[str, int], List[dict]]:
    """Execute tool calls either in parallel or sequentially based on tool metadata.
    
    Args:
        tool_calls: List of tool call dictionaries from the API
        tool_executors: Dictionary mapping tool names to executor functions
        provider: API provider (openai, anthropic, etc.)
        timeout: Timeout for tool execution
        tool_metadata: Optional dictionary mapping tool names to metadata (including 'parallel' flag)
        agent_hierarchy: Optional list of agent names representing the call hierarchy
        step: Current step number
        tool_call_counts: Optional dictionary tracking tool call counts (will be updated in place)
    
    Returns:
        Tuple of (formatted_messages, tool_results, updated_tool_call_counts, executed_tool_call_list)
    """
    if not tool_calls or not tool_executors:
        return [], [], tool_call_counts or {}, []
    
    tool_metadata = tool_metadata or {}
    tool_call_counts = tool_call_counts or {}
    tool_call_order = tool_calls.copy()
    tool_call_id_to_result: Dict[str, str] = {}
    
    parallel_calls = []
    sequential_calls = []
    seen_tool_signatures = set()
    
    for idx, tool_call in enumerate(tool_calls):
        fn = tool_call.get("function", {})
        tool_name = fn.get("name") or tool_call.get("name", "")
        args_raw = fn.get("arguments") or "{}"
        tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"
        
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception as e:
            LOG.warning(f"Failed to parse tool arguments for {tool_name}: {e}")
            args = {}
        
        tool_signature = (tool_name, json.dumps(args, sort_keys=True))
        if tool_signature in seen_tool_signatures:
            LOG.warning(f"Duplicate tool call detected: {tool_name} with same arguments. Skipping duplicate.")
            error_msg = json.dumps({"error": f"Duplicate tool call for '{tool_name}' detected. Only executing once."})
            tool_call_id_to_result[tool_call_id] = error_msg
            continue
        seen_tool_signatures.add(tool_signature)
        
        metadata = tool_metadata.get(tool_name, {})
        limit_calls = metadata.get("limit_calls", 0)
        current_count = tool_call_counts.get(tool_name, 0)
        
        if limit_calls > 0 and current_count >= limit_calls:
            LOG.warning(f"Tool '{tool_name}' has reached its call limit ({limit_calls}). Skipping this call.")
            error_msg = json.dumps({"error": f"Tool '{tool_name}' call limit ({limit_calls}) reached. Skipping execution."})
            tool_call_id_to_result[tool_call_id] = error_msg
            continue
        
        is_parallel = metadata.get("parallel", True)  # Default to parallel
        
        if is_parallel:
            parallel_calls.append((tool_name, args, tool_call_id))
        else:
            sequential_calls.append((tool_name, args, tool_call_id))
    
    if parallel_calls:
        print("we are executing parallel tool calls: ", parallel_calls)
        with ThreadPoolExecutor(max_workers=len(parallel_calls)) as executor:
            futures = {}
            for tool_name, args, tool_call_id in parallel_calls:
                future = executor.submit(execute_tool, tool_name, args, tool_executors, timeout, agent_hierarchy, step)
                futures[future] = (tool_name, tool_call_id)

            for future in futures:
                tool_name, tool_call_id = futures[future]
                try:
                    result = future.result()
                    tool_call_id_to_result[tool_call_id] = result
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                except Exception as e:
                    error_msg = f"Parallel tool execution error: {str(e)}"
                    LOG.error(error_msg)
                    tool_call_id_to_result[tool_call_id] = json.dumps({"error": error_msg})
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
    print(sequential_calls)
    for tool_name, args, tool_call_id in sequential_calls:
        print("we are executing sequential tool calls: ", tool_name, "with args: ", args, "and tool call id: ", tool_call_id)
        result = execute_tool(tool_name, args, tool_executors, timeout, agent_hierarchy, step)
        tool_call_id_to_result[tool_call_id] = result
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
    
    tool_results = []
    for idx, tool_call in enumerate(tool_call_order):
        fn = tool_call.get("function", {})
        tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"
        tool_results.append(tool_call_id_to_result.get(tool_call_id, json.dumps({"error": "No result"})))
    
    if provider == "deepseek" or provider == "openai":
        formatted_messages = format_openai_results(tool_call_order, tool_results)
    elif provider == "anthropic":
        formatted_messages = format_anthropic_results(tool_call_order, tool_results)
    elif provider == "gemini":
        formatted_messages = format_gemini_results(tool_call_order, tool_results)
    else:
        formatted_messages = []
    
    return formatted_messages, tool_results, tool_call_counts, tool_call_order


async def async_execute_tool(
    tool_name: str,
    tool_args: dict,
    tool_executors: Dict[str, Callable],
    timeout: float = 900.0,
    agent_hierarchy: Optional[List[str]] = None,
    step: int = 0
) -> str:
    cli = get_cli_output()
    hierarchy = list(agent_hierarchy or []) + [tool_name]

    if tool_name not in tool_executors:
        error_msg = f"Tool '{tool_name}' not found in tool executors"
        LOG.error(error_msg)
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        return json.dumps({"error": error_msg})

    try:
        validated_args = validate_tool_args(tool_name, tool_args)
    except ValueError as e:
        error_msg = f"Validation error for tool '{tool_name}': {str(e)}"
        LOG.warning(error_msg)
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        return json.dumps({"error": error_msg})

    executor = tool_executors[tool_name]
    cli.tool_call(tool_name, tool_args, hierarchy, step)

    try:
        loop = asyncio.get_event_loop()
        # Check if executor is a coroutine function
        if asyncio.iscoroutinefunction(executor):
            result = await asyncio.wait_for(executor(validated_args), timeout=timeout)
        else:
            # Run sync function in thread pool
            result = await asyncio.wait_for(
                loop.run_in_executor(None, executor, validated_args),
                timeout=timeout
            )

        result_str = result if isinstance(result, str) else json.dumps(result)
        cli.tool_result(tool_name, result_str, hierarchy, step)

        if isinstance(result, str):
            return result
        return json.dumps(result)
    except asyncio.TimeoutError:
        timeout_msg = f"Tool '{tool_name}' execution timed out after {timeout}s"
        cli.tool_result(tool_name, timeout_msg, hierarchy, step, is_timeout=True)
        LOG.warning(timeout_msg)
        return json.dumps({"error": timeout_msg})
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        LOG.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


async def async_execute_tool_calls(
    tool_calls: List[dict],
    tool_executors: Dict[str, Callable],
    provider: str,
    timeout: float = 900.0,
    tool_metadata: Optional[Dict[str, dict]] = None,
    agent_hierarchy: Optional[List[str]] = None,
    step: int = 0,
    tool_call_counts: Optional[Dict[str, int]] = None
) -> tuple[List[dict], List[str], Dict[str, int], List[dict]]:
    if not tool_calls or not tool_executors:
        return [], [], tool_call_counts or {}, []

    tool_metadata = tool_metadata or {}
    tool_call_counts = tool_call_counts or {}
    tool_call_order = tool_calls.copy()
    tool_call_id_to_result: Dict[str, str] = {}

    parallel_calls = []
    sequential_calls = []
    seen_tool_signatures = set()

    for idx, tool_call in enumerate(tool_calls):
        fn = tool_call.get("function", {})
        tool_name = fn.get("name") or tool_call.get("name", "")
        args_raw = fn.get("arguments") or "{}"
        tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"

        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception as e:
            LOG.warning(f"Failed to parse tool arguments for {tool_name}: {e}")
            args = {}

        tool_signature = (tool_name, json.dumps(args, sort_keys=True))
        if tool_signature in seen_tool_signatures:
            LOG.warning(f"Duplicate tool call detected: {tool_name} with same arguments. Skipping duplicate.")
            error_msg = json.dumps({"error": f"Duplicate tool call for '{tool_name}' detected. Only executing once."})
            tool_call_id_to_result[tool_call_id] = error_msg
            continue
        seen_tool_signatures.add(tool_signature)

        metadata = tool_metadata.get(tool_name, {})
        limit_calls = metadata.get("limit_calls", 0)
        current_count = tool_call_counts.get(tool_name, 0)
        
        if limit_calls > 0 and current_count >= limit_calls:
            LOG.warning(f"Tool '{tool_name}' has reached its call limit ({limit_calls}). Skipping this call.")
            error_msg = json.dumps({"error": f"Tool '{tool_name}' call limit ({limit_calls}) reached. Skipping execution."})
            tool_call_id_to_result[tool_call_id] = error_msg
            continue

        is_parallel = metadata.get("parallel", True)

        if is_parallel:
            parallel_calls.append((tool_name, args, tool_call_id))
        else:
            sequential_calls.append((tool_name, args, tool_call_id))

    # Execute parallel tools concurrently using asyncio.gather
    if parallel_calls:
        tasks = []
        tool_call_ids_for_parallel = []
        for tool_name, args, tool_call_id in parallel_calls:
            task = async_execute_tool(tool_name, args, tool_executors, timeout, agent_hierarchy, step)
            tasks.append(task)
            tool_call_ids_for_parallel.append(tool_call_id)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            tool_call_id = tool_call_ids_for_parallel[i]
            tool_name = parallel_calls[i][0] if i < len(parallel_calls) else ""
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            if isinstance(result, Exception):
                error_msg = f"Parallel tool execution error: {str(result)}"
                LOG.error(error_msg)
                tool_call_id_to_result[tool_call_id] = json.dumps({"error": error_msg})
            else:
                tool_call_id_to_result[tool_call_id] = result if isinstance(result, str) else json.dumps(result)

    # Execute sequential tools one by one
    for tool_name, args, tool_call_id in sequential_calls:
        result = await async_execute_tool(tool_name, args, tool_executors, timeout, agent_hierarchy, step)
        tool_call_id_to_result[tool_call_id] = result if isinstance(result, str) else json.dumps(result)
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
    
    # Build tool_results in the same order as tool_call_order
    tool_results = []
    for idx, tool_call in enumerate(tool_call_order):
        fn = tool_call.get("function", {})
        tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"
        tool_results.append(tool_call_id_to_result.get(tool_call_id, json.dumps({"error": "No result"})))

    # Format results based on provider
    if provider == "deepseek" or provider == "openai":
        formatted_messages = format_openai_results(tool_call_order, tool_results)
    elif provider == "anthropic":
        formatted_messages = format_anthropic_results(tool_call_order, tool_results)
    elif provider == "gemini":
        formatted_messages = format_gemini_results(tool_call_order, tool_results)
    else:
        formatted_messages = []

    return formatted_messages, tool_results, tool_call_counts, tool_call_order


def chat(
    barebone_model: BareBoneModel,
    messages: list[dict],
    message_history: Optional[dict] = None,
    tool_executors: Optional[Dict[str, Callable]] = None,
    logger: Optional[Any] = None,
    timeout: float = 900.0,
    max_tool_rounds: int = 5,
    max_tool_calls: Optional[int] = None,
    current_stage_index: Optional[int] = None,
    total_stages: int = 0
) -> Dict[str, Any]:
    if not barebone_model:
        raise ValueError("barebone_model is required")
    if not messages:
        raise ValueError("messages is required and cannot be empty")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    
    if not hasattr(barebone_model, 'model_id') or not barebone_model.model_id:
        raise ValueError("barebone_model.model_id is required")
    if not hasattr(barebone_model, 'api_key') or not barebone_model.api_key:
        raise ValueError("barebone_model.api_key is required")
    if not hasattr(barebone_model, 'api_url') or not barebone_model.api_url:
        raise ValueError("barebone_model.api_url is required")
    
    if tool_executors is not None and not isinstance(tool_executors, dict):
        raise ValueError("tool_executors must be a dictionary if provided")
    
    message_history = message_history or init_message_history()
    tool_executors = tool_executors or {}
    if logger:
        logger.log_input(messages)
    
    token_count = get_total_tokens(message_history)
    max_tokens = get_max_tokens(barebone_model.model_id)
    
    if token_count > max_tokens * 0.8:
        #LOG.info(f"Token count ({token_count}) approaching limit ({max_tokens}). Summarizing history...")
        summarise_message_history(barebone_model, message_history)
    
    if not message_history["first_input"]["message"] and messages:
        message_history["first_input"]["message"] = messages[0].get("content", str(messages[0]))
        message_history["first_input"]["tokens"] = 0
    
    provider = get_provider(barebone_model.model_id)
    headers = {}
    api_url = barebone_model.api_url
    
    if barebone_model.model_id.lower().startswith("deepseek") or "deepseek" in barebone_model.model_id.lower():
        def _build():
            _apply_tools_filter_for_payload(barebone_model)
            p = deepseek_fill_payload(barebone_model, messages, message_history)
            _restore_tools_after_payload(barebone_model)
            return (barebone_model.api_url, {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}, p)
        response = _api_request_with_context_fallback(_build, barebone_model, message_history, timeout)

    elif barebone_model.model_id.lower().startswith("gpt") or "openai" in barebone_model.model_id.lower():
        def _build():
            _apply_tools_filter_for_payload(barebone_model)
            p = openai_fill_payload(barebone_model, messages, message_history)
            _restore_tools_after_payload(barebone_model)
            return (barebone_model.api_url, {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}, p)
        response = _api_request_with_context_fallback(_build, barebone_model, message_history, timeout)

    elif "claude" in barebone_model.model_id.lower() or "anthropic" in barebone_model.model_id.lower():
        def _build():
            _apply_tools_filter_for_payload(barebone_model)
            p = anthropic_fill_payload(barebone_model, messages, message_history)
            _restore_tools_after_payload(barebone_model)
            if "max_tokens" not in p or not p["max_tokens"]:
                p["max_tokens"] = 4096
            if "haiku" in barebone_model.model_id.lower() and p["max_tokens"] > 4096:
                p["max_tokens"] = 4096
            return (barebone_model.api_url, {"x-api-key": barebone_model.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, p)
        response = _api_request_with_context_fallback(_build, barebone_model, message_history, timeout)

    elif "gemini" in barebone_model.model_id.lower():
        def _build():
            _apply_tools_filter_for_payload(barebone_model)
            p = gemini_fill_payload(barebone_model, messages, message_history)
            _restore_tools_after_payload(barebone_model)
            return (f"{barebone_model.api_url}?key={barebone_model.api_key}", {"Content-Type": "application/json"}, p)
        response = _api_request_with_context_fallback(_build, barebone_model, message_history, timeout)
    else:
        raise ValueError(f"Model {barebone_model.model_id} not supported")
    
    response.raise_for_status()
    data = response.json()
    
    content = ""
    reasoning_content: Optional[str] = None
    tokens = 0
    tool_calls: List[dict] = []
    usage_info: Dict[str, Any] = {}

    if "deepseek" in barebone_model.model_id.lower() or "gpt" in barebone_model.model_id.lower() or "openai" in barebone_model.model_id.lower():
        message_obj = data["choices"][0]["message"]
        content = message_obj.get("content") or ""
        tool_calls = message_obj.get("tool_calls", []) or []
        tokens = data.get("usage", {}).get("total_tokens", 0)
        reasoning_content = message_obj.get("reasoning_content") if "deepseek" in barebone_model.model_id.lower() else None
    elif "claude" in barebone_model.model_id.lower():
        content_blocks = data.get("content", [])
        content = "".join([block["text"] for block in content_blocks if block.get("type") == "text"])
        for block in content_blocks:
            if block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })
    elif "gemini" in barebone_model.model_id.lower():
        candidate = (data.get("candidates") or [{}])[0]
        candidate_content = candidate.get("content") or {}
        parts = candidate_content.get("parts", [])
        for part in parts:
            if "text" in part:
                content += part["text"]
            elif "functionCall" in part:
                func_call = part["functionCall"]
                tool_calls.append({
                    "name": func_call.get("name"),
                    "function": {
                        "name": func_call.get("name"),
                        "arguments": json.dumps(func_call.get("args", {}))
                    },
                    "thoughtSignature": part.get("thoughtSignature"),
                })
    else:
        content = ""
        tokens = 0

    usage_info = extract_usage(provider, data)
    if usage_info:
        tokens = usage_info.get("total_tokens", tokens)
    
    cost_info = logger.compute_cost(barebone_model.model_id, usage_info) if logger else None

    all_executed_tool_call_list: List[dict] = []
    content_before_tools = content
    tool_call_counts = getattr(barebone_model, '_tool_call_counts', None) or {}
    rounds = 0
    total_tool_calls_in_cycle = 0
    recent_tool_calls = []
    hijacked = False

    if not hasattr(barebone_model, '_current_step'):
        barebone_model._current_step = 0
    
    agent_hierarchy = getattr(barebone_model, 'agent_hierarchy', None)

    while tool_calls and tool_executors and rounds < max_tool_rounds:
        repeated_tool_names: List[str] = []
        for tool_call in tool_calls:
            fn = tool_call.get("function", {})
            tool_name = fn.get("name") or tool_call.get("name", "")
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            
            signature = (tool_name, json.dumps(args, sort_keys=True))
            recent_count = recent_tool_calls.count(signature)
            
            if recent_count >= 3:
                repeated_tool_names.append(tool_name)
                LOG.warning(f"Tool '{tool_name}' has been called {recent_count} times recently with identical arguments. Consider calling agent_end or changing your approach.")
        
        tool_metadata = {}
        if hasattr(barebone_model, 'agent_tools'):
            for agent_tool in barebone_model.agent_tools:
                tool_metadata[agent_tool.name] = {
                    "parallel": agent_tool.parallel if hasattr(agent_tool, 'parallel') else True,
                    "limit_calls": agent_tool.limit_calls if hasattr(agent_tool, 'limit_calls') else 0
                }
        step = getattr(barebone_model, '_current_step', 0)
        tool_messages, tool_results, updated_counts, executed_tool_call_list = execute_tool_calls(tool_calls, tool_executors, provider, timeout, tool_metadata, agent_hierarchy, step, tool_call_counts)
        if hasattr(barebone_model, '_tool_call_counts'):
            barebone_model._tool_call_counts.update(updated_counts)
        all_executed_tool_call_list.extend(executed_tool_call_list)
        
        num_tools_executed = len(executed_tool_call_list)
        total_tool_calls_in_cycle += num_tools_executed
        if hasattr(barebone_model, '_current_step'):
            barebone_model._current_step += num_tools_executed
        
        for tool_call in executed_tool_call_list:
            fn = tool_call.get("function", {})
            tool_name = fn.get("name") or tool_call.get("name", "")
            args = fn.get("arguments", "{}")
            signature = (tool_name, args)
            recent_tool_calls.append(signature)
            if len(recent_tool_calls) > 10:
                recent_tool_calls.pop(0)
        
        if logger:
            logger.log_tool_results(executed_tool_call_list, tool_results)
        
        if max_tool_calls and barebone_model._current_step >= max_tool_calls:
            is_last_stage = (total_stages > 0 and current_stage_index is not None 
                           and current_stage_index == total_stages - 1)
            has_stages = total_stages > 0
            
            if has_stages and not is_last_stage:
                control_tool = "stage_end"
                hijack_message = f"CRITICAL: Maximum tool call limit ({max_tool_calls}) reached at step {barebone_model._current_step}. You MUST call stage_end to advance to the next stage."
            else:
                control_tool = "agent_end"
                hijack_message = f"CRITICAL: Maximum tool call limit ({max_tool_calls}) reached at step {barebone_model._current_step}. You MUST call agent_end with your final answer."
            
            msg_id = str(uuid.uuid4())
            message_history["messages"][msg_id] = {
                "message": hijack_message,
                "tokens": 0,
                "type": "system_directive",
            }
            
            LOG.warning(f"Max tool calls ({max_tool_calls}) reached. Injected {control_tool} directive.")
            hijacked = True
            tool_calls = []
            break

        repeated_warning_msg = ""
        if repeated_tool_names:
            seen = list(dict.fromkeys(repeated_tool_names))
            repeated_warning_msg = (
                "System note: You have already called the following tool(s) multiple times with the same arguments: "
                + ", ".join(seen) + ". Do not repeat these calls. Proceed to the next step (e.g. use submit_discovery or other tools, then agent_end when done)."
            )

        if provider == "gemini":
            assistant_msg = {"role": "model", "parts": [{"text": content}]}
            for tool_call in executed_tool_call_list:
                fc_part = {
                    "functionCall": {
                        "name": tool_call.get("name") or tool_call.get("function", {}).get("name", ""),
                        "args": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    }
                }
                if tool_call.get("thoughtSignature"):
                    fc_part["thoughtSignature"] = tool_call["thoughtSignature"]
                assistant_msg["parts"].append(fc_part)
            messages.append(assistant_msg)
            function_responses = format_gemini_results(executed_tool_call_list, tool_results)
            tool_response_msg = {"role": "user", "parts": function_responses}
            messages.append(tool_response_msg)
            if repeated_warning_msg:
                messages.append({"role": "user", "parts": [{"text": repeated_warning_msg}]})
            msg_id = str(uuid.uuid4())
            message_history["messages"][msg_id] = {
                "message": json.dumps(assistant_msg),
                "tokens": tokens,
                "type": "assistant_with_tools",
            }
            msg_id = str(uuid.uuid4())
            message_history["messages"][msg_id] = {
                "message": json.dumps(tool_response_msg),
                "tokens": 0,
                "type": "tool",
            }
        elif provider == "deepseek" or provider == "openai":
            assistant_msg = {"role": "assistant", "content": content}
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
        elif provider == "anthropic":
            assistant_msg = {"role": "assistant", "content": []}
            if content:
                assistant_msg["content"].append({"type": "text", "text": content})
            for tool_call in executed_tool_call_list:
                assistant_msg["content"].append({
                    "type": "tool_use",
                    "id": tool_call.get("id"),
                    "name": tool_call.get("name"),
                    "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                })
            messages.append(assistant_msg)
            messages.extend(tool_messages)
            if repeated_warning_msg:
                messages.append({"role": "user", "content": [{"type": "text", "text": repeated_warning_msg}]})
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

        rounds += 1
        if rounds >= max_tool_rounds:
            tool_calls = []
            break

        if barebone_model.model_id.lower().startswith("deepseek") or "deepseek" in barebone_model.model_id.lower():
            def _build_follow():
                _apply_tools_filter_for_payload(barebone_model)
                p = deepseek_fill_payload(barebone_model, messages, message_history)
                _restore_tools_after_payload(barebone_model)
                return (barebone_model.api_url, {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}, p)
            response = _api_request_with_context_fallback(_build_follow, barebone_model, message_history, timeout)
        elif barebone_model.model_id.lower().startswith("gpt") or "openai" in barebone_model.model_id.lower():
            def _build_follow():
                _apply_tools_filter_for_payload(barebone_model)
                p = openai_fill_payload(barebone_model, messages, message_history)
                _restore_tools_after_payload(barebone_model)
                return (barebone_model.api_url, {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}, p)
            response = _api_request_with_context_fallback(_build_follow, barebone_model, message_history, timeout)
        elif "claude" in barebone_model.model_id.lower() or "anthropic" in barebone_model.model_id.lower():
            def _build_follow():
                _apply_tools_filter_for_payload(barebone_model)
                p = anthropic_fill_payload(barebone_model, messages, message_history)
                _restore_tools_after_payload(barebone_model)
                if "max_tokens" not in p or not p["max_tokens"]:
                    p["max_tokens"] = 4096
                return (barebone_model.api_url, {"x-api-key": barebone_model.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, p)
            response = _api_request_with_context_fallback(_build_follow, barebone_model, message_history, timeout)
        elif "gemini" in barebone_model.model_id.lower():
            def _build_follow():
                _apply_tools_filter_for_payload(barebone_model)
                p = gemini_fill_payload(barebone_model, messages, message_history)
                _restore_tools_after_payload(barebone_model)
                return (f"{barebone_model.api_url}?key={barebone_model.api_key}", {"Content-Type": "application/json"}, p)
            response = _api_request_with_context_fallback(_build_follow, barebone_model, message_history, timeout)
        else:
            break

        response.raise_for_status()
        data = response.json()
        content = ""
        reasoning_content = None
        tokens = 0
        tool_calls = []
        usage_info = extract_usage(provider, data)
        if usage_info:
            tokens = usage_info.get("total_tokens", 0)
        if "deepseek" in barebone_model.model_id.lower() or "gpt" in barebone_model.model_id.lower() or "openai" in barebone_model.model_id.lower():
            message_obj = data["choices"][0]["message"]
            content = message_obj.get("content") or ""
            tool_calls = message_obj.get("tool_calls", []) or []
            reasoning_content = message_obj.get("reasoning_content") if "deepseek" in barebone_model.model_id.lower() else None
        elif "claude" in barebone_model.model_id.lower():
            content_blocks = data.get("content", [])
            content = "".join([block["text"] for block in content_blocks if block.get("type") == "text"])
            for block in content_blocks:
                if block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {}))
                        }
                    })
        elif "gemini" in barebone_model.model_id.lower():
            candidate = (data.get("candidates") or [{}])[0]
            candidate_content = candidate.get("content") or {}
            parts = candidate_content.get("parts", [])
            for part in parts:
                if "text" in part:
                    content += part["text"]
                elif "functionCall" in part:
                    func_call = part["functionCall"]
                    tool_calls.append({
                        "name": func_call.get("name"),
                        "function": {
                            "name": func_call.get("name"),
                            "arguments": json.dumps(func_call.get("args", {}))
                        },
                        "thoughtSignature": part.get("thoughtSignature"),
                    })

    if tool_calls and tool_executors:
        tool_metadata = {}
        if hasattr(barebone_model, 'agent_tools'):
            for agent_tool in barebone_model.agent_tools:
                tool_metadata[agent_tool.name] = {
                    "parallel": agent_tool.parallel if hasattr(agent_tool, 'parallel') else True,
                    "limit_calls": agent_tool.limit_calls if hasattr(agent_tool, 'limit_calls') else 0
                }
        agent_hierarchy = getattr(barebone_model, 'agent_hierarchy', None)
        step = getattr(barebone_model, '_current_step', 0)
        tool_messages, tool_results, updated_counts, executed_tool_call_list = execute_tool_calls(tool_calls, tool_executors, provider, timeout, tool_metadata, agent_hierarchy, step, tool_call_counts)
        if hasattr(barebone_model, '_tool_call_counts'):
            barebone_model._tool_call_counts.update(updated_counts)
        all_executed_tool_call_list.extend(executed_tool_call_list)
        if logger:
            logger.log_tool_results(executed_tool_call_list, tool_results)
        if provider == "gemini":
            assistant_msg = {"role": "model", "parts": [{"text": content}]}
            for tool_call in executed_tool_call_list:
                fc_part = {
                    "functionCall": {
                        "name": tool_call.get("name") or tool_call.get("function", {}).get("name", ""),
                        "args": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    }
                }
                if tool_call.get("thoughtSignature"):
                    fc_part["thoughtSignature"] = tool_call["thoughtSignature"]
                assistant_msg["parts"].append(fc_part)
            messages.append(assistant_msg)
            messages.append({"role": "user", "parts": format_gemini_results(executed_tool_call_list, tool_results)})
        elif provider == "deepseek" or provider == "openai":
            assistant_msg = {"role": "assistant", "content": content}
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            assistant_msg["tool_calls"] = executed_tool_call_list
            messages.append(assistant_msg)
            messages.extend(tool_messages)
        elif provider == "anthropic":
            assistant_msg = {"role": "assistant", "content": []}
            if content:
                assistant_msg["content"].append({"type": "text", "text": content})
            for tool_call in executed_tool_call_list:
                assistant_msg["content"].append({
                    "type": "tool_use",
                    "id": tool_call.get("id"),
                    "name": tool_call.get("name"),
                    "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                })
            messages.append(assistant_msg)
            messages.extend(tool_messages)
        tool_calls = []

    cost_info = logger.compute_cost(barebone_model.model_id, usage_info) if logger else None
    if logger:
        logger.log_output(content, usage_info, cost_info, message_history)

    msg_id = str(uuid.uuid4())
    history_entry = {"message": content, "tokens": tokens}
    if reasoning_content:
        history_entry["reasoning_content"] = reasoning_content
    message_history["messages"][msg_id] = history_entry

    print("content: ", content)
    print("reasoning_content: ", reasoning_content)
    print("tool_calls: ", tool_calls)
    print("executed_tool_calls: ", all_executed_tool_call_list)
    print("content_before_tools: ", content_before_tools)
    print("message_history: ", message_history)
    print("usage: ", usage_info)
    print("cost: ", cost_info)
    print("hijacked: ", hijacked)
    
    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
        "executed_tool_calls": all_executed_tool_call_list,
        "content_before_tools": content_before_tools,
        "message_history": message_history,
        "usage": usage_info,
        "cost": cost_info,
        "hijacked": hijacked,
    }


async def async_summarise_message_history(
    barebone_model: BareBoneModel,
    message_history: dict,
    client: Optional[httpx.AsyncClient] = None
) -> str:
    if not message_history["first_input"]["message"] and not message_history["messages"]:
        return ""

    conversation_text = get_conversation_text(message_history)

    provider = get_provider(barebone_model.model_id)
    model_name, api_url = get_summary_model(provider)

    if not model_name or not api_url:
        LOG.warning(f"Could not determine low-end model for provider: {provider}")
        return ""

    payload, headers = create_summary_payload(provider, model_name, barebone_model.api_key, conversation_text)

    try:
        response = await async_api_request_retry(api_url, headers, payload, max_retries=3, wait_seconds=10, client=client)
        response.raise_for_status()
        data = response.json()
        summary = parse_summary_response(provider, response)

        summary_tokens = 0
        if provider == "deepseek" or provider == "openai":
            usage = data.get("usage", {})
            summary_tokens = usage.get("total_tokens", 0)
        elif provider == "anthropic":
            usage = data.get("usage", {})
            summary_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        elif provider == "gemini":
            usage = data.get("usageMetadata", {})
            summary_tokens = usage.get("totalTokenCount", 0)

        message_history["summary"]["message"] = f"[SUMMARY]\n{summary}"
        message_history["summary"]["tokens"] = summary_tokens
        message_history["messages"] = {}

        LOG.info("Message history summarized. Kept: system prompt, first input, and summary. Cleared all other messages.")
        return summary
    except httpx.HTTPError as e:
        LOG.error(f"Failed to summarize message history: {e}")
        return ""
    except Exception as e:
        LOG.error(f"Unexpected error during summarization: {e}")
        return ""


async def async_chat(
    barebone_model: BareBoneModel,
    messages: list[dict],
    message_history: Optional[dict] = None,
    tool_executors: Optional[Dict[str, Callable]] = None,
    logger: Optional[Any] = None,
    timeout: float = 900.0,
    client: Optional[httpx.AsyncClient] = None,
    max_tool_rounds: int = 5,
    max_tool_calls: Optional[int] = None,
    current_stage_index: Optional[int] = None,
    total_stages: int = 0
) -> Dict[str, Any]:
    if not barebone_model:
        raise ValueError("barebone_model is required")
    if not messages:
        raise ValueError("messages is required and cannot be empty")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    if not hasattr(barebone_model, 'model_id') or not barebone_model.model_id:
        raise ValueError("barebone_model.model_id is required")
    if not hasattr(barebone_model, 'api_key') or not barebone_model.api_key:
        raise ValueError("barebone_model.api_key is required")
    if not hasattr(barebone_model, 'api_url') or not barebone_model.api_url:
        raise ValueError("barebone_model.api_url is required")

    if tool_executors is not None and not isinstance(tool_executors, dict):
        raise ValueError("tool_executors must be a dictionary if provided")

    message_history = message_history or init_message_history()
    tool_executors = tool_executors or {}
    if logger:
        logger.log_input(messages)

    token_count = get_total_tokens(message_history)
    max_tokens = get_max_tokens(barebone_model.model_id)

    if token_count > max_tokens * 0.8:
        LOG.info(f"Token count ({token_count}) approaching limit ({max_tokens}). Summarizing history...")
        await async_summarise_message_history(barebone_model, message_history, client)

    if not message_history["first_input"]["message"] and messages:
        message_history["first_input"]["message"] = messages[0].get("content", str(messages[0]))
        message_history["first_input"]["tokens"] = 0

    provider = get_provider(barebone_model.model_id)
    headers = {}
    api_url = barebone_model.api_url

    # Create shared client if not provided
    should_close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)

    try:
        if barebone_model.model_id.lower().startswith("deepseek") or "deepseek" in barebone_model.model_id.lower():
            def _build():
                _apply_tools_filter_for_payload(barebone_model)
                p = deepseek_fill_payload(barebone_model, messages, message_history)
                _restore_tools_after_payload(barebone_model)
                return (barebone_model.api_url, {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}, p)
            response = await _api_request_with_context_fallback_async(_build, barebone_model, message_history, timeout, client)

        elif barebone_model.model_id.lower().startswith("gpt") or "openai" in barebone_model.model_id.lower():
            def _build():
                _apply_tools_filter_for_payload(barebone_model)
                p = openai_fill_payload(barebone_model, messages, message_history)
                _restore_tools_after_payload(barebone_model)
                return (barebone_model.api_url, {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}, p)
            response = await _api_request_with_context_fallback_async(_build, barebone_model, message_history, timeout, client)

        elif "claude" in barebone_model.model_id.lower() or "anthropic" in barebone_model.model_id.lower():
            def _build():
                _apply_tools_filter_for_payload(barebone_model)
                p = anthropic_fill_payload(barebone_model, messages, message_history)
                _restore_tools_after_payload(barebone_model)
                if "max_tokens" not in p or not p["max_tokens"]:
                    p["max_tokens"] = 4096
                if "haiku" in barebone_model.model_id.lower() and p["max_tokens"] > 4096:
                    p["max_tokens"] = 4096
                return (barebone_model.api_url, {"x-api-key": barebone_model.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, p)
            response = await _api_request_with_context_fallback_async(_build, barebone_model, message_history, timeout, client)

        elif "gemini" in barebone_model.model_id.lower():
            def _build():
                _apply_tools_filter_for_payload(barebone_model)
                p = gemini_fill_payload(barebone_model, messages, message_history)
                _restore_tools_after_payload(barebone_model)
                return (f"{barebone_model.api_url}?key={barebone_model.api_key}", {"Content-Type": "application/json"}, p)
            response = await _api_request_with_context_fallback_async(_build, barebone_model, message_history, timeout, client)
        else:
            raise ValueError(f"Model {barebone_model.model_id} not supported")

        response.raise_for_status()
        data = response.json()

        content = ""
        reasoning_content: Optional[str] = None
        tokens = 0
        tool_calls: List[dict] = []
        usage_info: Dict[str, Any] = {}

        if "deepseek" in barebone_model.model_id.lower() or "gpt" in barebone_model.model_id.lower() or "openai" in barebone_model.model_id.lower():
            message_obj = data["choices"][0]["message"]
            content = message_obj.get("content") or ""
            tool_calls = message_obj.get("tool_calls", []) or []
            tokens = data.get("usage", {}).get("total_tokens", 0)
            reasoning_content = message_obj.get("reasoning_content") if "deepseek" in barebone_model.model_id.lower() else None
        elif "claude" in barebone_model.model_id.lower():
            content_blocks = data.get("content", [])
            content = "".join([block["text"] for block in content_blocks if block.get("type") == "text"])
            for block in content_blocks:
                if block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {}))
                        }
                    })
        elif "gemini" in barebone_model.model_id.lower():
            candidate = (data.get("candidates") or [{}])[0]
            candidate_content = candidate.get("content") or {}
            parts = candidate_content.get("parts", [])
            for part in parts:
                if "text" in part:
                    content += part["text"]
                elif "functionCall" in part:
                    func_call = part["functionCall"]
                    tool_calls.append({
                        "name": func_call.get("name"),
                        "function": {
                            "name": func_call.get("name"),
                            "arguments": json.dumps(func_call.get("args", {}))
                        },
                        "thoughtSignature": part.get("thoughtSignature"),
                    })
        else:
            content = ""
            tokens = 0

        usage_info = extract_usage(provider, data)
        if usage_info:
            tokens = usage_info.get("total_tokens", tokens)

        cost_info = logger.compute_cost(barebone_model.model_id, usage_info) if logger else None

        all_executed_tool_call_list: List[dict] = []
        content_before_tools = content
        tool_call_counts = getattr(barebone_model, '_tool_call_counts', None) or {}
        rounds = 0
        total_tool_calls_in_cycle = 0
        recent_tool_calls = []
        hijacked = False

        if not hasattr(barebone_model, '_current_step'):
            barebone_model._current_step = 0
        
        agent_hierarchy = getattr(barebone_model, 'agent_hierarchy', None)

        while tool_calls and tool_executors and rounds < max_tool_rounds:
            repeated_tool_names_async: List[str] = []
            for tool_call in tool_calls:
                fn = tool_call.get("function", {})
                tool_name = fn.get("name") or tool_call.get("name", "")
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {}
                
                signature = (tool_name, json.dumps(args, sort_keys=True))
                recent_count = recent_tool_calls.count(signature)
                
                if recent_count >= 3:
                    repeated_tool_names_async.append(tool_name)
                    LOG.warning(f"Tool '{tool_name}' has been called {recent_count} times recently with identical arguments. Consider calling agent_end or changing your approach.")
            
            repeated_warning_msg_async = ""
            if repeated_tool_names_async:
                seen_async = list(dict.fromkeys(repeated_tool_names_async))
                repeated_warning_msg_async = (
                    "System note: You have already called the following tool(s) multiple times with the same arguments: "
                    + ", ".join(seen_async) + ". Do not repeat these calls. Proceed to the next step (e.g. use submit_discovery or other tools, then agent_end when done)."
                )

            tool_metadata = {}
            if hasattr(barebone_model, 'agent_tools'):
                for agent_tool in barebone_model.agent_tools:
                    tool_metadata[agent_tool.name] = {
                        "parallel": agent_tool.parallel if hasattr(agent_tool, 'parallel') else True,
                        "limit_calls": agent_tool.limit_calls if hasattr(agent_tool, 'limit_calls') else 0
                    }
            step = getattr(barebone_model, '_current_step', 0)

            tool_messages, tool_results, updated_counts, executed_tool_call_list = await async_execute_tool_calls(tool_calls, tool_executors, provider, timeout, tool_metadata, agent_hierarchy, step, tool_call_counts)
            if hasattr(barebone_model, '_tool_call_counts'):
                barebone_model._tool_call_counts.update(updated_counts)
            all_executed_tool_call_list.extend(executed_tool_call_list)
            
            num_tools_executed = len(executed_tool_call_list)
            total_tool_calls_in_cycle += num_tools_executed
            if hasattr(barebone_model, '_current_step'):
                barebone_model._current_step += num_tools_executed
            
            for tool_call in executed_tool_call_list:
                fn = tool_call.get("function", {})
                tool_name = fn.get("name") or tool_call.get("name", "")
                args = fn.get("arguments", "{}")
                signature = (tool_name, args)
                recent_tool_calls.append(signature)
                if len(recent_tool_calls) > 10:
                    recent_tool_calls.pop(0)
            
            if logger:
                logger.log_tool_results(executed_tool_call_list, tool_results)
            
            if max_tool_calls and barebone_model._current_step >= max_tool_calls:
                is_last_stage = (total_stages > 0 and current_stage_index is not None 
                               and current_stage_index == total_stages - 1)
                has_stages = total_stages > 0
                
                if has_stages and not is_last_stage:
                    control_tool = "stage_end"
                    hijack_message = f"CRITICAL: Maximum tool call limit ({max_tool_calls}) reached at step {barebone_model._current_step}. You MUST call stage_end to advance to the next stage."
                else:
                    control_tool = "agent_end"
                    hijack_message = f"CRITICAL: Maximum tool call limit ({max_tool_calls}) reached at step {barebone_model._current_step}. You MUST call agent_end with your final answer."
                
                msg_id = str(uuid.uuid4())
                message_history["messages"][msg_id] = {
                    "message": hijack_message,
                    "tokens": 0,
                    "type": "system_directive",
                }
                
                LOG.warning(f"Max tool calls ({max_tool_calls}) reached. Injected {control_tool} directive.")
                hijacked = True
                tool_calls = []
                break

            if provider == "gemini":
                assistant_msg = {"role": "model", "parts": [{"text": content}]}
                for tool_call in executed_tool_call_list:
                    fc_part = {
                        "functionCall": {
                            "name": tool_call.get("name") or tool_call.get("function", {}).get("name", ""),
                            "args": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                        }
                    }
                    if tool_call.get("thoughtSignature"):
                        fc_part["thoughtSignature"] = tool_call["thoughtSignature"]
                    assistant_msg["parts"].append(fc_part)
                messages.append(assistant_msg)
                messages.append({"role": "user", "parts": format_gemini_results(executed_tool_call_list, tool_results)})
                if repeated_warning_msg_async:
                    messages.append({"role": "user", "parts": [{"text": repeated_warning_msg_async}]})
            elif provider == "deepseek" or provider == "openai":
                assistant_msg = {"role": "assistant", "content": content}
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                if executed_tool_call_list:
                    assistant_msg["tool_calls"] = executed_tool_call_list
                messages.append(assistant_msg)
                messages.extend(tool_messages)
                if repeated_warning_msg_async:
                    messages.append({"role": "user", "content": repeated_warning_msg_async})
            elif provider == "anthropic":
                assistant_msg = {"role": "assistant", "content": []}
                if content:
                    assistant_msg["content"].append({"type": "text", "text": content})
                for tool_call in executed_tool_call_list:
                    assistant_msg["content"].append({
                        "type": "tool_use",
                        "id": tool_call.get("id"),
                        "name": tool_call.get("name"),
                        "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    })
                messages.append(assistant_msg)
                messages.extend(tool_messages)
                if repeated_warning_msg_async:
                    messages.append({"role": "user", "content": [{"type": "text", "text": repeated_warning_msg_async}]})

            rounds += 1
            if rounds >= max_tool_rounds:
                tool_calls = []
                break

            if barebone_model.model_id.lower().startswith("deepseek") or "deepseek" in barebone_model.model_id.lower():
                def _build_follow():
                    _apply_tools_filter_for_payload(barebone_model)
                    p = deepseek_fill_payload(barebone_model, messages, message_history)
                    _restore_tools_after_payload(barebone_model)
                    return (barebone_model.api_url, {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}, p)
                response = await _api_request_with_context_fallback_async(_build_follow, barebone_model, message_history, timeout, client)
            elif barebone_model.model_id.lower().startswith("gpt") or "openai" in barebone_model.model_id.lower():
                def _build_follow():
                    _apply_tools_filter_for_payload(barebone_model)
                    p = openai_fill_payload(barebone_model, messages, message_history)
                    _restore_tools_after_payload(barebone_model)
                    return (barebone_model.api_url, {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}, p)
                response = await _api_request_with_context_fallback_async(_build_follow, barebone_model, message_history, timeout, client)
            elif "claude" in barebone_model.model_id.lower() or "anthropic" in barebone_model.model_id.lower():
                def _build_follow():
                    _apply_tools_filter_for_payload(barebone_model)
                    p = anthropic_fill_payload(barebone_model, messages, message_history)
                    _restore_tools_after_payload(barebone_model)
                    if "max_tokens" not in p or not p["max_tokens"]:
                        p["max_tokens"] = 4096
                    return (barebone_model.api_url, {"x-api-key": barebone_model.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, p)
                response = await _api_request_with_context_fallback_async(_build_follow, barebone_model, message_history, timeout, client)
            elif "gemini" in barebone_model.model_id.lower():
                def _build_follow():
                    _apply_tools_filter_for_payload(barebone_model)
                    p = gemini_fill_payload(barebone_model, messages, message_history)
                    _restore_tools_after_payload(barebone_model)
                    return (f"{barebone_model.api_url}?key={barebone_model.api_key}", {"Content-Type": "application/json"}, p)
                response = await _api_request_with_context_fallback_async(_build_follow, barebone_model, message_history, timeout, client)
            else:
                break

            response.raise_for_status()
            data = response.json()
            content = ""
            reasoning_content = None
            tokens = 0
            tool_calls = []
            usage_info = extract_usage(provider, data)
            if usage_info:
                tokens = usage_info.get("total_tokens", 0)
            if "deepseek" in barebone_model.model_id.lower() or "gpt" in barebone_model.model_id.lower() or "openai" in barebone_model.model_id.lower():
                message_obj = data["choices"][0]["message"]
                content = message_obj.get("content") or ""
                tool_calls = message_obj.get("tool_calls", []) or []
                reasoning_content = message_obj.get("reasoning_content") if "deepseek" in barebone_model.model_id.lower() else None
            elif "claude" in barebone_model.model_id.lower():
                content_blocks = data.get("content", [])
                content = "".join([block["text"] for block in content_blocks if block.get("type") == "text"])
                for block in content_blocks:
                    if block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {}))
                            }
                        })
            elif "gemini" in barebone_model.model_id.lower():
                candidate = (data.get("candidates") or [{}])[0]
                candidate_content = candidate.get("content") or {}
                parts = candidate_content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        content += part["text"]
                    elif "functionCall" in part:
                        func_call = part["functionCall"]
                        tool_calls.append({
                            "name": func_call.get("name"),
                            "function": {
                                "name": func_call.get("name"),
                                "arguments": json.dumps(func_call.get("args", {}))
                            },
                            "thoughtSignature": part.get("thoughtSignature"),
                        })

        if tool_calls and tool_executors:
            tool_metadata = {}
            if hasattr(barebone_model, 'agent_tools'):
                for agent_tool in barebone_model.agent_tools:
                    tool_metadata[agent_tool.name] = {
                        "parallel": agent_tool.parallel if hasattr(agent_tool, 'parallel') else True,
                        "limit_calls": agent_tool.limit_calls if hasattr(agent_tool, 'limit_calls') else 0
                    }
            agent_hierarchy = getattr(barebone_model, 'agent_hierarchy', None)
            step = getattr(barebone_model, '_current_step', 0)
            tool_messages, tool_results, updated_counts, executed_tool_call_list = await async_execute_tool_calls(tool_calls, tool_executors, provider, timeout, tool_metadata, agent_hierarchy, step, tool_call_counts)
            if hasattr(barebone_model, '_tool_call_counts'):
                barebone_model._tool_call_counts.update(updated_counts)
            all_executed_tool_call_list.extend(executed_tool_call_list)
            if logger:
                logger.log_tool_results(executed_tool_call_list, tool_results)
            if provider == "gemini":
                assistant_msg = {"role": "model", "parts": [{"text": content}]}
                for tool_call in executed_tool_call_list:
                    fc_part = {
                        "functionCall": {
                            "name": tool_call.get("name") or tool_call.get("function", {}).get("name", ""),
                            "args": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                        }
                    }
                    if tool_call.get("thoughtSignature"):
                        fc_part["thoughtSignature"] = tool_call["thoughtSignature"]
                    assistant_msg["parts"].append(fc_part)
                messages.append(assistant_msg)
                messages.append({"role": "user", "parts": format_gemini_results(executed_tool_call_list, tool_results)})
            elif provider == "deepseek" or provider == "openai":
                assistant_msg = {"role": "assistant", "content": content}
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                assistant_msg["tool_calls"] = executed_tool_call_list
                messages.append(assistant_msg)
                messages.extend(tool_messages)
            elif provider == "anthropic":
                assistant_msg = {"role": "assistant", "content": []}
                if content:
                    assistant_msg["content"].append({"type": "text", "text": content})
                for tool_call in executed_tool_call_list:
                    assistant_msg["content"].append({
                        "type": "tool_use",
                        "id": tool_call.get("id"),
                        "name": tool_call.get("name"),
                        "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    })
                messages.append(assistant_msg)
                messages.extend(tool_messages)
            tool_calls = []

        cost_info = logger.compute_cost(barebone_model.model_id, usage_info) if logger else None
        if logger:
            logger.log_output(content, usage_info, cost_info, message_history)

        msg_id = str(uuid.uuid4())
        history_entry = {"message": content, "tokens": tokens}
        if reasoning_content:
            history_entry["reasoning_content"] = reasoning_content
        message_history["messages"][msg_id] = history_entry
        
        return {
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "executed_tool_calls": all_executed_tool_call_list,
            "content_before_tools": content_before_tools,
            "message_history": message_history,
            "usage": usage_info,
            "cost": cost_info,
            "hijacked": hijacked,
        }

    finally:
        if should_close_client:
            await client.aclose()
