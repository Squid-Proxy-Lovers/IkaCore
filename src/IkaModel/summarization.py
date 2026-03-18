import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx

from .base import BareBoneModel
from .request_interface import get_provider, api_request_retry, async_api_request_retry

_LOG = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

COMPACT_BOUNDARY = "\n--- [COMPACT_BOUNDARY] ---\n"


def _load_prompt(relative_path: str, fallback: str) -> str:
    path = _PROMPTS_DIR / relative_path
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return fallback


SUMMARY_PROMPT = _load_prompt(
    "default_summary_system.txt",
    "Please summarize the following conversation history concisely, preserving key information and context.",
)
DEFAULT_SUMMARY_USER_PREFIX = _load_prompt(
    "default_summary_user_prefix.txt",
    "Please summarize the following conversation history:\n\n",
)
FORCE_ANSWER_SYSTEM = _load_prompt(
    "force_answer_system.txt",
    "You are a summarization assistant. Given a conversation and the original task, produce a single final response "
    "that directly answers the original prompt. Do not summarize loosely; answer as if the task had been completed. "
    "Output only the final answer the agent would give, with no meta-commentary.",
)
FORCE_ANSWER_USER_PREFIX = _load_prompt(
    "force_answer_user_prefix.txt",
    "Original task and conversation:\n\n",
)
WHAT_REMAINS_SYSTEM = _load_prompt(
    "what_remains_system.txt",
    "You are a triage assistant for incomplete agent runs. Output a short actionable brief in three parts: "
    "DONE, REMAINING, and NEXT. No preamble. Output only the three-part brief.",
)
WHAT_REMAINS_USER_PREFIX = _load_prompt(
    "what_remains_user_prefix.txt",
    "Below is the agent run that hit its step limit. Extract DONE / REMAINING / NEXT as specified.\n\n",
)


def _get_prompts_for_kind(prompt_kind: str) -> Tuple[str, str]:
    if prompt_kind == "force_answer":
        return FORCE_ANSWER_SYSTEM, FORCE_ANSWER_USER_PREFIX
    if prompt_kind == "what_remains":
        return WHAT_REMAINS_SYSTEM, WHAT_REMAINS_USER_PREFIX
    return SUMMARY_PROMPT, DEFAULT_SUMMARY_USER_PREFIX


def _normalize_provider_for_summary(provider: str) -> str:
    """Normalize provider for summarization (which always uses Chat Completions).

    The Responses API provider is only relevant for the main agent loop;
    summarization always uses a standard chat/completions call.
    """
    if provider == "openai_responses":
        return "openai"
    return provider


def get_summary_model(provider: str) -> tuple[Optional[str], Optional[str]]:
    provider = _normalize_provider_for_summary(provider)
    models: Dict[str, tuple[str, str]] = {
        "deepseek": ("deepseek-chat", "https://api.deepseek.com/chat/completions"),
        "openai": ("gpt-4.1-mini-2025-04-14", "https://api.openai.com/v1/chat/completions"),
        "anthropic": ("claude-sonnet-4-20250514", "https://api.anthropic.com/v1/messages"),
        "gemini": ("gemini-flash-latest", "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"),
        "openrouter": ("openai/gpt-4o-mini", "https://openrouter.ai/api/v1/chat/completions"),
    }
    result = models.get(provider.lower())
    return result if result is not None else (None, None)


def create_summary_payload(
    provider: str,
    model_name: str,
    api_key: str,
    conversation_text: str,
    system_prompt: Optional[str] = None,
    user_prompt_prefix: Optional[str] = None,
) -> tuple[dict, dict]:
    sys_prompt = system_prompt if system_prompt is not None else SUMMARY_PROMPT
    user_prefix = user_prompt_prefix if user_prompt_prefix is not None else DEFAULT_SUMMARY_USER_PREFIX
    user_content = f"{user_prefix}{conversation_text}"

    headers = {
        "Content-Type": "application/json",
    }

    if provider == "deepseek":
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
            "stream": False
        }
    elif provider == "openai" or provider == "openrouter":
        headers["Authorization"] = f"Bearer {api_key}"
        # gpt-4.1+ and gpt-5+ require max_completion_tokens instead of max_tokens
        _model_lower = model_name.lower()
        _use_mct = any(x in _model_lower for x in ["gpt-4.1", "gpt-5", "o1", "o3"])
        _token_key = "max_completion_tokens" if _use_mct else "max_tokens"
        # GPT-5 family only supports default temperature (1)
        _skip_temp = any(x in _model_lower for x in ["gpt-5", "o1", "o3"])
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            _token_key: 2000
        }
        if not _skip_temp:
            payload["temperature"] = 0.3
    elif provider == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": model_name,
            "max_tokens": 2000,
            "system": sys_prompt,
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3
        }
    elif provider == "gemini":
        headers["x-goog-api-key"] = api_key
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{sys_prompt}\n\n{user_content}"
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

    if provider in ("deepseek", "openai", "openrouter"):
        return data["choices"][0]["message"]["content"]
    elif provider == "anthropic":
        content_blocks = data.get("content", [])
        text_parts = [block["text"] for block in content_blocks if block.get("type") == "text"]
        return "".join(text_parts)
    elif provider == "gemini":
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_conversation_text(message_history: dict, *, include_first_input: bool = True) -> str:
    parts = []
    if include_first_input and message_history["first_input"]["message"]:
        parts.append(message_history["first_input"]["message"])
    if message_history["summary"]["message"]:
        parts.append(message_history["summary"]["message"])
    for msg_id in message_history["messages"]:
        parts.append(message_history["messages"][msg_id]["message"])
    full_text = "\n".join(parts)

    # Compact boundary: only summarize content after the last boundary
    if COMPACT_BOUNDARY.strip() in full_text:
        boundary_idx = full_text.rfind(COMPACT_BOUNDARY.strip())
        # Keep a short reference to the prior summary, then only the new content
        prior_context = full_text[:boundary_idx].strip()
        new_content = full_text[boundary_idx + len(COMPACT_BOUNDARY.strip()):].strip()
        if new_content:
            return f"[Prior compact summary exists — preserved above boundary]\n\n{new_content}"
        # If no new content after boundary, summarize everything (edge case)
        return full_text

    return full_text


def get_context_usage(message_history: dict, model_id: str, context_budget: Optional[int] = None) -> dict:
    """Return context usage stats: token_count, max_tokens, usage_ratio, warning_level."""
    from .request_interface import get_max_tokens
    from .chat_interface import get_total_tokens

    token_count = get_total_tokens(message_history)
    max_tokens = context_budget or get_max_tokens(model_id)
    ratio = token_count / max_tokens if max_tokens > 0 else 0.0

    if ratio >= 0.8:
        warning_level = "critical"
    elif ratio >= 0.6:
        warning_level = "warning"
    else:
        warning_level = None

    return {
        "token_count": token_count,
        "max_tokens": max_tokens,
        "usage_ratio": ratio,
        "warning_level": warning_level,
    }


def run_summarization(
    barebone_model: BareBoneModel,
    message_history: dict,
    prompt_kind: str = "default",
    write_to_history: bool = True,
    use_same_model: bool = True,
) -> str:
    if not message_history["first_input"]["message"] and not message_history["messages"]:
        return ""

    include_first_input = not (write_to_history and prompt_kind == "default")
    conversation_text = get_conversation_text(message_history, include_first_input=include_first_input)
    system_prompt, user_prompt_prefix = _get_prompts_for_kind(prompt_kind)

    provider = _normalize_provider_for_summary(
        get_provider(barebone_model.model_id, barebone_model.api_url)
    )

    # Use the same model as the agent if requested (default), otherwise use a cheaper model
    if use_same_model:
        model_name = barebone_model.model_id
        api_url = barebone_model.api_url
        # openai_responses models need Chat Completions URL for summarization
        if not api_url or "responses" in (api_url or ""):
            api_url = "https://api.openai.com/v1/chat/completions"
    else:
        model_name, api_url = get_summary_model(provider)
        if not model_name or not api_url:
            _LOG.warning(f"Could not determine low-end model for provider: {provider}")
            return ""

    payload, headers = create_summary_payload(
        provider,
        model_name,
        barebone_model.api_key,
        conversation_text,
        system_prompt=system_prompt,
        user_prompt_prefix=user_prompt_prefix,
    )

    try:
        response = api_request_retry(api_url, headers, payload, max_retries=3, wait_seconds=10)
        response.raise_for_status()
        data = response.json()
        summary = parse_summary_response(provider, response)

        summary_tokens = 0
        if provider in ("deepseek", "openai", "openrouter"):
            usage = data.get("usage", {})
            summary_tokens = usage.get("total_tokens", 0)
        elif provider == "anthropic":
            usage = data.get("usage", {})
            summary_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        elif provider == "gemini":
            usage = data.get("usageMetadata", {})
            summary_tokens = usage.get("totalTokenCount", 0)

        if write_to_history:
            message_history["summary"]["message"] = f"[SUMMARY]\n{summary}{COMPACT_BOUNDARY}"
            message_history["summary"]["tokens"] = summary_tokens
            message_history["messages"] = {}
            message_history.setdefault("compaction_count", 0)
            message_history["compaction_count"] += 1
            _LOG.info(
                "Message history compacted (compaction #%d). Kept: system prompt, first input, and summary. Cleared all other messages.",
                message_history["compaction_count"],
            )

        return summary
    except httpx.HTTPError as e:
        _LOG.error(f"Failed to summarize message history: {e}")
        return ""
    except Exception as e:
        _LOG.error(f"Unexpected error during summarization: {e}")
        return ""


def summarise_message_history(barebone_model: BareBoneModel, message_history: dict) -> str:
    return run_summarization(barebone_model, message_history, prompt_kind="default", write_to_history=True)


async def async_summarise_message_history(
    barebone_model: BareBoneModel,
    message_history: dict,
    client: Optional[httpx.AsyncClient] = None,
    use_same_model: bool = True,
    prompt_kind: str = "default",
    write_to_history: bool = True,
) -> str:
    if not message_history["first_input"]["message"] and not message_history["messages"]:
        return ""

    include_first_input = not (write_to_history and prompt_kind == "default")
    conversation_text = get_conversation_text(message_history, include_first_input=include_first_input)
    system_prompt, user_prompt_prefix = _get_prompts_for_kind(prompt_kind)

    provider = _normalize_provider_for_summary(
        get_provider(barebone_model.model_id, barebone_model.api_url)
    )

    # Use the same model as the agent if requested (default), otherwise use a cheaper model
    if use_same_model:
        model_name = barebone_model.model_id
        api_url = barebone_model.api_url
        if not api_url or "responses" in (api_url or ""):
            api_url = "https://api.openai.com/v1/chat/completions"
    else:
        model_name, api_url = get_summary_model(provider)
        if not model_name or not api_url:
            _LOG.warning(f"Could not determine low-end model for provider: {provider}")
            return ""

    payload, headers = create_summary_payload(
        provider,
        model_name,
        barebone_model.api_key,
        conversation_text,
        system_prompt=system_prompt,
        user_prompt_prefix=user_prompt_prefix,
    )

    try:
        response = await async_api_request_retry(api_url, headers, payload, max_retries=3, wait_seconds=10, client=client)
        response.raise_for_status()
        data = response.json()
        summary = parse_summary_response(provider, response)

        summary_tokens = 0
        if provider in ("deepseek", "openai", "openrouter"):
            usage = data.get("usage", {})
            summary_tokens = usage.get("total_tokens", 0)
        elif provider == "anthropic":
            usage = data.get("usage", {})
            summary_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        elif provider == "gemini":
            usage = data.get("usageMetadata", {})
            summary_tokens = usage.get("totalTokenCount", 0)

        if write_to_history:
            message_history["summary"]["message"] = f"[SUMMARY]\n{summary}{COMPACT_BOUNDARY}"
            message_history["summary"]["tokens"] = summary_tokens
            message_history["messages"] = {}
            message_history.setdefault("compaction_count", 0)
            message_history["compaction_count"] += 1
            _LOG.info(
                "Message history compacted (compaction #%d). Kept: system prompt, first input, and summary. Cleared all other messages.",
                message_history["compaction_count"],
            )

        return summary
    except httpx.HTTPError as e:
        _LOG.error(f"Failed to summarize message history: {e}")
        return ""
    except Exception as e:
        _LOG.error(f"Unexpected error during summarization: {e}")
        return ""
