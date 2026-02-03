import logging
from pathlib import Path
from typing import Dict, Optional

import httpx

from .base import BareBoneModel
from .request_interface import get_provider, api_request_retry, async_api_request_retry

_LOG = logging.getLogger(__name__)

_SUMMARY_PROMPT_PATH = Path(__file__).parent / "summary_prompt"
try:
    with open(_SUMMARY_PROMPT_PATH, "r", encoding="utf-8") as f:
        SUMMARY_PROMPT = f.read()
except FileNotFoundError:
    SUMMARY_PROMPT = "Please summarize the following conversation history concisely, preserving key information and context."


def get_summary_model(provider: str) -> tuple[Optional[str], Optional[str]]:
    models: Dict[str, tuple[str, str]] = {
        "deepseek": ("deepseek-chat", "https://api.deepseek.com/chat/completions"),
        "openai": ("gpt-4.1-mini-2025-04-14", "https://api.openai.com/v1/chat/completions"),
        "anthropic": ("claude-sonnet-4-20250514", "https://api.anthropic.com/v1/messages"),
        "gemini": ("gemini-flash-latest", "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"),
    }
    result = models.get(provider.lower())
    return result if result is not None else (None, None)


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
        _LOG.warning(f"Could not determine low-end model for provider: {provider}")
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

        _LOG.info("Message history summarized. Kept: system prompt, first input, and summary. Cleared all other messages.")
        return summary
    except httpx.HTTPError as e:
        _LOG.error(f"Failed to summarize message history: {e}")
        return ""
    except Exception as e:
        _LOG.error(f"Unexpected error during summarization: {e}")
        return ""


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
        _LOG.warning(f"Could not determine low-end model for provider: {provider}")
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

        _LOG.info("Message history summarized. Kept: system prompt, first input, and summary. Cleared all other messages.")
        return summary
    except httpx.HTTPError as e:
        _LOG.error(f"Failed to summarize message history: {e}")
        return ""
    except Exception as e:
        _LOG.error(f"Unexpected error during summarization: {e}")
        return ""
