import json
import logging
import uuid
from typing import Any, Dict, Optional, List

import httpx

from .base import BareBoneModel, TOKENMAX_MAPPING, SUMMARY_PROMPT
from .deepseek import deepseek_fill_payload
from .openai import openai_fill_payload
from .claude import anthropic_fill_payload
from .google import gemini_fill_payload

_LOG = logging.getLogger(__name__)


def _get_low_end_model(provider: str) -> tuple[str, str]:
    models = {
        "deepseek": ("deepseek-v3.2", "https://api.deepseek.com/chat/completions"),
        "openai": ("gpt-4.1-mini-2025-04-14", "https://api.openai.com/v1/chat/completions"),
        "anthropic": ("claude-sonnet-4-20250514", "https://api.anthropic.com/v1/messages"),
        "gemini": ("gemini-1.5-pro", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"),
    }
    return models.get(provider.lower(), (None, None))


def _get_provider_from_model_id(model_id: str) -> str:
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


def _create_summary_payload(provider: str, model_name: str, api_key: str, conversation_text: str) -> tuple[dict, dict]:
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


def _parse_summary_response(provider: str, response: httpx.Response) -> str:
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


def _init_message_history() -> dict:
    return {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def _get_total_tokens(message_history: dict) -> int:
    total = 0
    total += message_history["system"]["tokens"]
    total += message_history["first_input"]["tokens"]
    total += message_history["summary"]["tokens"]
    total += sum(msg["tokens"] for msg in message_history["messages"].values())
    return total


def _get_conversation_text(message_history: dict) -> str:
    parts = []
    if message_history["first_input"]["message"]:
        parts.append(message_history["first_input"]["message"])
    if message_history["summary"]["message"]:
        parts.append(message_history["summary"]["message"])
    for msg_id in sorted(message_history["messages"].keys()):
        parts.append(message_history["messages"][msg_id]["message"])
    return "\n".join(parts)


def summarise_message_history(barebone_model: BareBoneModel, message_history: dict) -> str:
    if not message_history["first_input"]["message"] and not message_history["messages"]:
        return ""
    
    conversation_text = _get_conversation_text(message_history)
    
    provider = _get_provider_from_model_id(barebone_model.model_id)
    model_name, api_url = _get_low_end_model(provider)
    
    if not model_name or not api_url:
        _LOG.warning(f"Could not determine low-end model for provider: {provider}")
        return ""
    
    payload, headers = _create_summary_payload(provider, model_name, barebone_model.api_key, conversation_text)
    
    try:
        response = httpx.post(api_url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        summary = _parse_summary_response(provider, response)
        
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

def _get_model_max_tokens(model_id: str) -> int:
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
        return TOKENMAX_MAPPING.get("deepseek-v3.2", 131072)
    
    return 128000


def chat(barebone_model: BareBoneModel, messages: list[dict], message_history: Optional[dict] = None) -> Dict[str, Any]:
    message_history = message_history or _init_message_history()
    
    token_count = _get_total_tokens(message_history)
    max_tokens = _get_model_max_tokens(barebone_model.model_id)
    
    if token_count > max_tokens * 0.8:
        _LOG.info(f"Token count ({token_count}) approaching limit ({max_tokens}). Summarizing history...")
        summarise_message_history(barebone_model, message_history)
    
    if not message_history["first_input"]["message"] and messages:
        message_history["first_input"]["message"] = messages[0].get("content", str(messages[0]))
        message_history["first_input"]["tokens"] = 0
    
    response = None
    headers = {}

    if barebone_model.model_id.lower().startswith("deepseek") or "deepseek" in barebone_model.model_id.lower():
        headers = {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}
        payload = deepseek_fill_payload(barebone_model, messages, message_history)
        response = httpx.post(barebone_model.api_url, headers=headers, json=payload)

    elif barebone_model.model_id.lower().startswith("gpt") or "openai" in barebone_model.model_id.lower():
        headers = {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}
        payload = openai_fill_payload(barebone_model, messages, message_history)
        response = httpx.post(barebone_model.api_url, headers=headers, json=payload)

    elif "claude" in barebone_model.model_id.lower() or "anthropic" in barebone_model.model_id.lower():
        headers = {"x-api-key": barebone_model.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        payload = anthropic_fill_payload(barebone_model, messages, message_history)
        if "max_tokens" not in payload or not payload["max_tokens"]:
            payload["max_tokens"] = 4096
        _LOG.debug(f"Anthropic payload: {json.dumps(payload, indent=2)[:500]}")
        response = httpx.post(barebone_model.api_url, headers=headers, json=payload)
        
    elif "gemini" in barebone_model.model_id.lower():
        payload = gemini_fill_payload(barebone_model, messages, message_history)
        api_url = f"{barebone_model.api_url}?key={barebone_model.api_key}"
        headers = {"Content-Type": "application/json"}
        response = httpx.post(api_url, headers=headers, json=payload)
    else:
        raise ValueError(f"Model {barebone_model.model_id} not supported")
    
    if response is not None:
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_text = json.dumps(error_data, indent=2)[:1000]
            except:
                error_text = response.text[:500] if hasattr(response, 'text') else str(response.status_code)
            _LOG.error(f"API error {response.status_code}: {error_text}")
            raise Exception(f"API error {response.status_code}: {error_text}")
        response.raise_for_status()
        data = response.json()
        
        content = ""
        tokens = 0
        tool_calls: List[dict] = []

        if "deepseek" in barebone_model.model_id.lower() or "gpt" in barebone_model.model_id.lower() or "openai" in barebone_model.model_id.lower():
            message_obj = data["choices"][0]["message"]
            content = message_obj.get("content") or ""
            tool_calls = message_obj.get("tool_calls", []) or []
            tokens = data.get("usage", {}).get("total_tokens", 0)
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
            usage = data.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        elif "gemini" in barebone_model.model_id.lower():
            candidate = data["candidates"][0]["content"]
            parts = candidate.get("parts", [])
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
                        }
                    })
            usage = data.get("usageMetadata", {})
            tokens = usage.get("totalTokenCount", 0)
        else:
            content = ""
            tokens = 0
        
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {"message": content, "tokens": tokens}
        return {"content": content, "tool_calls": tool_calls, "message_history": message_history}
    
    return {"content": "", "tool_calls": [], "message_history": message_history}