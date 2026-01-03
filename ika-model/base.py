import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

import httpx

_LOG = logging.getLogger(__name__)

try:
    from .deepseek import deepseek_fill_payload
except ImportError:
    def deepseek_fill_payload(model, messages):
        return {"model": model.model_id, "messages": messages}

try:
    from .openai import openai_fill_payload
except ImportError:
    def openai_fill_payload(model, messages):
        return {"model": model.model_id, "messages": messages}

try:
    from .claude import anthropic_fill_payload
except ImportError:
    def anthropic_fill_payload(model, messages):
        return {"model": model.model_id, "messages": messages}

_gemini_fill_payload_loaded = False
gemini_fill_payload = None

def _load_gemini_fill_payload():
    global gemini_fill_payload, _gemini_fill_payload_loaded
    if _gemini_fill_payload_loaded:
        return gemini_fill_payload
    
    try:
        from .google import gemini_fill_payload as _gemini_func
        gemini_fill_payload = _gemini_func
        _gemini_fill_payload_loaded = True
        return gemini_fill_payload
    except (ImportError, ModuleNotFoundError):
        try:
            import sys
            from pathlib import Path
            google_path = Path(__file__).parent / "google.py"
            import importlib.util
            spec = importlib.util.spec_from_file_location("google_module", google_path)
            google_mod = importlib.util.module_from_spec(spec)
            google_mod.MESSAGE_HISTORY = MESSAGE_HISTORY
            spec.loader.exec_module(google_mod)
            gemini_fill_payload = google_mod.gemini_fill_payload
            _gemini_fill_payload_loaded = True
            return gemini_fill_payload
        except Exception as e:
            _LOG.warning(f"Failed to load gemini_fill_payload: {e}")
            def _fallback(model, messages):
                return {"model": model.model_id, "messages": messages}
            gemini_fill_payload = _fallback
            _gemini_fill_payload_loaded = True
            return gemini_fill_payload


DEFAULT_SYSTEM_PROMPT = "You are a memeber in a complex Agentic system, every behavior you take is to help the system achieve its goals."

MESSAGE_HISTORY: dict = {
    "system": {"message": "", "tokens": 0},
    "first_input": {"message": "", "tokens": 0},
    "summary": {"message": "", "tokens": 0},
    "messages": {}
}

TOKENMAX_MAPPING = {
    "gpt-4o": 128000,                    # 128K tokens context window :contentReference[oaicite:2]{index=2}
    "gpt-4.1": 1000000,                  # 1M tokens context window :contentReference[oaicite:3]{index=3}
    "gpt-4.1-mini": 1000000,             # same 1M tokens context :contentReference[oaicite:4]{index=4}
    "gpt-4.1-nano": 1000000,             # same 1M tokens context :contentReference[oaicite:5]{index=5}

    # Anthropic – Claude family
    "claude-2": 100000,                  # ~100K context (historical) :contentReference[oaicite:6]{index=6}
    "claude-2.1": 200000,                # ~200K context (expanded) :contentReference[oaicite:7]{index=7}
    "claude-3-haiku": 200000,            # typical 200K context :contentReference[oaicite:8]{index=8}
    "claude-3-sonnet": 200000,           # typical 200K context :contentReference[oaicite:9]{index=9}
    "claude-3-opus": 200000,             # typical 200K context :contentReference[oaicite:10]{index=10}
    "claude-sonnet-4": 200000,           # base context (200K) :contentReference[oaicite:11]{index=11}
    "claude-opus-4": 200000,              # base context window :contentReference[oaicite:12]{index=12}
    "claude-sonnet-4 (1M beta)": 1000000,# 1M context via API beta/enterprise :contentReference[oaicite:13]{index=13}

    # Google – Gemini family
    "gemini-1.5-pro": 1000000,            # 1M token window on many configs :contentReference[oaicite:14]{index=14}
    "gemini-2.5-pro": 1000000,            # ~1M token window :contentReference[oaicite:15]{index=15}
    "gemini-3-pro": 1000000,              # ~1M token window reported :contentReference[oaicite:16]{index=16}
    # Some sources list expansions up to ~2M tokens for Pro variants on enterprise/preview :contentReference[oaicite:17]{index=17}

    # DeepSeek – V3.x family
    "deepseek-v3.2": 131072,              # ~131K tokens context window :contentReference[oaicite:18]{index=18}
    "deepseek-v3.2-speciale": 131072,     # similar ~131K context :contentReference[oaicite:19]{index=19}
    "deepseek-r1": 131072,                # ~131K context (preview/hosted) :contentReference[oaicite:20]{index=20}
}



@dataclass
class ToolArgs:
    type: str
    description: str
    # we are going to assume that all args are required


@dataclass
class AgentTool:
    def __init__(self, name: str, description: str, args: ToolArgs, required: bool = True): 
        self.validate(name)
        self.name = name
        self.description = description
        self.args = args
        self.required = required
    
    def validate(name: str):
        if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', name):
            # we need for deepseek
            raise ValueError("Name must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.")


@dataclass
class BareBoneModel:
    def __init__(self, model_id: str, api_key: str, api_url: str, system_prompt=DEFAULT_SYSTEM_PROMPT, content_prompt: str = "", max_tokens: int = 20000, temperature: float = 0):
        self.model_id = model_id
        self.api_key = api_key
        self.api_url = api_url if api_url else self.find_api_url(model_id)  
        self.system_prompt = system_prompt
        self.content_prompt = content_prompt
        self.max_tokens = max_tokens
        if temperature < 0 or temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        self.temperature = temperature # define this as a percentage between 0 and 1, for any provider that uses a different scale, we will need to convert it to the correct scale
        self.agent_tools: list[AgentTool] = [] 
        self.deepthinking: bool = False
        MESSAGE_HISTORY["system"]["message"] = self.system_prompt
        MESSAGE_HISTORY["system"]["tokens"] = 0

    def find_api_url(self, model_id: str) -> str:
        return {
            "deepseek": "https://api.deepseek.com/chat/completions",
            "openai": "https://api.openai.com/v1/chat/completions",
            "anthropic": "https://api.anthropic.com/v1/messages",
            "gemini": "https://api.gemini.com/v1/chat/completions",
        }.get(model_id)


_SUMMARY_PROMPT_PATH = Path(__file__).parent / "summary_prompt"
with open(_SUMMARY_PROMPT_PATH, "r", encoding="utf-8") as f:
    SUMMARY_PROMPT = f.read()


def _get_low_end_model(provider: str) -> tuple[str, str]:
    """Returns (model_name, api_url) for the cheapest model of the given provider."""
    models = {
        "deepseek": ("deepseek-v3.2", "https://api.deepseek.com/chat/completions"),
        "openai": ("gpt-4.1-mini-2025-04-14", "https://api.openai.com/v1/chat/completions"),
        "anthropic": ("claude-sonnet-4-20250514", "https://api.anthropic.com/v1/messages"),
        "gemini": ("gemini-1.5-pro", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"),
    }
    return models.get(provider.lower(), (None, None))


def _get_provider_from_model_id(model_id: str) -> str:
    """Extracts provider name from model_id."""
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
    """Creates payload and headers for summarization request based on provider."""
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
    """Parses the API response based on provider format."""
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


def _get_total_tokens() -> int:
    """Calculates total tokens from MESSAGE_HISTORY."""
    total = 0
    total += MESSAGE_HISTORY["system"]["tokens"]
    total += MESSAGE_HISTORY["first_input"]["tokens"]
    total += MESSAGE_HISTORY["summary"]["tokens"]
    total += sum(msg["tokens"] for msg in MESSAGE_HISTORY["messages"].values())
    return total


def _get_conversation_text() -> str:
    """Builds conversation text from MESSAGE_HISTORY for summarization."""
    parts = []
    if MESSAGE_HISTORY["first_input"]["message"]:
        parts.append(MESSAGE_HISTORY["first_input"]["message"])
    if MESSAGE_HISTORY["summary"]["message"]:
        parts.append(MESSAGE_HISTORY["summary"]["message"])
    for msg_id in sorted(MESSAGE_HISTORY["messages"].keys()):
        parts.append(MESSAGE_HISTORY["messages"][msg_id]["message"])
    return "\n".join(parts)


def summarise_message_history(barebone_model: BareBoneModel) -> str:
    """Summarizes the entire message history using a low-end model from the same provider.
    After summarization, keeps only: system prompt, first input, and the new summary.
    Clears all other messages."""
    if not MESSAGE_HISTORY["first_input"]["message"] and not MESSAGE_HISTORY["messages"]:
        return ""
    
    conversation_text = _get_conversation_text()
    
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
        
        MESSAGE_HISTORY["summary"]["message"] = f"[SUMMARY]\n{summary}"
        MESSAGE_HISTORY["summary"]["tokens"] = summary_tokens
        MESSAGE_HISTORY["messages"] = {}
        
        _LOG.info("Message history summarized. Kept: system prompt, first input, and summary. Cleared all other messages.")
        return summary
    except httpx.HTTPError as e:
        _LOG.error(f"Failed to summarize message history: {e}")
        return ""
    except Exception as e:
        _LOG.error(f"Unexpected error during summarization: {e}")
        return ""

def _get_model_max_tokens(model_id: str) -> int:
    """Gets the maximum token limit for a model, with fallback logic."""
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


def chat(barebone_model: BareBoneModel, messages: list[dict]) -> str:
    """Main chat function with sliding window management."""
    
    token_count = _get_total_tokens()
    max_tokens = _get_model_max_tokens(barebone_model.model_id)
    
    if token_count > max_tokens * 0.8:
        _LOG.info(f"Token count ({token_count}) approaching limit ({max_tokens}). Summarizing history...")
        summarise_message_history(barebone_model)
    
    if not MESSAGE_HISTORY["first_input"]["message"] and messages:
        MESSAGE_HISTORY["first_input"]["message"] = messages[0].get("content", str(messages[0]))
        MESSAGE_HISTORY["first_input"]["tokens"] = 0
    
    response = None
    headers = {}

    if barebone_model.model_id.lower().startswith("deepseek") or "deepseek" in barebone_model.model_id.lower():
        headers = {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}
        payload = deepseek_fill_payload(barebone_model, messages)
        response = httpx.post(barebone_model.api_url, headers=headers, json=payload)
    elif barebone_model.model_id.lower().startswith("gpt") or "openai" in barebone_model.model_id.lower():
        headers = {"Authorization": f"Bearer {barebone_model.api_key}", "Content-Type": "application/json"}
        payload = openai_fill_payload(barebone_model, messages)
        response = httpx.post(barebone_model.api_url, headers=headers, json=payload)
    elif "claude" in barebone_model.model_id.lower() or "anthropic" in barebone_model.model_id.lower():
        headers = {"x-api-key": barebone_model.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        payload = anthropic_fill_payload(barebone_model, messages)
        if "max_tokens" not in payload or not payload["max_tokens"]:
            payload["max_tokens"] = 4096
        _LOG.debug(f"Anthropic payload: {json.dumps(payload, indent=2)[:500]}")
        response = httpx.post(barebone_model.api_url, headers=headers, json=payload)
    elif "gemini" in barebone_model.model_id.lower():
        _load_gemini_fill_payload()
        payload = gemini_fill_payload(barebone_model, messages)
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
        
        if "deepseek" in barebone_model.model_id.lower() or "gpt" in barebone_model.model_id.lower() or "openai" in barebone_model.model_id.lower():
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
        elif "claude" in barebone_model.model_id.lower():
            content_blocks = data.get("content", [])
            content = "".join([block["text"] for block in content_blocks if block.get("type") == "text"])
            usage = data.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        elif "gemini" in barebone_model.model_id.lower():
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            tokens = usage.get("totalTokenCount", 0)
        else:
            content = ""
            tokens = 0
        
        msg_id = str(uuid.uuid4())
        MESSAGE_HISTORY["messages"][msg_id] = {"message": content, "tokens": tokens}
        return content
    
    return ""