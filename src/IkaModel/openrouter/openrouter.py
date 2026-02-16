from typing import Any, Dict, List, Optional
from ..openai.openai import openai_fill_payload


def openrouter_fill_payload(
    model,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None,
    plugins: Optional[List[str]] = None,
    response_format: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build OpenRouter API payload (OpenAI-compatible with extensions)."""
    # Start with OpenAI payload as base
    payload = openai_fill_payload(model, messages, message_history)

    # Add OpenRouter-specific extensions
    if plugins:
        payload["plugins"] = plugins

    if response_format:
        payload["response_format"] = response_format

    # Override max_tokens (OpenRouter routes to multiple providers, no OpenAI-specific limits)
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    is_reasoning_model = any(x in model.model_id.lower() for x in ["o1", "o3"])

    if is_reasoning_model:
        payload["max_completion_tokens"] = max_tokens_value
    else:
        payload["max_tokens"] = max_tokens_value

    return payload
