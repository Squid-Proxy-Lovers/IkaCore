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

    # OpenRouter routes to multiple upstream providers and handles their
    # individual token limits itself, so remove OpenAI-specific model caps
    # that the base function may have applied (e.g. 16384 for gpt-4o-mini).
    # Preserve whichever key the base function chose (max_completion_tokens
    # vs max_tokens) to avoid sending both.
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = max_tokens_value
        # Ensure we don't have both keys
        payload.pop("max_tokens", None)
    elif "max_tokens" in payload:
        payload["max_tokens"] = max_tokens_value
        payload.pop("max_completion_tokens", None)
    else:
        # Fallback if base function set neither (shouldn't happen)
        payload["max_tokens"] = max_tokens_value

    return payload
