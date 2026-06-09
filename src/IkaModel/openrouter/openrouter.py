# pyright: strict

from __future__ import annotations

from typing import Any, Optional, Protocol

from ..model_metadata import (
    openrouter_should_exclude_reasoning_for_tools,
    openrouter_supports_forced_tool_choice,
)
from ..openai.openai import openai_fill_payload

JsonDict = dict[str, Any]


class OpenRouterModel(Protocol):
    model_id: str
    max_tokens: int
    temperature: float
    agent_tools: list[Any]
    parallel_tool_calls: bool
    reasoning_effort: Optional[str]


def openrouter_fill_payload(
    model: OpenRouterModel,
    messages: list[JsonDict],
    message_history: Optional[JsonDict] = None,
    plugins: Optional[list[Any]] = None,
    response_format: Optional[JsonDict] = None,
    agent_tools: Optional[list[Any]] = None,
) -> JsonDict:
    """Build OpenRouter API payload (OpenAI-compatible with extensions)."""
    # Start with OpenAI payload as base
    payload = openai_fill_payload(model, messages, message_history, agent_tools=agent_tools)

    # Add OpenRouter-specific extensions
    if plugins:
        payload["plugins"] = plugins

    if response_format:
        payload["response_format"] = response_format

    # Many OpenRouter providers don't support forced tool_choice values
    # ("required" or {"type":"function","function":{...}}). Fall back to
    # "auto" for non-OpenAI/non-GLM models so the request doesn't 404.
    if not openrouter_supports_forced_tool_choice(model.model_id) and "tool_choice" in payload:
        tc = payload["tool_choice"]
        if tc != "auto" and tc != "none":
            payload["tool_choice"] = "auto"

    # Reasoning models (qwen3.6+, etc.) waste output tokens on internal CoT
    # which crowds out tool calls. Exclude reasoning when tools are present
    # so the model focuses on calling tools rather than thinking out loud.
    if openrouter_should_exclude_reasoning_for_tools(model.model_id) and "tools" in payload:
        payload["reasoning"] = {"exclude": True}

    # OpenRouter routes to multiple upstream providers and handles their
    # individual token limits itself, so remove OpenAI-specific model caps
    # that the base function may have applied (e.g. 16384 for gpt-4o-mini).
    # Preserve whichever key the base function chose (max_completion_tokens
    # vs max_tokens) to avoid sending both.
    max_tokens_value = model.max_tokens if model.max_tokens > 0 else 4096
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
