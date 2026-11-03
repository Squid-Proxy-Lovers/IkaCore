# pyright: strict

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Protocol, cast

from ..model_metadata import (
    cap_openai_chat_completion_tokens,
    supports_custom_temperature,
    uses_openai_max_completion_tokens,
)
from ..reasoning_replay import ordered_replay_messages
from ..tool_schema import build_provider_tool_payload

JsonDict = dict[str, Any]


class OpenAIPayloadModel(Protocol):
    model_id: str
    max_tokens: int
    temperature: float
    agent_tools: list[Any]
    parallel_tool_calls: bool
    reasoning_effort: Optional[str]


def _default_message_history() -> JsonDict:
    return {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def _history_message(message_history: JsonDict, key: str) -> str:
    entry = message_history.get(key)
    if not isinstance(entry, Mapping):
        return ""
    message = cast(Mapping[str, object], entry).get("message")
    return message if isinstance(message, str) else ""


def _normalise_openai_message(message: object, replay_provider: Optional[str] = None) -> JsonDict:
    if isinstance(message, dict) or isinstance(message, Mapping):
        mapped = cast(Mapping[str, Any], message)
        if "role" in mapped and "content" in mapped:
            normal = {key: value for key, value in mapped.items() if key != "provider_data"}
            data = mapped.get("provider_data")
            if isinstance(data, dict):
                replay_data = cast(JsonDict, data)
                if replay_data.get("provider") == replay_provider:
                    normal.update(replay_data.get("fields", {}))
            return normal
        if "content" in mapped:
            return {"role": "user", "content": mapped["content"]}
        return {"role": "user", "content": str(dict(mapped))}
    return {"role": "user", "content": str(message)}


def _build_openai_messages(messages: list[JsonDict], message_history: JsonDict,
                           replay_provider: Optional[str] = None) -> list[JsonDict]:
    system = _history_message(message_history, "system")
    api_messages = [{"role": "system", "content": system}] if system else []
    api_messages.extend(_normalise_openai_message(message, replay_provider)
                        for message in ordered_replay_messages(messages, message_history, summary_first=False))
    return api_messages


def _apply_generation_config(payload: JsonDict, model: OpenAIPayloadModel) -> None:
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    max_tokens_value = cap_openai_chat_completion_tokens(model.model_id, max_tokens_value)
    if supports_custom_temperature(model.model_id):
        payload["temperature"] = model.temperature
    if uses_openai_max_completion_tokens(model.model_id):
        payload["max_completion_tokens"] = max_tokens_value
    else:
        payload["max_tokens"] = max_tokens_value


def _apply_tool_config(payload: JsonDict, model: OpenAIPayloadModel, agent_tools: list[Any]) -> None:
    tool_payload = build_provider_tool_payload("openai", agent_tools)
    payload["tools"] = tool_payload.tools

    forced_tool_name = getattr(model, "forced_tool_name", None)
    if forced_tool_name and forced_tool_name in tool_payload.names:
        payload["tool_choice"] = {"type": "function", "function": {"name": forced_tool_name}}
    elif len(tool_payload.required_names) == 1:
        payload["tool_choice"] = {"type": "function", "function": {"name": tool_payload.required_names[0]}}
    elif len(tool_payload.required_names) > 1:
        payload["tool_choice"] = "required"
    else:
        payload["tool_choice"] = "auto"

    payload["parallel_tool_calls"] = bool(hasattr(model, 'parallel_tool_calls') and model.parallel_tool_calls)


def openai_fill_payload(
    model: OpenAIPayloadModel,
    messages: list[JsonDict],
    message_history: Optional[JsonDict] = None,
    agent_tools: Optional[list[Any]] = None,
    *, replay_provider: Optional[str] = None,
) -> JsonDict:
    message_history = message_history or _default_message_history()
    payload: JsonDict = {
        "model": model.model_id,
        "messages": _build_openai_messages(messages, message_history, replay_provider),
    }
    _apply_generation_config(payload, model)

    is_zai_endpoint = "api.z.ai/" in str(getattr(model, "api_url", "")).lower()
    if is_zai_endpoint:
        payload["thinking"] = {"type": "enabled"}

    # reasoning_effort is not supported with function tools on /v1/chat/completions.
    # Keep it only when tools are absent (or when using the Responses API path elsewhere).
    agent_tools = model.agent_tools if agent_tools is None else agent_tools
    if getattr(model, "reasoning_effort", None) and not agent_tools and not is_zai_endpoint:
        payload["reasoning_effort"] = model.reasoning_effort
    if agent_tools:
        _apply_tool_config(payload, model, agent_tools)
    return payload
