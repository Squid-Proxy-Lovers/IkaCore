"""Provider-specific tool result formatting helpers."""

# pyright: strict

from __future__ import annotations

import json
from typing import Any, Callable, TypeAlias, cast

JsonDict: TypeAlias = dict[str, Any]
ToolCall: TypeAlias = JsonDict
ToolMessage: TypeAlias = JsonDict
ToolResultFormatterFn = Callable[[list[ToolCall], list[str]], list[ToolMessage]]


def _as_dict(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return cast(JsonDict, value)
    return {}


def _tool_result_at(tool_results: list[str], index: int) -> str:
    return tool_results[index] if index < len(tool_results) else json.dumps({"error": "No result"})


def _tool_call_id(tool_call: ToolCall, fallback: str) -> str:
    return str(tool_call.get("id", fallback))


def format_openai_responses_results(tool_calls: list[ToolCall], tool_results: list[str]) -> list[ToolMessage]:
    """Format tool results for the Responses API (role='tool' with tool_call_id).

    The openai_responses payload builder converts these to function_call_output
    items automatically when it processes the messages list.
    """
    tool_messages: list[ToolMessage] = []
    for i, tool_call in enumerate(tool_calls):
        tool_call_id = _tool_call_id(tool_call, f"call_{i}")
        tool_messages.append({
            "role": "tool",
            "content": _tool_result_at(tool_results, i),
            "tool_call_id": tool_call_id
        })
    return tool_messages


def format_openai_results(tool_calls: list[ToolCall], tool_results: list[str]) -> list[ToolMessage]:
    tool_messages: list[ToolMessage] = []
    for i, tool_call in enumerate(tool_calls):
        tool_call_id = _tool_call_id(tool_call, f"call_{i}")
        tool_messages.append({
            "role": "tool",
            "content": _tool_result_at(tool_results, i),
            "tool_call_id": tool_call_id
        })
    return tool_messages


def format_anthropic_results(tool_calls: list[ToolCall], tool_results: list[str]) -> list[ToolMessage]:
    tool_messages: list[ToolMessage] = []
    for i, tool_call in enumerate(tool_calls):
        tool_call_id = _tool_call_id(tool_call, f"call_{i}")
        tool_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": _tool_result_at(tool_results, i)
                }
            ]
        })
    return tool_messages


def format_gemini_results(tool_calls: list[ToolCall], tool_results: list[str]) -> list[ToolMessage]:
    function_responses: list[ToolMessage] = []
    for i, tool_call in enumerate(tool_calls):
        function_payload = _as_dict(tool_call.get("function"))
        tool_name = str(tool_call.get("name") or function_payload.get("name", ""))
        result_data = _coerce_gemini_tool_result(tool_results[i] if i < len(tool_results) else None)
        if not isinstance(result_data, dict):
            result_data = {"result": result_data}
        function_responses.append({
            "functionResponse": {
                "name": tool_name,
                "response": result_data
            }
        })
    return function_responses


def _coerce_gemini_tool_result(result: Any) -> Any:
    if result is None:
        return {"error": "No result"}
    if isinstance(result, dict):
        return cast(JsonDict, result)
    if not isinstance(result, str):
        return result

    stripped = result.strip()
    if stripped == "{}":
        return {}
    if not stripped:
        return {"result": result}
    if stripped[0] not in "{[\"-0123456789tfn":
        return {"result": result}
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return {"result": result}


TOOL_RESULT_FORMATTERS: dict[str, ToolResultFormatterFn] = {
    "anthropic": format_anthropic_results,
    "codex": format_openai_responses_results,
    "deepseek": format_openai_results,
    "gemini": format_gemini_results,
    "openai": format_openai_results,
    "openai_responses": format_openai_responses_results,
    "openrouter": format_openai_results,
}

def format_provider_tool_results(provider: str, tool_calls: list[ToolCall], tool_results: list[str]) -> list[ToolMessage]:
    formatter = TOOL_RESULT_FORMATTERS.get(provider)
    if formatter is None:
        return []
    return formatter(tool_calls, tool_results)
