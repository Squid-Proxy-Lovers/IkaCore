import json
from typing import Any, Dict, List, Optional

from ..model_metadata import supports_custom_temperature
from ..tool_schema import build_provider_tool_payload


def _default_message_history() -> Dict[str, Any]:
    return {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return message.get("content", str(message))
    return str(message)


def _should_skip_first_message(messages: List[Dict[str, Any]], first_input_content: str) -> bool:
    return bool(first_input_content and messages and _message_content(messages[0]) == first_input_content)


def _append_responses_context(input_items: List[Dict[str, Any]], message_history: Dict[str, Any]) -> None:
    if message_history["summary"]["message"]:
        input_items.append({
            "type": "message",
            "role": "assistant",
            "content": message_history["summary"]["message"],
        })
    if message_history["first_input"]["message"]:
        input_items.append({
            "type": "message",
            "role": "user",
            "content": message_history["first_input"]["message"],
        })


def _tool_message_item(message: Dict[str, Any]) -> Dict[str, Any]:
    content = message.get("content", "")
    return {
        "type": "function_call_output",
        "call_id": message.get("tool_call_id", ""),
        "output": content if isinstance(content, str) else json.dumps(content),
    }


def _assistant_message_items(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", []) or []
    if not tool_calls:
        return [{"type": "message", "role": "assistant", "content": content}] if content else []
    items = []
    for tool_call in tool_calls:
        fn = tool_call.get("function", {})
        items.append({
            "type": "function_call",
            "call_id": tool_call.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        })
    return items


def _user_message_items(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = message.get("content", "")
    if not isinstance(content, list):
        return [{"type": "message", "role": "user", "content": content or str(message)}]

    items = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            items.append({
                "type": "function_call_output",
                "call_id": block.get("tool_use_id", ""),
                "output": block.get("content", ""),
            })
        else:
            items.append({
                "type": "message",
                "role": "user",
                "content": json.dumps(block) if not isinstance(block, str) else block,
            })
    return items


def _responses_items_for_message(message: Any) -> List[Dict[str, Any]]:
    if not isinstance(message, dict):
        return [{"type": "message", "role": "user", "content": str(message)}]
    role = message.get("role", "user")
    if role == "tool":
        return [_tool_message_item(message)]
    if role == "assistant":
        return _assistant_message_items(message)
    if role == "user":
        return _user_message_items(message)
    content = message.get("content", "")
    return [{"type": "message", "role": role, "content": content or str(message)}]


def _build_responses_input_items(messages: List[Dict[str, Any]], message_history: Dict[str, Any]) -> List[Dict[str, Any]]:
    input_items: List[Dict[str, Any]] = []
    _append_responses_context(input_items, message_history)
    skip_first = _should_skip_first_message(messages, message_history["first_input"]["message"])
    for index, message in enumerate(messages):
        if skip_first and index == 0:
            continue
        input_items.extend(_responses_items_for_message(message))
    return input_items


def _apply_responses_tool_config(payload: Dict[str, Any], model: Any, agent_tools: list[Any]) -> None:
    tool_payload = build_provider_tool_payload("openai_responses", agent_tools)
    payload["tools"] = tool_payload.tools

    forced_tool_name = getattr(model, "forced_tool_name", None)
    if forced_tool_name and forced_tool_name in tool_payload.names:
        payload["tool_choice"] = {"type": "function", "name": forced_tool_name}
    elif len(tool_payload.required_names) == 1:
        payload["tool_choice"] = {"type": "function", "name": tool_payload.required_names[0]}
    elif len(tool_payload.required_names) > 1:
        payload["tool_choice"] = "required"
    else:
        payload["tool_choice"] = "auto"

    payload["parallel_tool_calls"] = bool(hasattr(model, "parallel_tool_calls") and model.parallel_tool_calls)


def openai_responses_fill_payload(
    model: Any,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None,
    agent_tools: Optional[list[Any]] = None,
) -> Dict[str, Any]:
    """Build a payload for the OpenAI Responses API (POST /v1/responses)."""
    message_history = message_history or _default_message_history()

    payload: Dict[str, Any] = {
        "model": model.model_id,
        "input": _build_responses_input_items(messages, message_history),
        "store": False,  # We manage our own conversation state
    }

    if message_history["system"]["message"]:
        payload["instructions"] = message_history["system"]["message"]

    # max_output_tokens instead of max_tokens / max_completion_tokens
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    payload["max_output_tokens"] = max_tokens_value

    if supports_custom_temperature(model.model_id):
        payload["temperature"] = model.temperature

    # reasoning_effort
    if getattr(model, "reasoning_effort", None):
        payload["reasoning"] = {"effort": model.reasoning_effort}

    agent_tools = model.agent_tools if agent_tools is None else agent_tools

    # Tools — flatter schema: no nested "function" key
    if agent_tools:
        _apply_responses_tool_config(payload, model, agent_tools)

    return payload
