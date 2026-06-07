import json
from typing import Any, Dict, List, Optional

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


def _append_deepseek_context(api_messages: List[Dict[str, Any]], message_history: Dict[str, Any]) -> None:
    if message_history["system"]["message"]:
        api_messages.append({"role": "system", "content": message_history["system"]["message"]})
    if message_history["first_input"]["message"]:
        api_messages.append({"role": "user", "content": message_history["first_input"]["message"]})
    if message_history["summary"]["message"]:
        api_messages.append({"role": "assistant", "content": message_history["summary"]["message"]})


def _append_deepseek_history(api_messages: List[Dict[str, Any]], message_history: Dict[str, Any]) -> None:
    for msg_id in message_history["messages"]:
        msg = message_history["messages"][msg_id]
        msg_type = msg.get("type", "assistant")
        raw = msg.get("message", "")
        if msg_type in {"assistant_with_tools", "tool"}:
            try:
                api_messages.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            assistant_entry = {"role": "assistant", "content": raw if isinstance(raw, str) else str(raw)}
            reasoning_text = msg.get("reasoning_content")
            if reasoning_text:
                assistant_entry["reasoning_content"] = reasoning_text
            api_messages.append(assistant_entry)


def _append_live_messages(api_messages: List[Dict[str, Any]], messages: List[Dict[str, Any]], first_input_content: str) -> None:
    skip_first = bool(first_input_content and messages and _message_content(messages[0]) == first_input_content)
    for index, msg in enumerate(messages):
        if skip_first and index == 0:
            continue
        if isinstance(msg, dict):
            if "role" in msg and "content" in msg:
                api_messages.append(msg)
            elif "content" in msg:
                api_messages.append({"role": "user", "content": msg["content"]})
            else:
                api_messages.append({"role": "user", "content": str(msg)})
        else:
            api_messages.append({"role": "user", "content": str(msg)})


def _deepseek_max_tokens(model: Any) -> int:
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    return min(max_tokens_value, 32768)


def deepseek_fill_payload(
    model: Any,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None,
    agent_tools: Optional[list[Any]] = None,
) -> Dict[str, Any]:
    message_history = message_history or _default_message_history()
    api_messages: List[Dict[str, Any]] = []
    _append_deepseek_context(api_messages, message_history)
    _append_deepseek_history(api_messages, message_history)
    _append_live_messages(api_messages, messages, message_history["first_input"]["message"])

    payload = {
        "model": model.model_id,
        "messages": api_messages,
        "temperature": model.temperature,
        "max_tokens": _deepseek_max_tokens(model),
        "stream": False,
    }

    model_id_lower = (model.model_id or "").lower()
    thinking_enabled = bool(getattr(model, "deepthinking", False)) or "reasoner" in model_id_lower
    if thinking_enabled:
        payload["thinking"] = {"type": "enabled"}
    else:
        payload["thinking"] = {"type": "disabled"}

    agent_tools = model.agent_tools if agent_tools is None else agent_tools

    if agent_tools:
        # DeepSeek defaults to auto tool choice when tools are present, and
        # some V4 routes reject even tool_choice="auto".
        payload["tools"] = build_provider_tool_payload("deepseek", agent_tools).tools

    return payload
