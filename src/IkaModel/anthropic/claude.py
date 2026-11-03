import json
from typing import Any, Dict, List, Optional

from ..model_metadata import is_anthropic_haiku_model, supports_anthropic_parallel_tool_use
from ..tool_schema import build_provider_tool_payload

_PARALLEL_TOOL_PROMPT = "\n\n<use_parallel_tool_calls>\nFor maximum efficiency, whenever you perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially. Prioritize calling tools in parallel whenever possible. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. When running multiple read-only commands like `ls` or `list_dir`, always run all of the commands in parallel. Err on the side of maximizing parallel tool calls rather than running too many tools sequentially.\n</use_parallel_tool_calls>"


def _ensure_anthropic_assistant_content(msg: Dict[str, Any]) -> Dict[str, Any]:
    if msg.get("role") != "assistant":
        return msg
    content = msg.get("content")
    if isinstance(content, list) and content:
        has_text = any(block.get("type") == "text" for block in content)
        if not has_text:
            msg = {**msg, "content": [{"type": "text", "text": " "}] + content}
    elif isinstance(content, list) and not content:
        msg = {**msg, "content": [{"type": "text", "text": " "}]}
    return msg


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


def _append_anthropic_context(api_messages: List[Dict[str, Any]], message_history: Dict[str, Any]) -> None:
    if message_history["first_input"]["message"]:
        api_messages.append({"role": "user", "content": message_history["first_input"]["message"]})
    if message_history["summary"]["message"]:
        api_messages.append({"role": "assistant", "content": message_history["summary"]["message"]})


def _append_anthropic_history(api_messages: List[Dict[str, Any]], message_history: Dict[str, Any]) -> None:
    for msg_id in message_history["messages"]:
        msg = message_history["messages"][msg_id]
        msg_type = msg.get("type", "assistant")
        raw = msg.get("message", "")
        if msg_type == "assistant_with_tools":
            try:
                api_messages.append(_ensure_anthropic_assistant_content(json.loads(raw)))
            except (json.JSONDecodeError, TypeError):
                continue
        elif msg_type == "tool":
            try:
                api_messages.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            api_messages.append({"role": "assistant", "content": raw if isinstance(raw, str) else str(raw)})


def _append_anthropic_live_messages(api_messages: List[Dict[str, Any]], messages: List[Dict[str, Any]], first_input_content: str) -> None:
    skip_first = bool(first_input_content and messages and _message_content(messages[0]) == first_input_content)
    for index, msg in enumerate(messages):
        if skip_first and index == 0:
            continue
        if isinstance(msg, dict):
            if "role" in msg and "content" in msg:
                if msg["role"] == "user":
                    api_messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    api_messages.append(_ensure_anthropic_assistant_content({"role": "assistant", "content": msg["content"]}))
            elif "content" in msg:
                api_messages.append({"role": "user", "content": msg["content"]})
            else:
                api_messages.append({"role": "user", "content": str(msg)})
        else:
            api_messages.append({"role": "user", "content": str(msg)})


def _anthropic_max_tokens(model: Any) -> int:
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    if is_anthropic_haiku_model(model.model_id):
        return min(max_tokens_value, 4096)
    return max_tokens_value


def _anthropic_system_prompt(model: Any, message_history: Dict[str, Any]) -> str:
    system_prompt = message_history["system"]["message"] or model.system_prompt or ""
    if getattr(model, "parallel_tool_calls", False) and supports_anthropic_parallel_tool_use(model.model_id):
        if _PARALLEL_TOOL_PROMPT not in system_prompt:
            system_prompt += _PARALLEL_TOOL_PROMPT
    return system_prompt


def _anthropic_tool_choice(model: Any, tool_payload: Any) -> Dict[str, Any]:
    forced_tool_name = getattr(model, "forced_tool_name", None)
    if forced_tool_name and forced_tool_name in tool_payload.names:
        tool_choice: Dict[str, Any] = {"type": "tool", "name": forced_tool_name}
    elif len(tool_payload.required_names) == 1:
        tool_choice = {"type": "tool", "name": tool_payload.required_names[0]}
    elif len(tool_payload.required_names) > 1:
        tool_choice = {"type": "required"}
    else:
        tool_choice = {"type": "auto"}

    if hasattr(model, 'parallel_tool_calls') and not model.parallel_tool_calls:
        if supports_anthropic_parallel_tool_use(model.model_id) and tool_choice.get("type") in ["auto", "required"]:
            tool_choice["disable_parallel_tool_use"] = True
    return tool_choice


def _apply_anthropic_tools(payload: Dict[str, Any], model: Any, agent_tools: list[Any]) -> None:
    tool_payload = build_provider_tool_payload("anthropic", agent_tools)
    payload["tools"] = tool_payload.tools
    payload["tool_choice"] = _anthropic_tool_choice(model, tool_payload)


def anthropic_fill_payload(
    model: Any,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None,
    agent_tools: Optional[list[Any]] = None,
) -> Dict[str, Any]:
    message_history = message_history or _default_message_history()
    api_messages: List[Dict[str, Any]] = []
    _append_anthropic_context(api_messages, message_history)
    _append_anthropic_history(api_messages, message_history)
    _append_anthropic_live_messages(api_messages, messages, message_history["first_input"]["message"])

    payload = {
        "model": model.model_id,
        "max_tokens": _anthropic_max_tokens(model),
        "temperature": model.temperature,
        "system": _anthropic_system_prompt(model, message_history),
        "messages": api_messages,
    }

    agent_tools = model.agent_tools if agent_tools is None else agent_tools
    if agent_tools:
        _apply_anthropic_tools(payload, model, agent_tools)

    return payload
