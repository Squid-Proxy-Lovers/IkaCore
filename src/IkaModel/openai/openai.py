from typing import Any, Dict, List, Optional

from ..model_metadata import (
    cap_openai_chat_completion_tokens,
    supports_custom_temperature,
    uses_openai_max_completion_tokens,
)
from ..tool_schema import build_provider_tool_payload


def openai_fill_payload(
    model,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None,
    agent_tools: Optional[list[Any]] = None,
) -> Dict[str, Any]:
    message_history = message_history or {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {}
    }
    api_messages = []
    
    if message_history["system"]["message"]:
        api_messages.append({
            "role": "system",
            "content": message_history["system"]["message"]
        })
    
    if message_history["first_input"]["message"]:
        api_messages.append({
            "role": "user",
            "content": message_history["first_input"]["message"]
        })
    
    if message_history["summary"]["message"]:
        api_messages.append({
            "role": "assistant",
            "content": message_history["summary"]["message"]
        })

    # Skip first message if it duplicates first_input to prevent duplicate user messages
    first_input_content = message_history["first_input"]["message"]
    skip_first = False
    if first_input_content and messages:
        first_msg = messages[0]
        first_msg_content = ""
        if isinstance(first_msg, dict):
            first_msg_content = first_msg.get("content", str(first_msg))
        else:
            first_msg_content = str(first_msg)
        if first_msg_content == first_input_content:
            skip_first = True

    for i, msg in enumerate(messages):
        if skip_first and i == 0:
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
    
    payload = {
        "model": model.model_id,
        "messages": api_messages
    }
    
    # Cap completion tokens based on model and endpoint limits.
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    max_tokens_value = cap_openai_chat_completion_tokens(model.model_id, max_tokens_value)

    if supports_custom_temperature(model.model_id):
        payload["temperature"] = model.temperature

    if uses_openai_max_completion_tokens(model.model_id):
        payload["max_completion_tokens"] = max_tokens_value
    else:
        payload["max_tokens"] = max_tokens_value

    # reasoning_effort is not supported with function tools on /v1/chat/completions.
    # Keep it only when tools are absent (or when using the Responses API path elsewhere).
    agent_tools = model.agent_tools if agent_tools is None else agent_tools

    if getattr(model, "reasoning_effort", None) and not agent_tools:
        payload["reasoning_effort"] = model.reasoning_effort

    if agent_tools:
        tool_payload = build_provider_tool_payload("openai", agent_tools)
        payload["tools"] = tool_payload.tools
        
        forced_tool_name = getattr(model, "forced_tool_name", None)
        if forced_tool_name and forced_tool_name in tool_payload.names:
            payload["tool_choice"] = {"type": "function", "function": {"name": forced_tool_name}}
        else:
            if len(tool_payload.required_names) == 1:
                payload["tool_choice"] = {"type": "function", "function": {"name": tool_payload.required_names[0]}}
            elif len(tool_payload.required_names) > 1:
                payload["tool_choice"] = "required"
            else:
                payload["tool_choice"] = "auto"
        
        # Enable parallel tool calls if the model supports it
        if hasattr(model, 'parallel_tool_calls') and model.parallel_tool_calls:
            payload["parallel_tool_calls"] = True
        else:
            payload["parallel_tool_calls"] = False
    
    return payload
