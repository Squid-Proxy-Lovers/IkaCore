import json
from typing import Any, Dict, List, Optional

from ..tool_schema import build_provider_tool_payload


def deepseek_fill_payload(
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

    for msg_id in message_history["messages"]:
        msg = message_history["messages"][msg_id]
        msg_type = msg.get("type", "assistant")
        raw = msg.get("message", "")
        if msg_type == "assistant_with_tools":
            try:
                api_messages.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                continue
        elif msg_type == "tool":
            try:
                api_messages.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            assistant_entry = {
                "role": "assistant",
                "content": raw if isinstance(raw, str) else str(raw)
            }
            reasoning_text = msg.get("reasoning_content")
            if reasoning_text:
                assistant_entry["reasoning_content"] = reasoning_text
            api_messages.append(assistant_entry)

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
    
    # DeepSeek V4 supports up to 32K output tokens (vs 8K on the legacy
    # deepseek-chat / deepseek-reasoner models). The old 8192 cap was causing
    # verbose non-thinking-mode replies to truncate mid-response, after which
    # the model would restate the same analysis on the next turn and truncate
    # again — a soft loop that never reached submit_triage_decision /
    # agent_end. 32K gives the model room to finish and emit the closing
    # tool call in a single turn.
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    if max_tokens_value > 32768:
        max_tokens_value = 32768
    
    payload = {
        "model": model.model_id,
        "messages": api_messages,
        "temperature": model.temperature,
        "max_tokens": max_tokens_value,
        "stream": False
    }
    
    model_id_lower = (model.model_id or "").lower()
    thinking_enabled = model.deepthinking or "reasoner" in model_id_lower
    if thinking_enabled:
        payload["thinking"] = {"type": "enabled"}
    else:
        payload["thinking"] = {"type": "disabled"}
    
    agent_tools = model.agent_tools if agent_tools is None else agent_tools

    if agent_tools:
        payload["tools"] = build_provider_tool_payload("deepseek", agent_tools).tools
        
        # Do not send tool_choice to DeepSeek. The OpenAI-compatible API
        # defaults to auto tool choice when tools are present, and some
        # DeepSeek V4 routes reject even tool_choice="auto" with a legacy
        # "deepseek-reasoner does not support this tool_choice" error.
        # Termination is enforced by the agent loop, not by provider-level
        # forced tool choice.
    
    return payload
