import json
from typing import Any, Dict, List, Optional

from ..model_metadata import is_anthropic_haiku_model, supports_anthropic_parallel_tool_use
from ..tool_schema import build_provider_tool_payload


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


def anthropic_fill_payload(
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
    system_prompt = message_history["system"]["message"] or model.system_prompt or ""
    
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
                api_messages.append(_ensure_anthropic_assistant_content(json.loads(raw)))
            except (json.JSONDecodeError, TypeError):
                continue
        elif msg_type == "tool":
            try:
                api_messages.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            api_messages.append({
                "role": "assistant",
                "content": raw if isinstance(raw, str) else str(raw)
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
                if msg["role"] == "user":
                    api_messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })
                elif msg["role"] == "assistant":
                    api_messages.append(_ensure_anthropic_assistant_content({
                        "role": "assistant",
                        "content": msg["content"]
                    }))
            elif "content" in msg:
                api_messages.append({"role": "user", "content": msg["content"]})
            else:
                api_messages.append({"role": "user", "content": str(msg)})
        else:
            api_messages.append({"role": "user", "content": str(msg)})
    
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    
    if is_anthropic_haiku_model(model.model_id):
        if max_tokens_value > 4096:
            max_tokens_value = 4096
    
    payload = {
        "model": model.model_id,
        "max_tokens": max_tokens_value,
        "temperature": model.temperature,
        "system": system_prompt,
        "messages": api_messages
    }
    
    # Add parallel tool use prompt for Claude 4 models if enabled
    if hasattr(model, 'parallel_tool_calls') and model.parallel_tool_calls:
        if supports_anthropic_parallel_tool_use(model.model_id):
            parallel_prompt = "\n\n<use_parallel_tool_calls>\nFor maximum efficiency, whenever you perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially. Prioritize calling tools in parallel whenever possible. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. When running multiple read-only commands like `ls` or `list_dir`, always run all of the commands in parallel. Err on the side of maximizing parallel tool calls rather than running too many tools sequentially.\n</use_parallel_tool_calls>"
            current_system = payload.get("system", "")
            if parallel_prompt not in current_system:
                payload["system"] = current_system + parallel_prompt
    
    if system_prompt:
        payload["system"] = system_prompt
    
    agent_tools = model.agent_tools if agent_tools is None else agent_tools

    if agent_tools:
        tool_payload = build_provider_tool_payload("anthropic", agent_tools)
        payload["tools"] = tool_payload.tools

        # Set tool_choice with optional parallel tool use control
        # disable_parallel_tool_use must be inside tool_choice, not at top level
        forced_tool_name = getattr(model, "forced_tool_name", None)
        if forced_tool_name and forced_tool_name in tool_payload.names:
            tool_choice = {"type": "tool", "name": forced_tool_name}
        else:
            if len(tool_payload.required_names) == 1:
                tool_choice = {"type": "tool", "name": tool_payload.required_names[0]}
            elif len(tool_payload.required_names) > 1:
                tool_choice = {"type": "required"}
            else:
                tool_choice = {"type": "auto"}

        # Disable parallel tool use if the model doesn't support it
        # Only available for Claude 4 models (opus-4, sonnet-4)
        # By default Claude allows parallel, so we disable it if parallel_tool_calls is False
        if hasattr(model, 'parallel_tool_calls') and not model.parallel_tool_calls:
            if supports_anthropic_parallel_tool_use(model.model_id):
                if isinstance(tool_choice, dict) and tool_choice.get("type") in ["auto", "required"]:
                    tool_choice["disable_parallel_tool_use"] = True

        payload["tool_choice"] = tool_choice
    
    return payload
