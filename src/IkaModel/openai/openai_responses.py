import json
from typing import Any, Dict, List, Optional

from ..model_metadata import supports_custom_temperature
from ..tool_schema import build_provider_tool_payload


def openai_responses_fill_payload(
    model,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None,
    agent_tools: Optional[list[Any]] = None,
) -> Dict[str, Any]:
    """Build a payload for the OpenAI Responses API (POST /v1/responses)."""
    message_history = message_history or {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {}
    }

    # System prompt goes into the top-level `instructions` field
    instructions = None
    if message_history["system"]["message"]:
        instructions = message_history["system"]["message"]

    # Build the input items array
    input_items: List[Dict[str, Any]] = []

    # Summary (previous conversation context) as an assistant message
    if message_history["summary"]["message"]:
        input_items.append({
            "type": "message",
            "role": "assistant",
            "content": message_history["summary"]["message"]
        })

    # First input (original user message)
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

    if first_input_content:
        input_items.append({
            "type": "message",
            "role": "user",
            "content": first_input_content
        })

    # Convert messages list → Responses API input items
    for i, msg in enumerate(messages):
        if skip_first and i == 0:
            continue

        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "tool":
                # Convert Chat Completions tool result → Responses API function_call_output
                tool_call_id = msg.get("tool_call_id", "")
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": content if isinstance(content, str) else json.dumps(content)
                })
            elif role == "assistant":
                # May contain tool_calls — emit function_call items
                tool_calls = msg.get("tool_calls", []) or []
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}")
                        })
                else:
                    if content:
                        input_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": content
                        })
            elif role == "user":
                # Check if content is a list (Anthropic-style tool results embedded)
                if isinstance(content, list):
                    # Convert tool_result blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            input_items.append({
                                "type": "function_call_output",
                                "call_id": block.get("tool_use_id", ""),
                                "output": block.get("content", "")
                            })
                        else:
                            input_items.append({
                                "type": "message",
                                "role": "user",
                                "content": json.dumps(block) if not isinstance(block, str) else block
                            })
                else:
                    input_items.append({
                        "type": "message",
                        "role": "user",
                        "content": content or str(msg)
                    })
            else:
                input_items.append({
                    "type": "message",
                    "role": role,
                    "content": content or str(msg)
                })
        else:
            input_items.append({
                "type": "message",
                "role": "user",
                "content": str(msg)
            })

    payload: Dict[str, Any] = {
        "model": model.model_id,
        "input": input_items,
        "store": False,  # We manage our own conversation state
    }

    if instructions:
        payload["instructions"] = instructions

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
        tool_payload = build_provider_tool_payload("openai_responses", agent_tools)
        payload["tools"] = tool_payload.tools

        forced_tool_name = getattr(model, "forced_tool_name", None)
        if forced_tool_name and forced_tool_name in tool_payload.names:
            payload["tool_choice"] = {"type": "function", "name": forced_tool_name}
        else:
            if len(tool_payload.required_names) == 1:
                payload["tool_choice"] = {"type": "function", "name": tool_payload.required_names[0]}
            elif len(tool_payload.required_names) > 1:
                payload["tool_choice"] = "required"
            else:
                payload["tool_choice"] = "auto"

        if hasattr(model, "parallel_tool_calls") and model.parallel_tool_calls:
            payload["parallel_tool_calls"] = True
        else:
            payload["parallel_tool_calls"] = False

    return payload
