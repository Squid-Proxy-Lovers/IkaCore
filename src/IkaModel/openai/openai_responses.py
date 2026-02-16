import json
from typing import Any, Dict, List, Optional


def _build_parameters(tool) -> dict:
    """Reuse the parameter-building logic from openai.py but standalone."""
    parameters = None
    if tool.name == "agent_end" and getattr(tool.args, "type", "") == "input":
        parameters = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": getattr(tool.args, "description", None) or "Final response content. This is REQUIRED - provide your complete final answer here."
                }
            },
            "required": ["input"]
        }
    elif getattr(tool.args, "type", "") == "input":
        parameters = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": getattr(tool.args, "description", None) or f"Input for {tool.name}"
                }
            },
            "required": ["input"]
        }
    elif tool.args.properties is not None and isinstance(tool.args.properties, dict):
        if len(tool.args.properties) > 0:
            if "type" in tool.args.properties and tool.args.properties["type"] == "object":
                parameters = {
                    "type": "object",
                    "properties": tool.args.properties.get("properties", {}),
                    "required": tool.args.properties.get("required", [])
                }
            elif "type" not in tool.args.properties:
                required_list = list(tool.args.properties.get("__required__", []))
                props = {k: v for k, v in tool.args.properties.items() if k != "__required__" and isinstance(v, dict)}
                parameters = {
                    "type": "object",
                    "properties": props,
                    "required": required_list
                }
            else:
                parameters = tool.args.properties
        else:
            parameters = {"type": "object", "properties": {}, "required": []}
    else:
        arg_name = getattr(tool.args, "type", "string")
        json_type = "string"
        if arg_name in ["stage_index"]:
            json_type = "integer"
            parameters = {
                "type": "object",
                "properties": {
                    arg_name: {
                        "type": json_type,
                        "description": getattr(tool.args, "description", None) or f"Parameter for {tool.name}"
                    }
                },
                "required": []
            }
        elif arg_name == "input":
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": getattr(tool.args, "description", None) or "Final response content."
                    }
                },
                "required": ["input"]
            }
        elif arg_name == "object":
            parameters = {"type": "object", "properties": {}, "required": []}
        else:
            parameters = {
                "type": "object",
                "properties": {
                    arg_name: {
                        "type": json_type,
                        "description": getattr(tool.args, "description", None) or f"Parameter for {tool.name}"
                    }
                },
                "required": []
            }

    if parameters is None or parameters.get("type") != "object":
        parameters = {
            "type": "object",
            "properties": parameters.get("properties", {}) if isinstance(parameters, dict) else {},
            "required": parameters.get("required", []) if isinstance(parameters, dict) else []
        }
    return parameters


def openai_responses_fill_payload(
    model,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None
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

    # Temperature (not supported for reasoning models)
    is_reasoning_model = any(x in model.model_id.lower() for x in ["o1", "o3", "o4"])
    is_gpt5_family = "gpt-5" in model.model_id.lower()
    if not is_reasoning_model and not is_gpt5_family:
        payload["temperature"] = model.temperature

    # reasoning_effort
    if getattr(model, "reasoning_effort", None):
        payload["reasoning_effort"] = model.reasoning_effort

    # Tools — flatter schema: no nested "function" key
    if model.agent_tools:
        tools = []
        for tool in model.agent_tools:
            parameters = _build_parameters(tool)
            tools.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters
            })
        payload["tools"] = tools

        required_tools = [t for t in model.agent_tools if t.required]
        if len(required_tools) == 1:
            payload["tool_choice"] = {"type": "function", "name": required_tools[0].name}
        elif len(required_tools) > 1:
            payload["tool_choice"] = "required"
        else:
            payload["tool_choice"] = "auto"

        if hasattr(model, "parallel_tool_calls") and model.parallel_tool_calls:
            payload["parallel_tool_calls"] = True
        else:
            payload["parallel_tool_calls"] = False

    return payload
