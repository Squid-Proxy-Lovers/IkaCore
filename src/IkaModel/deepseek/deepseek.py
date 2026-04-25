import json
from typing import Any, Dict, List, Optional


def deepseek_fill_payload(model, messages: List[Dict[str, Any]], message_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    
    if model.agent_tools:
        tools = []
        for tool in model.agent_tools:
            # Ensure we always have a valid JSON schema
            # tool.args.properties can be None, empty dict {}, or a dict with properties
            # For agent_end, always use type-based conversion to ensure correct parameters
            if tool.name == "agent_end" and tool.args.type == "input":
                parameters = {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": tool.args.description or "Final response content. This is REQUIRED - provide your complete final answer here."
                        }
                    },
                    "required": ["input"]
                }
            elif tool.args.type == "input":
                parameters = {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": tool.args.description or f"Input for {tool.name}"
                        }
                    },
                    "required": ["input"]
                }
            elif tool.args.properties is not None and isinstance(tool.args.properties, dict):
                if len(tool.args.properties) > 0:
                    # Non-empty properties dict - use it but ensure proper structure
                    if "type" in tool.args.properties and tool.args.properties["type"] == "object":
                        # Already has type: object, use as-is but ensure properties key exists
                        required_list = tool.args.properties.get("required", [])
                        parameters = {
                            "type": "object",
                            "properties": tool.args.properties.get("properties", {}),
                            "required": required_list
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
                        # Has type but might not be object, use as-is
                        parameters = tool.args.properties
                else:
                    # Empty properties dict {} - create valid empty schema
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
            else:
                # No properties or properties is None - create schema from tool.args.type
                arg_name = tool.args.type
                json_type = "string"
                if arg_name in ["stage_index"]:
                    json_type = "integer"
                elif arg_name == "input":
                    json_type = "string"

                if arg_name == "object":
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                else:
                    parameters = {
                        "type": "object",
                        "properties": {
                            arg_name: {
                                "type": json_type,
                                "description": tool.args.description or f"Parameter for {tool.name}"
                            }
                        },
                        "required": []
                    }

            # Ensure parameters is never None and always has type: object
            if parameters is None or parameters.get("type") != "object":
                parameters = {
                    "type": "object",
                    "properties": parameters.get("properties", {}) if isinstance(parameters, dict) else {},
                    "required": parameters.get("required", []) if isinstance(parameters, dict) else []
                }

            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters
                }
            })
        payload["tools"] = tools
        
        model_id_lower = (model.model_id or "").lower()
        is_reasoner = "reasoner" in model_id_lower

        # NOTE: AgentTool.required means "this tool is expected to be used at
        # some point during the run" — NOT "every API turn must emit a tool
        # call". Mapping it to OpenAI tool_choice="required" or to a forced
        # function pin traps the model in an infinite tool-call loop on
        # providers that strictly honor tool_choice (e.g. DeepSeek V4), because
        # the model can never emit a natural-language finish. Termination is
        # already enforced by the agent loop (agent_end + max_tool_calls +
        # repeat-call guard), so always let the provider choose freely.
        if not is_reasoner:
            payload["tool_choice"] = "auto"
    
    return payload
