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
    
    for msg_id in sorted(message_history["messages"].keys()):
        msg = message_history["messages"][msg_id]
        api_messages.append({
            "role": "assistant",
            "content": msg["message"]
        })
    
    for msg in messages:
        if isinstance(msg, dict):
            if "role" in msg and "content" in msg:
                api_messages.append(msg)
            elif "content" in msg:
                api_messages.append({"role": "user", "content": msg["content"]})
            else:
                api_messages.append({"role": "user", "content": str(msg)})
        else:
            api_messages.append({"role": "user", "content": str(msg)})
    
    # DeepSeek has a max_tokens limit of 8192
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    if max_tokens_value > 8192:
        max_tokens_value = 8192
    
    payload = {
        "model": model.model_id,
        "messages": api_messages,
        "temperature": model.temperature,
        "max_tokens": max_tokens_value,
        "stream": False
    }
    
    if model.deepthinking:
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
                        # Properties dict without type, wrap it properly
                        required_list = tool.args.properties.pop("__required__", [])
                        parameters = {
                            "type": "object",
                            "properties": tool.args.properties,
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
                elif arg_name == "object":
                    # If type is "object", create an empty properties schema
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                else:
                    required_list = [arg_name] if tool.required else []
                    parameters = {
                        "type": "object",
                        "properties": {
                            arg_name: {
                                "type": json_type,
                                "description": tool.args.description or f"Parameter for {tool.name}"
                            }
                        },
                        "required": required_list
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
        payload["tool_choice"] = "auto"
    
    return payload
