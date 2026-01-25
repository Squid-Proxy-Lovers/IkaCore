import json
from typing import Any, Dict, List, Optional


def openai_fill_payload(model, messages: List[Dict[str, Any]], message_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            api_messages.append({
                "role": "assistant",
                "content": raw if isinstance(raw, str) else str(raw)
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
    
    payload = {
        "model": model.model_id,
        "messages": api_messages
    }
    
    # Newer OpenAI models (like o1/o3/gpt-4o) prefer 'max_completion_tokens' over 'max_tokens'
    # We'll use max_completion_tokens if the model ID suggests it's a newer model
    is_reasoning_model = any(x in model.model_id.lower() for x in ["o1", "o3", "gpt-5"])
    
    # Cap max_tokens based on model limits
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    model_id_lower = model.model_id.lower()
    
    if "gpt-4o-mini" in model_id_lower:
        # gpt-4o-mini supports at most 16384 completion tokens
        if max_tokens_value > 16384:
            max_tokens_value = 16384
    elif "gpt-4o" in model_id_lower and "mini" not in model_id_lower:
        # gpt-4o supports up to 16384
        if max_tokens_value > 16384:
            max_tokens_value = 16384
    elif "gpt-3.5" in model_id_lower:
        # gpt-3.5-turbo supports up to 4096
        if max_tokens_value > 4096:
            max_tokens_value = 4096
    elif "gpt-4" in model_id_lower and "turbo" in model_id_lower:
        # gpt-4-turbo supports up to 4096
        if max_tokens_value > 4096:
            max_tokens_value = 4096
    
    if not is_reasoning_model:
        # Only set temperature if it is NOT a reasoning model (or force it to 1.0 if needed, but safer to omit)
        payload["temperature"] = model.temperature
    
    if is_reasoning_model:
        payload["max_completion_tokens"] = max_tokens_value
    else:
        payload["max_tokens"] = max_tokens_value
    
    if model.agent_tools:
        tools = []
        for tool in model.agent_tools:
            parameters = None
            # Ensure we always have a valid JSON schema
            # tool.args.properties can be None, empty dict {}, or a dict with properties
            # For agent_end and subagent tools with type="input", always use input parameter
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
                        parameters = {
                            "type": "object",
                            "properties": tool.args.properties.get("properties", {}),
                            "required": tool.args.properties.get("required", [])
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
                elif arg_name == "input":
                    parameters = {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": tool.args.description or "Final response content."
                            }
                        },
                        "required": ["input"]
                    }
                elif arg_name == "object":
                    # If type is "object", create an empty properties schema
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
        
        required_tools = [t for t in model.agent_tools if t.required]
        if len(required_tools) == 1:
            payload["tool_choice"] = {"type": "function", "function": {"name": required_tools[0].name}}
        elif len(required_tools) > 1:
            payload["tool_choice"] = "required"
        else:
            payload["tool_choice"] = "auto"
        
        # Enable parallel tool calls if the model supports it
        if hasattr(model, 'parallel_tool_calls') and model.parallel_tool_calls:
            payload["parallel_tool_calls"] = True
        else:
            payload["parallel_tool_calls"] = False
    
    return payload
