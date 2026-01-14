import json
from typing import Any, Dict, List, Optional


def openai_fill_payload(model, messages: List[Dict[str, Any]], message_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Creates OpenAI API payload from model and messages."""
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
    
    payload = {
        "model": model.model_id,
        "messages": api_messages
    }
    
    # Newer OpenAI models (like o1/o3/gpt-4o) prefer 'max_completion_tokens' over 'max_tokens'
    # We'll use max_completion_tokens if the model ID suggests it's a newer model
    is_reasoning_model = any(x in model.model_id.lower() for x in ["o1", "o3", "gpt-5"])
    
    if not is_reasoning_model:
        # Only set temperature if it is NOT a reasoning model (or force it to 1.0 if needed, but safer to omit)
        payload["temperature"] = model.temperature
    
    if is_reasoning_model:
        payload["max_completion_tokens"] = model.max_tokens
    else:
        payload["max_tokens"] = model.max_tokens
    
    if model.agent_tools:
        tools = []
        for tool in model.agent_tools:
            if tool.args.properties:
                parameters = tool.args.properties
            else:
                arg_name = tool.args.type
                json_type = "string"
                if arg_name in ["stage_index"]:
                    json_type = "integer"
                elif arg_name == "input":
                    json_type = "string"
                
                parameters = {
                    "type": "object",
                    "properties": {
                        arg_name: {
                            "type": json_type,
                            "description": tool.args.description
                        }
                    },
                    "required": [arg_name] if tool.required else []
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
