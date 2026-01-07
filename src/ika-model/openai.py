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
        "messages": api_messages,
        "temperature": model.temperature,
        "max_tokens": model.max_tokens
    }
    
    if model.agent_tools:
        tools = []
        for tool in model.agent_tools:
            arg_name = tool.args.type
            json_type = "string"
            if arg_name in ["stage_index"]:
                json_type = "integer"
            elif arg_name == "input":
                json_type = "string"
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            arg_name: {
                                "type": json_type,
                                "description": tool.args.description
                            }
                        },
                        "required": [arg_name] if tool.required else []
                    }
                }
            })
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    return payload
