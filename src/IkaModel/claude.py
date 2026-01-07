import json
from typing import Any, Dict, List, Optional


def anthropic_fill_payload(model, messages: List[Dict[str, Any]], message_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Creates Anthropic Claude API payload from model and messages."""
    message_history = message_history or {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {}
    }
    api_messages = []
    system_prompt = message_history["system"]["message"] or model.system_prompt
    
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
                if msg["role"] == "user":
                    api_messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })
                elif msg["role"] == "assistant":
                    api_messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
            elif "content" in msg:
                api_messages.append({"role": "user", "content": msg["content"]})
            else:
                api_messages.append({"role": "user", "content": str(msg)})
        else:
            api_messages.append({"role": "user", "content": str(msg)})
    
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    
    payload = {
        "model": model.model_id,
        "max_tokens": max_tokens_value,
        "temperature": model.temperature,
        "messages": api_messages
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
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
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        arg_name: {
                            "type": json_type,
                            "description": tool.args.description
                        }
                    },
                    "required": [arg_name] if tool.required else []
                }
            })
        payload["tools"] = tools
        payload["tool_choice"] = {"type": "auto"}
    
    return payload
