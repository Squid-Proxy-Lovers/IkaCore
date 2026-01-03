import json
from typing import Any, Dict, List, Optional


def gemini_fill_payload(model, messages: List[Dict[str, Any]], message_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Creates Google Gemini API payload from model and messages."""
    message_history = message_history or {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {}
    }
    contents = []
    
    system_instruction = message_history["system"]["message"] or model.system_prompt
    
    if message_history["first_input"]["message"]:
        contents.append({
            "role": "user",
            "parts": [{"text": message_history["first_input"]["message"]}]
        })
    
    if message_history["summary"]["message"]:
        contents.append({
            "role": "model",
            "parts": [{"text": message_history["summary"]["message"]}]
        })
    
    for msg_id in sorted(message_history["messages"].keys()):
        msg = message_history["messages"][msg_id]
        contents.append({
            "role": "model",
            "parts": [{"text": msg["message"]}]
        })
    
    for msg in messages:
        if isinstance(msg, dict):
            if "content" in msg:
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg["content"]}]
                })
            else:
                contents.append({
                    "role": "user",
                    "parts": [{"text": str(msg)}]
                })
        else:
            contents.append({
                "role": "user",
                "parts": [{"text": str(msg)}]
            })
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": model.temperature,
            "maxOutputTokens": model.max_tokens if model.max_tokens and model.max_tokens > 0 else 100
        }
    }
    
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
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
                "functionDeclarations": [{
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
                }]
            })
        payload["tools"] = tools
    
    return payload
