import json
from typing import Any, Dict, List

try:
    from .base import MESSAGE_HISTORY
except ImportError:
    import sys
    from pathlib import Path
    base_path = Path(__file__).parent / "base.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("base", base_path)
    base = importlib.util.module_from_spec(spec)
    sys.modules["base"] = base
    spec.loader.exec_module(base)
    MESSAGE_HISTORY = base.MESSAGE_HISTORY


def gemini_fill_payload(model, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Creates Google Gemini API payload from model and messages."""
    contents = []
    
    system_instruction = MESSAGE_HISTORY["system"]["message"] or model.system_prompt
    
    if MESSAGE_HISTORY["first_input"]["message"]:
        contents.append({
            "role": "user",
            "parts": [{"text": MESSAGE_HISTORY["first_input"]["message"]}]
        })
    
    if MESSAGE_HISTORY["summary"]["message"]:
        contents.append({
            "role": "model",
            "parts": [{"text": MESSAGE_HISTORY["summary"]["message"]}]
        })
    
    for msg_id in sorted(MESSAGE_HISTORY["messages"].keys()):
        msg = MESSAGE_HISTORY["messages"][msg_id]
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
            tools.append({
                "functionDeclarations": [{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            tool.args.type: {
                                "type": tool.args.type,
                                "description": tool.args.description
                            }
                        },
                        "required": [tool.args.type] if tool.required else []
                    }
                }]
            })
        payload["tools"] = tools
    
    return payload
