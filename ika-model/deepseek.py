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


def deepseek_fill_payload(model, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Creates DeepSeek API payload from model and messages."""
    api_messages = []
    
    if MESSAGE_HISTORY["system"]["message"]:
        api_messages.append({
            "role": "system",
            "content": MESSAGE_HISTORY["system"]["message"]
        })
    
    if MESSAGE_HISTORY["first_input"]["message"]:
        api_messages.append({
            "role": "user",
            "content": MESSAGE_HISTORY["first_input"]["message"]
        })
    
    if MESSAGE_HISTORY["summary"]["message"]:
        api_messages.append({
            "role": "assistant",
            "content": MESSAGE_HISTORY["summary"]["message"]
        })
    
    for msg_id in sorted(MESSAGE_HISTORY["messages"].keys()):
        msg = MESSAGE_HISTORY["messages"][msg_id]
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
        "max_tokens": model.max_tokens,
        "stream": False
    }
    
    if model.deepthinking:
        payload["thinking"] = {"type": "enabled"}
    else:
        payload["thinking"] = {"type": "disabled"}
    
    if model.agent_tools:
        tools = []
        for tool in model.agent_tools:
            tools.append({
                "type": "function",
                "function": {
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
                }
            })
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    return payload
