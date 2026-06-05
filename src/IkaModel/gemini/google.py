import json
from typing import Any, Dict, List, Optional

from ..tool_schema import build_provider_tool_payload


def gemini_fill_payload(
    model,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None,
    agent_tools: Optional[list[Any]] = None,
) -> Dict[str, Any]:
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

    for msg_id in message_history["messages"]:
        msg = message_history["messages"][msg_id]
        msg_type = msg.get("type", "assistant")
        raw = msg.get("message", "")
        if msg_type == "assistant_with_tools" or msg_type == "tool":
            try:
                contents.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                contents.append({"role": "model", "parts": [{"text": raw if isinstance(raw, str) else str(raw)}]})
        else:
            contents.append({
                "role": "model",
                "parts": [{"text": raw if isinstance(raw, str) else str(raw)}]
            })

    first_input_text = (message_history.get("first_input") or {}).get("message") or ""
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            text = str(msg)
            if i == 0 and first_input_text and text.strip() == first_input_text.strip():
                continue
            contents.append({"role": "user", "parts": [{"text": text}]})
            continue
        if "parts" in msg and "role" in msg:
            if i == 0 and msg.get("role") == "user" and first_input_text:
                parts_text = " ".join(
                    p.get("text", "") for p in msg.get("parts", []) if isinstance(p, dict) and "text" in p
                ).strip()
                if parts_text.strip() == first_input_text.strip():
                    continue
            contents.append(msg)
            continue
        text = msg.get("content", str(msg))
        if i == 0 and first_input_text and text.strip() == first_input_text.strip():
            continue
        contents.append({"role": "user", "parts": [{"text": text}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": model.temperature,
            "maxOutputTokens": model.max_tokens if model.max_tokens and model.max_tokens > 0 else 100,
            "thinkingConfig": {
                "thinkingBudget": 0,
            },
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    agent_tools = model.agent_tools if agent_tools is None else agent_tools

    if agent_tools:
        tool_payload = build_provider_tool_payload("gemini", agent_tools)
        payload["tools"] = [{"functionDeclarations": tool_payload.tools}]
        forced_tool_name = getattr(model, "forced_tool_name", None)
        if forced_tool_name and forced_tool_name in tool_payload.names:
            payload["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [forced_tool_name],
                }
            }
    
    return payload
