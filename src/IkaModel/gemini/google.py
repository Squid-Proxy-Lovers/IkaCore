import json
from typing import Any, Dict, List, Optional

from ..tool_schema import build_provider_tool_payload


def _default_message_history() -> Dict[str, Any]:
    return {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def _gemini_text_part(text: Any) -> Dict[str, Any]:
    return {"text": text if isinstance(text, str) else str(text)}


def _append_gemini_context(contents: List[Dict[str, Any]], message_history: Dict[str, Any]) -> None:
    if message_history["first_input"]["message"]:
        contents.append({"role": "user", "parts": [_gemini_text_part(message_history["first_input"]["message"])]})
    if message_history["summary"]["message"]:
        contents.append({"role": "model", "parts": [_gemini_text_part(message_history["summary"]["message"])]})


def _append_gemini_history(contents: List[Dict[str, Any]], message_history: Dict[str, Any]) -> None:
    for msg_id in message_history["messages"]:
        msg = message_history["messages"][msg_id]
        msg_type = msg.get("type", "assistant")
        raw = msg.get("message", "")
        if msg_type in {"assistant_with_tools", "tool"}:
            try:
                contents.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                contents.append({"role": "model", "parts": [_gemini_text_part(raw)]})
        else:
            contents.append({"role": "model", "parts": [_gemini_text_part(raw)]})


def _gemini_parts_text(message: Dict[str, Any]) -> str:
    return " ".join(
        part.get("text", "")
        for part in message.get("parts", [])
        if isinstance(part, dict) and "text" in part
    ).strip()


def _is_duplicate_first_gemini_message(index: int, message: Any, first_input_text: str) -> bool:
    if index != 0 or not first_input_text:
        return False
    if isinstance(message, dict) and "parts" in message and message.get("role") == "user":
        return _gemini_parts_text(message).strip() == first_input_text.strip()
    text = message.get("content", str(message)) if isinstance(message, dict) else str(message)
    return isinstance(text, str) and text.strip() == first_input_text.strip()


def _append_gemini_live_messages(contents: List[Dict[str, Any]], messages: List[Dict[str, Any]], first_input_text: str) -> None:
    for index, message in enumerate(messages):
        if _is_duplicate_first_gemini_message(index, message, first_input_text):
            continue
        if isinstance(message, dict) and "parts" in message and "role" in message:
            contents.append(message)
            continue
        text = message.get("content", str(message)) if isinstance(message, dict) else str(message)
        contents.append({"role": "user", "parts": [_gemini_text_part(text)]})


def _apply_gemini_tools(payload: Dict[str, Any], model: Any, agent_tools: list[Any]) -> None:
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


def gemini_fill_payload(
    model: Any,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None,
    agent_tools: Optional[list[Any]] = None,
) -> Dict[str, Any]:
    message_history = message_history or _default_message_history()
    contents: List[Dict[str, Any]] = []

    system_instruction = message_history["system"]["message"] or model.system_prompt
    _append_gemini_context(contents, message_history)
    _append_gemini_history(contents, message_history)

    first_input_text = (message_history.get("first_input") or {}).get("message") or ""
    _append_gemini_live_messages(contents, messages, first_input_text)

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
        payload["systemInstruction"] = {"parts": [_gemini_text_part(system_instruction)]}

    agent_tools = model.agent_tools if agent_tools is None else agent_tools

    if agent_tools:
        _apply_gemini_tools(payload, model, agent_tools)

    return payload
