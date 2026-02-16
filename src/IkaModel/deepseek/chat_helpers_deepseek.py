import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .deepseek import deepseek_fill_payload
from ..request_interface import _apply_tools_filter_for_payload, _restore_tools_after_payload


def build_deepseek_request(
    barebone_model: Any,
    messages: List[dict],
    message_history: dict
) -> Tuple[str, Dict[str, str], dict]:
    _apply_tools_filter_for_payload(barebone_model)
    payload = deepseek_fill_payload(barebone_model, messages, message_history)
    _restore_tools_after_payload(barebone_model)
    
    api_url = barebone_model.api_url
    headers = {
        "Authorization": f"Bearer {barebone_model.api_key}",
        "Content-Type": "application/json"
    }
    
    return api_url, headers, payload


def parse_deepseek_response(data: dict, model_id: str) -> Tuple[str, Optional[str], List[dict], int]:
    message_obj = data["choices"][0]["message"]
    content = message_obj.get("content") or ""
    tool_calls = message_obj.get("tool_calls", []) or []
    tokens = data.get("usage", {}).get("total_tokens", 0)
    reasoning_content = message_obj.get("reasoning_content")
    
    return content, reasoning_content, tool_calls, tokens


def append_deepseek_tool_messages(
    messages: List[dict],
    message_history: dict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: List[dict],
    tool_messages: List[dict],
    tokens: int,
    repeated_warning_msg: str = ""
) -> None:
    assistant_msg = {"role": "assistant", "content": content}
    if reasoning_content:
        assistant_msg["reasoning_content"] = reasoning_content
    if executed_tool_call_list:
        assistant_msg["tool_calls"] = executed_tool_call_list
    
    messages.append(assistant_msg)
    messages.extend(tool_messages)
    
    if repeated_warning_msg:
        messages.append({"role": "user", "content": repeated_warning_msg})
    
    if executed_tool_call_list:
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {  # type: ignore
            "message": json.dumps(assistant_msg),
            "tokens": tokens,
            "type": "assistant_with_tools",
        }
    
    for tool_msg in tool_messages:
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {  # type: ignore
            "message": json.dumps(tool_msg),
            "tokens": 0,
            "type": "tool",
        }
