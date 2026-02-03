import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .google import gemini_fill_payload
from .request_interface import _apply_tools_filter_for_payload, _restore_tools_after_payload

LOG = logging.getLogger(__name__)


def build_gemini_request(
    barebone_model: Any,
    messages: List[dict],
    message_history: dict
) -> Tuple[str, Dict[str, str], dict]:
    _apply_tools_filter_for_payload(barebone_model)
    payload = gemini_fill_payload(barebone_model, messages, message_history)
    _restore_tools_after_payload(barebone_model)
    
    api_url = f"{barebone_model.api_url}?key={barebone_model.api_key}"
    headers = {"Content-Type": "application/json"}
    
    return api_url, headers, payload


def parse_gemini_response(data: dict, model_id: str) -> Tuple[str, Optional[str], List[dict], int]:
    candidate = (data.get("candidates") or [{}])[0]
    candidate_content = candidate.get("content") or {}
    parts = candidate_content.get("parts", [])
    
    content = ""
    tool_calls = []
    
    for part in parts:
        if "text" in part:
            content += part["text"]
        elif "functionCall" in part:
            LOG.info(f"Gemini function call part: {json.dumps(part)}")
            func_call = part["functionCall"]
            tool_call_dict = {
                "name": func_call.get("name"),
                "gemini_raw": func_call,
                "function": {
                    "name": func_call.get("name"),
                    "arguments": json.dumps(func_call.get("args", {}))
                }
            }
            if "thoughtSignature" in part:
                tool_call_dict["gemini_thought_signature"] = part.get("thoughtSignature")
                tool_call_dict["gemini_thought_signature_key"] = "thoughtSignature"
            elif "thought_signature" in part:
                tool_call_dict["gemini_thought_signature"] = part.get("thought_signature")
                tool_call_dict["gemini_thought_signature_key"] = "thought_signature"
            tool_calls.append(tool_call_dict)
    
    tokens = 0
    reasoning_content = None
    
    return content, reasoning_content, tool_calls, tokens


def append_gemini_tool_messages(
    messages: List[dict],
    message_history: dict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: List[dict],
    tool_results: List[str],
    tokens: int,
    repeated_warning_msg: str = "",
    format_gemini_results_fn: Any = None
) -> None:
    assistant_msg = {"role": "model", "parts": [{"text": content}]}
    
    for tool_call in executed_tool_call_list:
        part_dict: Dict[str, Any] = {}
        if "gemini_raw" in tool_call:
            part_dict["functionCall"] = tool_call["gemini_raw"]
        else:
            part_dict["functionCall"] = {
                "name": tool_call.get("name") or tool_call.get("function", {}).get("name", ""),
                "args": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            }
        if "gemini_thought_signature" in tool_call:
            sig_key = tool_call.get("gemini_thought_signature_key") or "thoughtSignature"
            part_dict[sig_key] = tool_call["gemini_thought_signature"]
        assistant_msg["parts"].append(part_dict)
    
    messages.append(assistant_msg)
    
    if format_gemini_results_fn:
        function_responses = format_gemini_results_fn(executed_tool_call_list, tool_results)
        tool_response_msg = {"role": "user", "parts": function_responses}
        messages.append(tool_response_msg)
    
    if repeated_warning_msg:
        messages.append({"role": "user", "parts": [{"text": repeated_warning_msg}]})
    
    msg_id = str(uuid.uuid4())
    message_history["messages"][msg_id] = {  # type: ignore
        "message": json.dumps(assistant_msg),
        "tokens": tokens,
        "type": "assistant_with_tools",
    }
    
    if format_gemini_results_fn:
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {  # type: ignore
            "message": json.dumps(tool_response_msg),
            "tokens": 0,
            "type": "tool",
        }
