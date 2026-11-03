# pyright: strict
# pyright: reportUnusedFunction=false

import json
import logging
import uuid
from typing import Any, Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value

from ..base import BareBoneModel
from ..request_interface import agent_tools_for_payload
from .google import gemini_fill_payload

LOG = logging.getLogger(__name__)
ProviderRequest = tuple[str, dict[str, str], JsonDict]
ProviderRound = tuple[str, Optional[str], list[JsonDict], int]
MessageList = list[JsonDict]
GeminiResultFormatter = Any


def build_gemini_request(
    barebone_model: BareBoneModel,
    messages: MessageList,
    message_history: JsonDict
) -> ProviderRequest:
    payload = gemini_fill_payload(
        barebone_model,
        messages,
        message_history,
        agent_tools=agent_tools_for_payload(barebone_model),
    )
    
    api_url = barebone_model.api_url
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": barebone_model.api_key,
    }
    
    return api_url, headers, payload


def parse_gemini_response(data: JsonDict, model_id: str) -> ProviderRound:
    raw_candidates: object = cast(object, data.get("candidates"))
    candidates = cast(list[object], raw_candidates) if isinstance(raw_candidates, list) else []
    candidate = json_dict(candidates[0]) if candidates else {}
    candidate_content = json_dict(candidate.get("content"))
    raw_parts: object = candidate_content.get("parts", [])
    parts = cast(list[object], raw_parts) if isinstance(raw_parts, list) else []
    
    content = ""
    tool_calls: list[JsonDict] = []
    
    for part in parts:
        part_data = json_dict(part)
        if "text" in part_data:
            content += string_value(part_data.get("text"))
        elif "functionCall" in part_data:
            LOG.info(f"Gemini function call part: {json.dumps(part_data)}")
            func_call = json_dict(part_data.get("functionCall"))
            tool_call_dict: JsonDict = {
                "name": func_call.get("name"),
                "gemini_raw": func_call,
                "function": {
                    "name": func_call.get("name"),
                    "arguments": json.dumps(func_call.get("args", {}))
                }
            }
            if "thoughtSignature" in part_data:
                tool_call_dict["gemini_thought_signature"] = part_data.get("thoughtSignature")
                tool_call_dict["gemini_thought_signature_key"] = "thoughtSignature"
            elif "thought_signature" in part_data:
                tool_call_dict["gemini_thought_signature"] = part_data.get("thought_signature")
                tool_call_dict["gemini_thought_signature_key"] = "thought_signature"
            tool_calls.append(tool_call_dict)
    
    tokens = 0
    reasoning_content = None
    
    return content, reasoning_content, tool_calls, tokens


def _decode_tool_args_object(raw_args: Any) -> JsonDict:
    try:
        parsed = json.loads(raw_args or "{}") if isinstance(raw_args, str) else raw_args
    except (json.JSONDecodeError, TypeError):
        return {}
    return cast(JsonDict, parsed) if isinstance(parsed, dict) else {}


def append_gemini_tool_messages(
    messages: MessageList,
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: list[JsonDict],
    tool_results: list[str],
    tokens: int,
    repeated_warning_msg: str = "",
    format_gemini_results_fn: Any = None
) -> None:
    assistant_msg: JsonDict = {"role": "model", "parts": [{"text": content}]}
    parts_list = cast(list[JsonDict], assistant_msg["parts"])
    
    for tool_call in executed_tool_call_list:
        part_dict: JsonDict = {}
        if "gemini_raw" in tool_call:
            part_dict["functionCall"] = tool_call["gemini_raw"]
        else:
            function_payload = json_dict(tool_call.get("function"))
            part_dict["functionCall"] = {
                "name": tool_call.get("name") or function_payload.get("name", ""),
                "args": _decode_tool_args_object(function_payload.get("arguments", "{}")),
            }
        if "gemini_thought_signature" in tool_call:
            sig_key = string_value(tool_call.get("gemini_thought_signature_key") or "thoughtSignature")
            part_dict[sig_key] = tool_call["gemini_thought_signature"]
        parts_list.append(part_dict)

    messages.append(assistant_msg)
    tool_response_msg: Optional[JsonDict] = None

    if format_gemini_results_fn:
        function_responses = format_gemini_results_fn(executed_tool_call_list, tool_results)
        tool_response_msg = {"role": "user", "parts": function_responses}
        messages.append(tool_response_msg)
    
    if repeated_warning_msg:
        messages.append({"role": "user", "parts": [{"text": repeated_warning_msg}]})
    
    msg_id = str(uuid.uuid4())
    history_section(message_history, "messages")[msg_id] = {
        "message": json.dumps(assistant_msg),
        "tokens": tokens,
        "type": "assistant_with_tools",
    }
    
    if tool_response_msg is not None:
        msg_id = str(uuid.uuid4())
        history_section(message_history, "messages")[msg_id] = {
            "message": json.dumps(tool_response_msg),
            "tokens": 0,
            "type": "tool",
        }
