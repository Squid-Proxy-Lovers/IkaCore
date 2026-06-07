# pyright: strict
# pyright: reportUnusedFunction=false

import json
import uuid
from typing import Any, Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value

from ..base import BareBoneModel
from ..model_metadata import is_anthropic_haiku_model
from ..request_interface import agent_tools_for_payload
from .claude import anthropic_fill_payload

ProviderRequest = tuple[str, dict[str, str], JsonDict]
ProviderRound = tuple[str, Optional[str], list[JsonDict], int]
MessageList = list[JsonDict]


def build_anthropic_request(
    barebone_model: BareBoneModel,
    messages: MessageList,
    message_history: JsonDict
) -> ProviderRequest:
    payload = anthropic_fill_payload(
        barebone_model,
        messages,
        message_history,
        agent_tools=agent_tools_for_payload(barebone_model),
    )
    
    if "max_tokens" not in payload or not payload["max_tokens"]:
        payload["max_tokens"] = 4096
    if is_anthropic_haiku_model(barebone_model.model_id) and payload["max_tokens"] > 4096:
        payload["max_tokens"] = 4096
    
    api_url = barebone_model.api_url
    headers = {
        "x-api-key": barebone_model.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    return api_url, headers, payload


def parse_anthropic_response(data: JsonDict, model_id: str) -> ProviderRound:
    raw_content_blocks: object = data.get("content", [])
    content_blocks = cast(list[object], raw_content_blocks) if isinstance(raw_content_blocks, list) else []
    content = "".join([
        string_value(block.get("text"))
        for block in (json_dict(item) for item in content_blocks)
        if block.get("type") == "text"
    ])
    
    tool_calls: list[JsonDict] = []
    for block in content_blocks:
        block_data = json_dict(block)
        if block_data.get("type") == "tool_use":
            tool_calls.append({
                "id": block_data.get("id"),
                "name": block_data.get("name"),
                "function": {
                    "name": block_data.get("name"),
                    "arguments": json.dumps(block_data.get("input", {}))
                }
            })
    
    tokens = 0
    reasoning_content = None
    
    return content, reasoning_content, tool_calls, tokens


def _decode_tool_args_object(raw_args: Any) -> JsonDict:
    try:
        parsed = json.loads(raw_args or "{}") if isinstance(raw_args, str) else raw_args
    except (json.JSONDecodeError, TypeError):
        return {}
    return cast(JsonDict, parsed) if isinstance(parsed, dict) else {}


def append_anthropic_tool_messages(
    messages: MessageList,
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: list[JsonDict],
    tool_messages: MessageList,
    tokens: int,
    repeated_warning_msg: str = ""
) -> None:
    content_blocks: list[JsonDict] = []
    if content:
        content_blocks.append({"type": "text", "text": content})
    else:
        content_blocks.append({"type": "text", "text": " "})
    for tool_call in executed_tool_call_list:
        function_payload = json_dict(tool_call.get("function"))
        content_blocks.append({
            "type": "tool_use",
            "id": tool_call.get("id"),
            "name": tool_call.get("name"),
            "input": _decode_tool_args_object(function_payload.get("arguments", "{}")),
        })
    assistant_msg: JsonDict = {"role": "assistant", "content": content_blocks}
    messages.append(assistant_msg)
    messages.extend(tool_messages)
    
    if repeated_warning_msg:
        messages.append({"role": "user", "content": [{"type": "text", "text": repeated_warning_msg}]})
    
    msg_id = str(uuid.uuid4())
    history_section(message_history, "messages")[msg_id] = {
        "message": json.dumps(assistant_msg),
        "tokens": tokens,
        "type": "assistant_with_tools",
    }
    
    for tool_msg in tool_messages:
        msg_id = str(uuid.uuid4())
        history_section(message_history, "messages")[msg_id] = {
            "message": json.dumps(tool_msg),
            "tokens": 0,
            "type": "tool",
        }
