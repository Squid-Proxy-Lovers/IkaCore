import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..model_metadata import is_anthropic_haiku_model
from ..request_interface import agent_tools_for_payload
from .claude import anthropic_fill_payload


def build_anthropic_request(
    barebone_model: Any,
    messages: List[dict],
    message_history: dict
) -> Tuple[str, Dict[str, str], dict]:
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


def parse_anthropic_response(data: dict, model_id: str) -> Tuple[str, Optional[str], List[dict], int]:
    content_blocks = data.get("content", [])
    content = "".join([block["text"] for block in content_blocks if block.get("type") == "text"])
    
    tool_calls = []
    for block in content_blocks:
        if block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "name": block.get("name"),
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input", {}))
                }
            })
    
    tokens = 0
    reasoning_content = None
    
    return content, reasoning_content, tool_calls, tokens


def append_anthropic_tool_messages(
    messages: List[dict],
    message_history: dict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: List[dict],
    tool_messages: List[dict],
    tokens: int,
    repeated_warning_msg: str = ""
) -> None:
    content_blocks: List[dict] = []
    if content:
        content_blocks.append({"type": "text", "text": content})
    else:
        content_blocks.append({"type": "text", "text": " "})
    for tool_call in executed_tool_call_list:
        content_blocks.append({
            "type": "tool_use",
            "id": tool_call.get("id"),
            "name": tool_call.get("name"),
            "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        })
    assistant_msg = {"role": "assistant", "content": content_blocks}
    messages.append(assistant_msg)
    messages.extend(tool_messages)
    
    if repeated_warning_msg:
        messages.append({"role": "user", "content": [{"type": "text", "text": repeated_warning_msg}]})
    
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
