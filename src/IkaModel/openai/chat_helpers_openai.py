import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .openai import openai_fill_payload
from .openai_responses import openai_responses_fill_payload
from ..request_interface import _apply_tools_filter_for_payload, _restore_tools_after_payload


def build_openai_request(
    barebone_model: Any,
    messages: List[dict],
    message_history: dict
) -> Tuple[str, Dict[str, str], dict]:
    _apply_tools_filter_for_payload(barebone_model)
    payload = openai_fill_payload(barebone_model, messages, message_history)
    _restore_tools_after_payload(barebone_model)
    
    api_url = barebone_model.api_url
    headers = {
        "Authorization": f"Bearer {barebone_model.api_key}",
        "Content-Type": "application/json"
    }
    
    return api_url, headers, payload


def parse_openai_response(data: dict, model_id: str) -> Tuple[str, Optional[str], List[dict], int]:
    message_obj = data["choices"][0]["message"]
    content = message_obj.get("content") or ""
    tool_calls = message_obj.get("tool_calls", []) or []
    tokens = data.get("usage", {}).get("total_tokens", 0)
    reasoning_content = None
    
    return content, reasoning_content, tool_calls, tokens


def append_openai_tool_messages(
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


# ---------------------------------------------------------------------------
# Responses API helpers
# ---------------------------------------------------------------------------

def build_openai_responses_request(
    barebone_model: Any,
    messages: List[dict],
    message_history: dict
) -> Tuple[str, Dict[str, str], dict]:
    """Build a request for the OpenAI Responses API endpoint."""
    _apply_tools_filter_for_payload(barebone_model)
    payload = openai_responses_fill_payload(barebone_model, messages, message_history)
    _restore_tools_after_payload(barebone_model)

    # Derive the responses endpoint from the model's api_url.
    # If the caller already set /v1/responses we keep it; otherwise we substitute.
    api_url = barebone_model.api_url
    if not api_url.rstrip("/").endswith("/v1/responses"):
        # Replace /v1/chat/completions (or any trailing path) with /v1/responses
        base = api_url.split("/v1/")[0] if "/v1/" in api_url else api_url.rstrip("/")
        api_url = f"{base}/v1/responses"

    headers = {
        "Authorization": f"Bearer {barebone_model.api_key}",
        "Content-Type": "application/json"
    }

    return api_url, headers, payload


def parse_openai_responses_response(
    data: dict,
    model_id: str
) -> Tuple[str, Optional[str], List[dict], int]:
    """Parse an OpenAI Responses API response into the internal format."""
    content = ""
    tool_calls: List[dict] = []
    reasoning_content = None

    output = data.get("output", []) or []
    for item in output:
        item_type = item.get("type", "")
        if item_type == "message":
            # Extract text from content blocks
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    content += block.get("text", "")
        elif item_type == "function_call":
            # Normalize to internal tool-call format:
            # {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
            tool_calls.append({
                "id": item.get("call_id", item.get("id", "")),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}")
                }
            })
        elif item_type == "reasoning":
            # Some reasoning models surface reasoning text
            reasoning_content = item.get("summary", [{}])[0].get("text", "") if item.get("summary") else None

    # Also check the convenience output_text field
    if not content and data.get("output_text"):
        content = data["output_text"]

    usage = data.get("usage", {}) or {}
    tokens = usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0))

    return content, reasoning_content, tool_calls, tokens


def append_openai_responses_tool_messages(
    messages: List[dict],
    message_history: dict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: List[dict],
    tool_messages: List[dict],
    tokens: int,
    repeated_warning_msg: str = ""
) -> None:
    """Append assistant + tool-result messages for a subsequent Responses API call.

    Tool results in the Responses API use the `function_call_output` format.
    The assistant's function_call items must appear *before* the outputs in the
    conversation flow, which is handled when the payload builder reads `messages`.
    We store them in the same flat `messages` list used by every other provider,
    but with Responses-API-native dicts so the payload builder can forward them.
    """
    # Store the assistant turn with its tool_calls (Chat-Completions-compatible
    # dict — the payload builder in openai_responses.py converts them).
    assistant_msg: dict = {"role": "assistant", "content": content}
    if reasoning_content:
        assistant_msg["reasoning_content"] = reasoning_content
    if executed_tool_call_list:
        assistant_msg["tool_calls"] = executed_tool_call_list

    messages.append(assistant_msg)

    # Append tool result messages.  The payload builder converts role=="tool"
    # entries to {"type": "function_call_output", ...} automatically.
    messages.extend(tool_messages)

    if repeated_warning_msg:
        messages.append({"role": "user", "content": repeated_warning_msg})

    # Record in message_history for summarization / context tracking
    if executed_tool_call_list:
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {
            "message": json.dumps(assistant_msg),
            "tokens": tokens,
            "type": "assistant_with_tools",
        }

    for tool_msg in tool_messages:
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {
            "message": json.dumps(tool_msg),
            "tokens": 0,
            "type": "tool",
        }
