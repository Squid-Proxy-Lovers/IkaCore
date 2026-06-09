# pyright: strict
# pyright: reportUnusedFunction=false

import json
import uuid
from typing import Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value

from ..base import BareBoneModel
from ..request_interface import agent_tools_for_payload
from .openai import openai_fill_payload
from .openai_responses import openai_responses_fill_payload

ProviderRequest = tuple[str, dict[str, str], JsonDict]
ProviderRound = tuple[str, Optional[str], list[JsonDict], int]
MessageList = list[JsonDict]


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def build_openai_request(
    barebone_model: BareBoneModel,
    messages: MessageList,
    message_history: JsonDict
) -> ProviderRequest:
    payload = openai_fill_payload(
        barebone_model,
        messages,
        message_history,
        agent_tools=agent_tools_for_payload(barebone_model),
    )
    
    api_url = barebone_model.api_url
    headers = {
        "Authorization": f"Bearer {barebone_model.api_key}",
        "Content-Type": "application/json"
    }
    
    return api_url, headers, payload


def parse_openai_response(data: JsonDict, model_id: str) -> ProviderRound:
    choices = cast(list[JsonDict], data["choices"])
    message_obj = json_dict(choices[0].get("message"))
    content = string_value(message_obj.get("content"))
    raw_tool_calls: object = message_obj.get("tool_calls", []) or []
    tool_calls = cast(list[JsonDict], raw_tool_calls) if isinstance(raw_tool_calls, list) else []
    tokens = _token_count(json_dict(data.get("usage")).get("total_tokens"))
    reasoning_content = None
    
    return content, reasoning_content, tool_calls, tokens


def append_openai_tool_messages(
    messages: MessageList,
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: list[JsonDict],
    tool_messages: MessageList,
    tokens: int,
    repeated_warning_msg: str = ""
) -> None:
    assistant_msg: JsonDict = {"role": "assistant", "content": content}
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


# ---------------------------------------------------------------------------
# Responses API helpers
# ---------------------------------------------------------------------------

def build_openai_responses_request(
    barebone_model: BareBoneModel,
    messages: MessageList,
    message_history: JsonDict
) -> ProviderRequest:
    """Build a request for the OpenAI Responses API endpoint."""
    payload = openai_responses_fill_payload(
        barebone_model,
        messages,
        message_history,
        agent_tools=agent_tools_for_payload(barebone_model),
    )

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
    data: JsonDict,
    model_id: str
) -> ProviderRound:
    """Parse an OpenAI Responses API response into the internal format."""
    content = ""
    tool_calls: list[JsonDict] = []
    reasoning_content = None

    raw_output: object = data.get("output", []) or []
    output = cast(list[object], raw_output) if isinstance(raw_output, list) else []
    for item in output:
        item_data = json_dict(item)
        item_type = item_data.get("type", "")
        if item_type == "message":
            # Extract text from content blocks
            raw_blocks: object = item_data.get("content", [])
            blocks = cast(list[object], raw_blocks) if isinstance(raw_blocks, list) else []
            for block in blocks:
                block_data = json_dict(block)
                if block_data.get("type") == "output_text":
                    content += string_value(block_data.get("text"))
        elif item_type == "function_call":
            # Normalize to internal tool-call format:
            # {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
            tool_calls.append({
                "id": item_data.get("call_id", item_data.get("id", "")),
                "type": "function",
                "function": {
                    "name": item_data.get("name", ""),
                    "arguments": item_data.get("arguments", "{}")
                }
            })
        elif item_type == "reasoning":
            # Some reasoning models surface reasoning text
            raw_summary: object = item_data.get("summary", [])
            summary = cast(list[object], raw_summary) if isinstance(raw_summary, list) else []
            reasoning_content = string_value(json_dict(summary[0]).get("text")) if summary else None

    # Also check the convenience output_text field
    if not content and data.get("output_text"):
        content = string_value(data.get("output_text"))

    usage = json_dict(data.get("usage"))
    fallback_tokens = _token_count(usage.get("input_tokens")) + _token_count(usage.get("output_tokens"))
    tokens = _token_count(usage.get("total_tokens", fallback_tokens))

    return content, reasoning_content, tool_calls, tokens


def append_openai_responses_tool_messages(
    messages: MessageList,
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: list[JsonDict],
    tool_messages: MessageList,
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
    assistant_msg: JsonDict = {"role": "assistant", "content": content}
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
