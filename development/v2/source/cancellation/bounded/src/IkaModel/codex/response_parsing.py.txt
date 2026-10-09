"""Parse Codex responses and record tool-message history."""

# pyright: strict

from __future__ import annotations

import json
import uuid
from typing import Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value

ProviderRound = tuple[str, Optional[str], list[JsonDict], int]
MessageList = list[JsonDict]


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def parse_codex_response(data: JsonDict, model_id: str) -> ProviderRound:
    """Convert a completed Codex response into IkaCore's provider tuple."""
    del model_id  # Codex response parsing is model-independent.
    content = ""
    tool_calls: list[JsonDict] = []
    reasoning_content: Optional[str] = None

    raw_output: object = data.get("output", []) or []
    output = cast(list[object], raw_output) if isinstance(raw_output, list) else []
    for item in output:
        item_data = json_dict(item)
        item_type = item_data.get("type", "")
        if item_type == "message":
            raw_blocks: object = item_data.get("content", []) or []
            blocks = cast(list[object], raw_blocks) if isinstance(raw_blocks, list) else []
            for block in blocks:
                block_data = json_dict(block)
                if block_data.get("type") == "output_text":
                    content += string_value(block_data.get("text"))
        elif item_type == "function_call":
            tool_calls.append({
                "id": item_data.get("call_id", item_data.get("id", "")),
                "type": "function",
                "function": {
                    "name": item_data.get("name", ""),
                    "arguments": item_data.get("arguments", "{}"),
                },
            })
        elif item_type == "reasoning":
            raw_summary: object = item_data.get("summary") or []
            summary = cast(list[object], raw_summary) if isinstance(raw_summary, list) else []
            parts = [
                string_value(cast(JsonDict, item).get("text"))
                for item in summary
                if isinstance(item, dict)
            ]
            joined = "\n".join(part for part in parts if part)
            if joined:
                reasoning_content = joined

    if not content and data.get("output_text"):
        content = string_value(data.get("output_text"))

    usage = json_dict(data.get("usage"))
    fallback_tokens = _token_count(usage.get("input_tokens")) + _token_count(usage.get("output_tokens"))
    tokens = _token_count(usage.get("total_tokens")) or fallback_tokens
    return content, reasoning_content, tool_calls, tokens


def append_codex_tool_messages(
    messages: MessageList,
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: list[JsonDict],
    tool_messages: MessageList,
    tokens: int,
    repeated_warning_msg: str = "",
) -> None:
    """Record one Codex assistant turn and its tool outputs."""
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
        history_section(message_history, "messages")[str(uuid.uuid4())] = {
            "message": json.dumps(assistant_msg),
            "tokens": tokens,
            "type": "assistant_with_tools",
        }
    for tool_msg in tool_messages:
        history_section(message_history, "messages")[str(uuid.uuid4())] = {
            "message": json.dumps(tool_msg),
            "tokens": 0,
            "type": "tool",
        }
