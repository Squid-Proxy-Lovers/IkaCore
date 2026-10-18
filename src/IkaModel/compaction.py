"""Structural retention for successful conversation compaction."""

# pyright: strict

import json
from typing import Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value


def _history_message_object(entry: JsonDict) -> JsonDict:
    try:
        return json_dict(json.loads(string_value(entry.get("message"))))
    except ValueError:
        return {}


def _tool_references(message: JsonDict) -> tuple[list[str], list[str]]:
    calls: list[object] = []
    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        calls.extend(cast(list[object], raw_calls))
    native_kind = message.get("type")
    if native_kind == "function_call":
        calls.append({"id": message.get("call_id", message.get("id"))})
    result = message.get("tool_call_id")
    if native_kind == "function_call_output":
        result = message.get("call_id")
    results = [result] if isinstance(result, str) else []
    blocks = message.get("content", message.get("parts"))
    if isinstance(blocks, list):
        for raw in cast(list[object], blocks):
            block = json_dict(raw)
            if block.get("type") == "tool_use":
                calls.append({"id": block.get("id")})
            if block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                results.append(str(block["tool_use_id"]))
            for field, target in (("functionCall", calls), ("functionResponse", None)):
                name = json_dict(block.get(field)).get("name")
                if isinstance(name, str):
                    if target is None:
                        results.append("gemini:" + name)
                    else:
                        target.append({"id": "gemini:" + name})
    identifiers = [json_dict(call).get("id") for call in calls]
    return [identifier for identifier in identifiers if isinstance(identifier, str)], results


def retained_compaction_entries(message_history: JsonDict) -> JsonDict:
    """Retain the latest user and whole tool exchanges, including pending calls."""
    entries = list(history_section(message_history, "messages").items())
    latest_user: Optional[int] = None
    latest_round: Optional[int] = None
    pending: dict[str, int] = {}
    reasoning_indexes: set[int] = set()
    native_start, previous_kind = 0, None
    for index, (_, raw) in enumerate(entries):
        entry = json_dict(raw)
        message = _history_message_object(entry)
        native_kind = message.get("type")
        if native_kind == "reasoning" or previous_kind not in {"reasoning", "function_call"}:
            native_start = index
        previous_kind = native_kind
        if entry.get("type") in {"stage_input", "hitl_input"}:
            latest_user = index
        if entry.get("type") == "assistant_with_tools":
            latest_round = native_start if native_kind == "function_call" else index
        if entry.get("reasoning_content") or native_kind == "reasoning":
            reasoning_indexes = {index}
        elif entry.get("type") == "tool" and index - 1 in reasoning_indexes:
            reasoning_indexes.add(index)
        calls, results = _tool_references(message)
        for call_id in calls:
            pending[call_id] = latest_round if latest_round is not None else index
        for result_id in results:
            pending.pop(result_id, None)
    starts = list(pending.values())
    if latest_round is not None:
        starts.append(latest_round)
    suffix = min(starts) if starts else len(entries)
    return {key: value for index, (key, value) in enumerate(entries)
            if index >= suffix or index == latest_user or index in reasoning_indexes}


def compact_live_messages(messages: list[JsonDict]) -> None:
    """Bound the live buffer as well as persisted history after a summary.

    Provider appenders use either roles or typed content blocks for tool
    exchanges. Keep those records verbatim so pairing and opaque reasoning
    survive; keep the latest ordinary user instruction separately.
    """
    entries: JsonDict = {}
    for index, message in enumerate(messages):
        blocks = message.get("content", message.get("parts", []))
        serialized = json.dumps(message)
        kind = "assistant"
        if message.get("role") == "tool" or message.get("type") == "function_call_output":
            kind = "tool"
        elif message.get("tool_calls") or message.get("type") == "function_call":
            kind = "assistant_with_tools"
        elif isinstance(blocks, list) and any(
            json_dict(block).get("type") == "tool_result" or "functionResponse" in json_dict(block)
            for block in cast(list[object], blocks)
        ):
            kind = "tool"
        elif isinstance(blocks, list) and any(
            json_dict(block).get("type") == "tool_use" or "functionCall" in json_dict(block)
            for block in cast(list[object], blocks)
        ):
            kind = "assistant_with_tools"
        elif message.get("role") == "user":
            kind = "hitl_input"
        entries[str(index)] = {"message": serialized, "type": kind,
                               "reasoning_content": message.get("reasoning_content")}
    retained = retained_compaction_entries({"messages": entries})
    keep = set(retained)
    if messages:
        keep.add(str(len(messages) - 1))
    messages[:] = [message for index, message in enumerate(messages)
                   if str(index) in keep or message.get("role") in {"system", "developer"}]
