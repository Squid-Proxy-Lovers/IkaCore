"""Opaque provider state belongs to its assistant turn, never to display text.

The public parsers retain their four-value interface. Passing history to
parse_provider_response also captures replay data for the pending turn;
appenders/finalization consume it into the persisted assistant entry. This
JSON-only envelope survives snapshots and is deliberately absent from tool
arguments, reasoning_content, and CLI output.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, cast

from .history_replay import is_user_entry, plan_history_replay

JsonDict = dict[str, Any]
_PENDING = "_pending_provider_data"


def capture_response_replay(provider: str, data: JsonDict, history: JsonDict) -> None:
    """Capture complete output items, including empty-summary encrypted reasoning."""
    history.pop(_PENDING, None)
    if provider in {"codex", "openai_responses"}:
        output = data.get("output") or []
        if any(isinstance(item, dict) and item.get("type") == "reasoning" for item in output):
            history[_PENDING] = {"provider": "responses", "items": deepcopy(output)}
    elif provider == "anthropic":
        blocks = data.get("content") or []
        if any(isinstance(block, dict) and block.get("type") in {"thinking", "redacted_thinking"} for block in blocks):
            history[_PENDING] = {"provider": "anthropic", "items": deepcopy(blocks)}
    elif provider == "gemini":
        candidates = data.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        if any(isinstance(part, dict) and ("thoughtSignature" in part or "thought_signature" in part or part.get("thought"))
               for part in parts):
            history[_PENDING] = {"provider": "gemini", "items": deepcopy(parts)}
    elif provider == "openrouter":
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        fields = {key: deepcopy(message[key]) for key in ("reasoning", "reasoning_content", "reasoning_details")
                  if key in message}
        if fields:
            history[_PENDING] = {"provider": "openrouter", "fields": fields}


def take_response_replay(history: JsonDict, *, final: bool = False) -> JsonDict:
    raw = history.pop(_PENDING, None)
    data = cast(JsonDict, raw) if isinstance(raw, dict) else {}
    if final and "items" in data:
        # Control tools finish locally, without outputs to redeem orphan calls.
        data["items"] = [item for item in data["items"]
                         if item.get("type") not in {"function_call", "tool_use"} and "functionCall" not in item]
    return data


def replay_items(message: JsonDict, provider: str) -> list[JsonDict] | None:
    """Return only this provider's opaque data; never translate signed material."""
    data = message.get("provider_data")
    if not isinstance(data, dict) or data.get("provider") != provider:
        return None
    items = data.get("items")
    return deepcopy(items) if isinstance(items, list) else None


def history_message(entry: JsonDict) -> JsonDict | None:
    raw = entry.get("message", "")
    if entry.get("type") in {"assistant_with_tools", "tool"}:
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return message if isinstance(message, dict) else None
    message = {"role": "user" if is_user_entry(entry) else "assistant", "content": raw}
    if entry.get("provider_data"):
        message["provider_data"] = entry["provider_data"]
    if entry.get("reasoning_content"):
        message["reasoning_content"] = entry["reasoning_content"]
    return message



def _merge_live_assistants(entries: list[JsonDict], messages: list[JsonDict]) -> tuple[list[JsonDict], list[JsonDict]]:
    # Agent step buffers also contain completed plain-text turns. Match backwards
    # one-for-one, moving opaque state onto that live turn instead of duplicating it.
    live = list(messages)
    matched: set[int] = set()
    kept = []
    for entry in reversed(entries):
        if entry.get("type", "assistant") in {"assistant_with_tools", "tool"} or is_user_entry(entry):
            kept.append(entry)
            continue
        index = next((i for i in range(len(live) - 1, -1, -1)
                      if i not in matched and isinstance(live[i], dict)
                      and live[i].get("role") == "assistant" and live[i].get("content") == entry.get("message")), None)
        if index is None:
            kept.append(entry)
        else:
            matched.add(index)
            if entry.get("provider_data"):
                live[index] = {**live[index], "provider_data": entry["provider_data"]}
    return list(reversed(kept)), live


def ordered_replay_messages(messages: list[JsonDict], history: JsonDict, *, summary_first: bool = True) -> list[JsonDict]:
    """Merge persisted and live turns once, retaining the owning assistant's order."""
    plan = plan_history_replay(history, messages)
    entries, messages = _merge_live_assistants(plan.entries, messages)
    replayed = [message for entry in entries if (message := history_message(entry)) is not None]
    first = history.get("first_input", {}).get("message", "")
    summary = history.get("summary", {}).get("message", "")
    context = []
    if summary:
        context.append({"role": "assistant", "content": summary})
    if first:
        context.append({"role": "user", "content": first})
    if not summary_first:
        context.reverse()
    if plan.first_input_first:
        ordered = context + replayed
    else:
        ordered = replayed + ([{"role": "user", "content": first}] if plan.emit_first_input else [])
    skip_first = bool(first and messages and isinstance(messages[0], dict) and messages[0].get("content") == first)
    return ordered + messages[1 if skip_first else 0:]


def redact_opaque_reasoning(value: Any) -> Any:
    """Return a log-only copy; signed/encrypted provider data must never be rendered."""
    if isinstance(value, dict):
        opaque = {"encrypted_content", "signature", "thoughtSignature", "thought_signature",
                  "gemini_thought_signature", "provider_data", "_pending_provider_data"}
        if value.get("type") in {"redacted_thinking", "reasoning.encrypted"}:
            opaque.add("data")
        return {key: "[REDACTED]" if key in opaque else redact_opaque_reasoning(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_opaque_reasoning(item) for item in value]
    return value
