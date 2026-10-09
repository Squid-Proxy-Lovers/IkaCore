"""Select and order persisted history for providers that replay it.

Providers that rebuild the transcript from ``message_history["messages"]``
(Anthropic, Gemini, DeepSeek) also receive the live ``messages`` buffer, and
the tool-message appenders write every round into both. Replaying all
persisted entries therefore sent each round of the current step twice.

The persisted history is still the only source for rounds that are *not*
in the live buffer: earlier stages (each stage starts a fresh buffer) and
rounds from before a checkpoint resume. So instead of dropping the replay,
we skip exactly those entries whose message is present in the live buffer.

Matching is exact and conservative: the appenders store ``json.dumps(msg)``
of the very dict they append to the buffer, so an entry matches only an
identical live message. Each live message cancels at most one entry,
matched from the most recent entry backwards, so identical content from an
earlier stage is kept. If an entry ever fails to match, it is replayed as
before; the helper can only remove duplicates, never information.

``plan_history_replay`` then decides the order. Each stage records its
opening prompt as a ``stage_input`` entry, so prior stages can be sent
chronologically with the current stage's prompt last; otherwise a request
could end on the model's own previous turn (which Anthropic continues as a
prefill) instead of on the instructions for the stage it is now in.
"""

# pyright: strict

from __future__ import annotations

import json
from collections import Counter
from typing import Any, NamedTuple, Optional, cast

JsonDict = dict[str, Any]

_ROUND_TYPES = frozenset({"assistant_with_tools", "tool"})
# Plain-text entries that were user turns: a stage's opening prompt and HITL answers.
USER_ENTRY_TYPES = frozenset({"stage_input", "hitl_input"})
_USER_TEXT_PREFIX = "\x00user:"


def _entry_key(entry: JsonDict) -> Optional[str]:
    raw = entry.get("message", "")
    if not isinstance(raw, str):
        return None
    entry_type = entry.get("type", "assistant")
    if entry_type in _ROUND_TYPES:
        return raw
    if entry_type in USER_ENTRY_TYPES:
        # Persisted as plain text; sent live as a user message.
        return _USER_TEXT_PREFIX + raw
    return None


def _live_keys(messages: list[Any]) -> Counter[str]:
    keys: Counter[str] = Counter()
    for message in messages:
        if not isinstance(message, dict):
            continue
        mapped = cast(JsonDict, message)
        content = mapped.get("content")
        if mapped.get("role") == "user" and isinstance(content, str):
            keys[_USER_TEXT_PREFIX + content] += 1
            continue
        keys[json.dumps(mapped)] += 1
    return keys


def history_entries_to_replay(message_history: JsonDict, messages: list[Any]) -> list[JsonDict]:
    """Persisted history entries, in order, minus those already present in ``messages``."""
    raw_entries = message_history.get("messages") or {}
    if not isinstance(raw_entries, dict):
        return []
    entries = [cast(JsonDict, e) for e in cast(dict[str, Any], raw_entries).values() if isinstance(e, dict)]
    if not entries or not messages:
        return entries

    live: Optional[Counter[str]] = None
    kept: list[JsonDict] = []
    for entry in reversed(entries):
        key = _entry_key(entry)
        if key is not None:
            if live is None:
                live = _live_keys(messages)
            if live[key] > 0:
                live[key] -= 1
                continue
        kept.append(entry)
    kept.reverse()
    return kept


class ReplayPlan(NamedTuple):
    """How a history-replaying provider should order one request's transcript."""

    entries: list[JsonDict]  # persisted entries to send, oldest first
    first_input_first: bool  # True: first_input, summary, entries, live (the original layout)
    emit_first_input: bool  # chronological layout only: whether to send first_input after entries


def is_user_entry(entry: JsonDict) -> bool:
    return entry.get("type") in USER_ENTRY_TYPES


def _is_empty_model_text(entry: JsonDict) -> bool:
    if entry.get("type", "assistant") in _ROUND_TYPES or is_user_entry(entry):
        return False
    raw = entry.get("message", "")
    return isinstance(raw, str) and not raw.strip()


def _section_text(message_history: JsonDict, key: str) -> str:
    section = message_history.get(key)
    if not isinstance(section, dict):
        return ""
    text = cast(JsonDict, section).get("message", "")
    return text if isinstance(text, str) else ""


def plan_history_replay(message_history: JsonDict, messages: list[Any]) -> ReplayPlan:
    """Order replayed history so a request never ends on a stale model turn.

    Prior stages are sent chronologically (their recorded stage prompts are user
    turns) and the current stage's prompt comes after them. The original layout
    is kept when there is nothing to replay, when a summary has replaced older
    history, or when the history predates recorded stage prompts (it would not
    start with a user turn). Empty model-text entries are never replayed: they
    carry nothing and some providers reject empty turns mid-conversation.
    """
    entries = [e for e in history_entries_to_replay(message_history, messages) if not _is_empty_model_text(e)]
    chronological = bool(entries) and is_user_entry(entries[0]) and not _section_text(message_history, "summary")
    if not chronological:
        return ReplayPlan(entries, first_input_first=True, emit_first_input=True)
    first_input = _section_text(message_history, "first_input")
    already_sent = any(e.get("type") == "stage_input" and e.get("message") == first_input for e in entries)
    return ReplayPlan(entries, first_input_first=False, emit_first_input=bool(first_input) and not already_sent)


__all__ = ["ReplayPlan", "USER_ENTRY_TYPES", "history_entries_to_replay", "is_user_entry", "plan_history_replay"]
