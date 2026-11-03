import json
from types import SimpleNamespace

import pytest

from IkaModel.anthropic.claude import anthropic_fill_payload
from IkaModel.deepseek.deepseek import deepseek_fill_payload
from IkaModel.gemini.google import gemini_fill_payload
from IkaModel.history_replay import history_entries_to_replay


def _history(*entries):
    return {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "Do the task", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {f"id-{i}": entry for i, entry in enumerate(entries)},
    }


def _round(msg, kind):
    return {"message": json.dumps(msg), "tokens": 0, "type": kind}


def _anthropic_round(call_id):
    assistant = {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": "lookup", "input": {}}]}
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": "ok"}]}
    return assistant, result


def test_rounds_present_in_live_buffer_are_not_replayed():
    assistant, result = _anthropic_round("call_1")
    history = _history(_round(assistant, "assistant_with_tools"), _round(result, "tool"))
    live = [{"role": "user", "content": "Do the task"}, assistant, result]
    assert history_entries_to_replay(history, live) == []


def test_earlier_stage_rounds_are_still_replayed_in_order():
    old_a, old_r = _anthropic_round("call_stage0")
    new_a, new_r = _anthropic_round("call_stage1")
    entries = [_round(old_a, "assistant_with_tools"), _round(old_r, "tool"),
               _round(new_a, "assistant_with_tools"), _round(new_r, "tool")]
    live = [{"role": "user", "content": "Stage 1 prompt"}, new_a, new_r]
    assert history_entries_to_replay(_history(*entries), live) == entries[:2]


def test_resume_with_fresh_buffer_replays_everything():
    assistant, result = _anthropic_round("call_1")
    entries = [_round(assistant, "assistant_with_tools"), _round(result, "tool")]
    assert history_entries_to_replay(_history(*entries), [{"role": "user", "content": "answer"}]) == entries


def test_identical_earlier_content_is_kept_matching_most_recent_first():
    msg = {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {"r": 1}}}]}
    entries = [_round(msg, "tool"), _round(msg, "tool")]
    kept = history_entries_to_replay(_history(*entries), [msg])
    assert kept == [entries[0]]


def test_hitl_answer_is_not_echoed_when_sent_live():
    entries = [{"message": "use 2024", "tokens": 0, "type": "hitl_input"}]
    assert history_entries_to_replay(_history(*entries), [{"role": "user", "content": "use 2024"}]) == []
    assert history_entries_to_replay(_history(*entries), [{"role": "user", "content": "other"}]) == entries


def test_plain_model_messages_and_unmatched_entries_are_untouched():
    assistant, _ = _anthropic_round("call_1")
    mutated = dict(assistant, content=[])
    entries = [{"message": "", "tokens": 0}, _round(assistant, "assistant_with_tools")]
    live = [{"role": "assistant", "content": ""}, mutated]
    assert history_entries_to_replay(_history(*entries), live) == entries


def _model(model_id):
    return SimpleNamespace(
        model_id=model_id, max_tokens=1000, temperature=0.0, system_prompt="", agent_tools=[],
        parallel_tool_calls=False, forced_tool_name=None,
    )


def _gemini_round(call_id):
    assistant = {"role": "model", "parts": [{"text": ""}, {"functionCall": {"name": "lookup", "args": {"id": call_id}}}]}
    result = {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {"id": call_id}}}]}
    return assistant, result


def _deepseek_round(call_id):
    call = {"id": call_id, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    return {"role": "assistant", "content": "", "tool_calls": [call]}, {"role": "tool", "tool_call_id": call_id, "content": "ok"}


@pytest.mark.parametrize(
    "fill, model_id, key, make_round",
    [
        (anthropic_fill_payload, "claude-sonnet-4-5", "messages", _anthropic_round),
        (deepseek_fill_payload, "deepseek-chat", "messages", _deepseek_round),
        (gemini_fill_payload, "gemini-2.5-flash", "contents", _gemini_round),
    ],
)
def test_provider_payloads_send_each_round_once(fill, model_id, key, make_round):
    first_a, first_r = make_round("call_1")
    second_a, second_r = make_round("call_2")
    history = _history(*(_round(m, k) for m, k in (
        (first_a, "assistant_with_tools"), (first_r, "tool"), (second_a, "assistant_with_tools"), (second_r, "tool"),
    )))
    live = [{"role": "user", "content": "Do the task"}, first_a, first_r, second_a, second_r]
    conversation = [json.dumps(m, sort_keys=True) for m in fill(_model(model_id), live, history)[key]]
    assert len(conversation) == len(set(conversation))
    assert sum("call_1" in m for m in conversation) == 2
    assert sum("call_2" in m for m in conversation) == 2
    assert conversation.index(next(m for m in conversation if "call_1" in m)) < conversation.index(
        next(m for m in conversation if "call_2" in m)
    )


# ---------------------------------------------------------------------------
# Stage-transition ordering (plan_history_replay)
# ---------------------------------------------------------------------------

from IkaModel.history_replay import plan_history_replay  # noqa: E402


def _stage_history(first_input, *entries, summary=""):
    history = _history(*entries)
    history["first_input"]["message"] = first_input
    history["summary"]["message"] = summary
    return history


def _stage_input(text):
    return {"message": text, "tokens": 0, "type": "stage_input"}


def test_first_stage_keeps_original_layout():
    history = _stage_history("Stage 0 prompt", _stage_input("Stage 0 prompt"))
    plan = plan_history_replay(history, [{"role": "user", "content": "Stage 0 prompt"}])
    assert plan.first_input_first and plan.entries == []


def test_later_stage_replays_prior_stage_then_ends_on_current_prompt():
    a, r = _anthropic_round("call_s0")
    entries = [_stage_input("Stage 0 prompt"), _round(a, "assistant_with_tools"), _round(r, "tool"),
               {"message": "Now I'll call stage_end.", "tokens": 0}, _stage_input("Stage 1 prompt")]
    history = _stage_history("Stage 1 prompt", *entries)
    live = [{"role": "user", "content": "Stage 1 prompt"}]
    plan = plan_history_replay(history, live)
    assert not plan.first_input_first and plan.emit_first_input
    assert plan.entries == entries[:4]

    payload = anthropic_fill_payload(_model("claude-sonnet-4-5"), live, history)["messages"]
    assert payload[0] == {"role": "user", "content": "Stage 0 prompt"}
    assert payload[-1] == {"role": "user", "content": "Stage 1 prompt"}
    assert sum(m["content"] == "Stage 1 prompt" for m in payload) == 1


def test_hitl_resume_keeps_stage_prompt_in_place_and_ends_on_answer():
    a, r = _anthropic_round("call_ask")
    entries = [_stage_input("Stage 0 prompt"), _round(a, "assistant_with_tools"), _round(r, "tool"),
               {"message": "use 2024", "tokens": 0, "type": "hitl_input"}]
    history = _stage_history("Stage 0 prompt", *entries)
    live = [{"role": "user", "content": "use 2024"}]
    plan = plan_history_replay(history, live)
    assert not plan.first_input_first and not plan.emit_first_input  # prompt already sent in place
    payload = anthropic_fill_payload(_model("claude-sonnet-4-5"), live, history)["messages"]
    assert payload[0]["content"] == "Stage 0 prompt" and payload[-1] == {"role": "user", "content": "use 2024"}
    assert sum(m["content"] == "use 2024" for m in payload) == 1


def test_summary_or_legacy_history_keeps_original_layout():
    a, r = _anthropic_round("call_old")
    legacy = _stage_history("Stage 1 prompt", _round(a, "assistant_with_tools"), _round(r, "tool"))
    assert plan_history_replay(legacy, [{"role": "user", "content": "Stage 1 prompt"}]).first_input_first
    summarized = _stage_history("Stage 1 prompt", _stage_input("Stage 0 prompt"), summary="earlier work")
    assert plan_history_replay(summarized, [{"role": "user", "content": "Stage 1 prompt"}]).first_input_first


def test_empty_model_text_entries_are_not_replayed():
    history = _stage_history("p", {"message": "", "tokens": 0}, {"message": "  ", "tokens": 0}, {"message": "kept", "tokens": 0})
    assert [e["message"] for e in plan_history_replay(history, [{"role": "user", "content": "p"}]).entries] == ["kept"]


def test_anthropic_payload_never_contains_whitespace_only_text():
    # Regression: tool-only assistant turns were padded with " ", which the Anthropic API rejects with a 400.
    tool_only = {"role": "assistant", "content": [{"type": "text", "text": " "},
                                                  {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}}]}
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
    blank = {"role": "assistant", "content": " "}
    history = _history(_round(tool_only, "assistant_with_tools"))
    live = [{"role": "user", "content": "Do the task"}, result, blank]
    messages = anthropic_fill_payload(_model("claude-sonnet-4-5"), live, history)["messages"]

    for m in messages:
        blocks = [m["content"]] if isinstance(m["content"], str) else [b.get("text") for b in m["content"] if b.get("type") == "text"]
        assert all(text.strip() for text in blocks if text is not None), m
    assistant = next(m for m in messages if m["role"] == "assistant")
    assert assistant["content"] == [{"type": "tool_use", "id": "t1", "name": "lookup", "input": {}}]
    assert blank not in messages


def test_anthropic_appender_does_not_pad_tool_only_turns():
    from IkaModel.anthropic.chat_helpers_anthropic import append_anthropic_tool_messages

    messages, history = [], _history()
    call = {"id": "t1", "name": "lookup", "function": {"name": "lookup", "arguments": "{}"}}
    append_anthropic_tool_messages(messages, history, "", None, [call], [], 0)
    assert messages[0]["content"] == [{"type": "tool_use", "id": "t1", "name": "lookup", "input": {}}]
