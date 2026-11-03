import json
import logging
from copy import deepcopy

import httpx
import pytest

from IkaModel.chat_helpers_common import parse_provider_response
from IkaModel.chat_interface.chat_response import _record_model_message
from IkaModel.reasoning_replay import (
    capture_response_replay,
    history_message,
    ordered_replay_messages,
    redact_opaque_reasoning,
    replay_items,
)
from IkaModel.request_interface import _log_http_request, _log_http_response


def history():
    return {"system": {"message": "", "tokens": 0}, "first_input": {"message": "task", "tokens": 0},
            "summary": {"message": "", "tokens": 0}, "messages": {}}


def test_completed_assistant_reconciles_live_buffer_and_keeps_metadata_once():
    h = history()
    h["messages"]["1"] = {"message": "same", "tokens": 1}
    h["messages"]["2"] = {"message": "same", "tokens": 1, "provider_data":
                         {"provider": "responses", "items": [{"type": "reasoning", "id": "rs_final", "summary": []}]}}
    live = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "same"}]
    original = deepcopy(live)
    merged = ordered_replay_messages(live, h)
    assert live == original
    assert len(merged) == 3
    assert merged[1] == {"role": "assistant", "content": "same"}
    assert merged[2]["provider_data"]["items"][0]["id"] == "rs_final"


def test_ordered_replay_context_chronology_and_malformed_entries():
    h = history()
    h["summary"]["message"] = "summary"
    assert [m["content"] for m in ordered_replay_messages([], h)] == ["summary", "task"]
    assert [m["content"] for m in ordered_replay_messages([], h, summary_first=False)] == ["task", "summary"]
    h["summary"]["message"] = ""
    h["messages"] = {"1": {"type": "stage_input", "message": "previous"},
                     "2": {"message": "result", "reasoning_content": "thought"},
                     "3": {"type": "assistant_with_tools", "message": "invalid-json"}}
    ordered = ordered_replay_messages([{"role": "user", "content": "task"}], h)
    assert [m["content"] for m in ordered] == ["previous", "result", "task"]
    assert ordered[1]["reasoning_content"] == "thought"
    assert history_message({"type": "tool", "message": "[]"}) is None
    assert history_message({"type": "tool", "message": None}) is None
    assert history_message({"type": "tool", "message": json.dumps({"role": "tool", "content": "ok"})})["content"] == "ok"


def test_opaque_provider_data_is_never_translated_to_a_different_provider():
    msg = {"provider_data": {"provider": "responses", "items": [{"type": "reasoning", "summary": []}]}}
    assert replay_items(msg, "anthropic") is None
    assert replay_items({"provider_data": "invalid"}, "responses") is None
    assert replay_items({"provider_data": {"provider": "responses", "items": None}}, "responses") is None
    native = replay_items(msg, "responses")
    native[0]["summary"].append("mutate")
    assert msg["provider_data"]["items"][0]["summary"] == []


@pytest.mark.parametrize("provider", ["codex", "openai_responses", "anthropic", "gemini", "openrouter", "deepseek"])
def test_empty_response_clears_pending_replay(provider):
    h = {"_pending_provider_data": {"provider": "stale"}}
    capture_response_replay(provider, {}, h)
    assert "_pending_provider_data" not in h


def test_final_gemini_control_call_does_not_leave_an_orphan_call():
    h = history()
    data = {"candidates": [{"content": {"parts": [
        {"text": "done", "thoughtSignature": "opaque"},
        {"functionCall": {"name": "stage_end", "args": {}}}
    ]}}]}
    content, reason, _, tokens = parse_provider_response("gemini", data, "gemini", h)
    _record_model_message(h, content, tokens, reason)
    entry = next(iter(h["messages"].values()))
    assert entry["provider_data"]["items"] == [{"text": "done", "thoughtSignature": "opaque"}]


def test_http_debug_redaction_is_nonmutating_and_covers_encrypted_data(caplog):
    caplog.set_level(logging.DEBUG)
    body = {"items": [
        {"encrypted_content": "opaque-encrypted", "signature": "opaque-signature"},
        {"type": "redacted_thinking", "data": "opaque-redacted"},
        {"type": "reasoning.encrypted", "data": "opaque-router"},
        {"thoughtSignature": "opaque-gemini", "thought_signature": "opaque-python"},
    ]}
    original = deepcopy(body)
    _log_http_request("FIXTURE", "https://example.test", {}, body, 0, 1)
    _log_http_response("FIXTURE", httpx.Response(200, json=body))
    _log_http_response("FIXTURE", httpx.Response(200, text="data: opaque-sse",
                                               headers={"content-type": "text/event-stream"}))
    assert "opaque-" not in caplog.text
    assert "[REDACTED]" in caplog.text and "[SSE body omitted]" in caplog.text
    assert body == original
    assert redact_opaque_reasoning(3) == 3
