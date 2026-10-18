"""Contracts where reasoning replay meets the merged hardening fixes."""

import json
from copy import deepcopy

import pytest

from IkaModel.base import BareBoneModel
from IkaModel.chat_helpers_common import append_provider_tool_messages, build_provider_request
from IkaModel.chat_interface.chat_request import _parse_provider_round, init_message_history
from IkaModel.compaction import compact_live_messages
from IkaModel.runtime_errors import IkaProviderResponseError
from IkaModel.summarization import _write_summary_to_history
from IkaTest.test_reasoning_snapshot_replay import OPAQUE, PROVIDERS, response_for


@pytest.mark.parametrize("provider", ["codex", "openai_responses", "anthropic", "gemini"])
@pytest.mark.parametrize("malformed", [7, None, "bad", []])
def test_malformed_native_reasoning_items_fail_at_parse_boundary(provider, malformed):
    data = response_for(provider)
    items = (data["output"] if provider in {"codex", "openai_responses"} else data["content"]
             if provider == "anthropic" else data["candidates"][0]["content"]["parts"])
    items.append(malformed)
    original = deepcopy(data)
    history = init_message_history()
    with pytest.raises(IkaProviderResponseError):
        _parse_provider_round(provider, data, "fixture", message_history=history)
    assert "_pending_provider_data" not in history
    assert data == original


@pytest.mark.parametrize("provider", ["codex", "openai_responses", "anthropic", "gemini"])
@pytest.mark.parametrize("malformed", [7, "bad", {"type": "reasoning"}])
def test_malformed_native_collections_fail_with_history_enabled(provider, malformed):
    data = response_for(provider)
    if provider in {"codex", "openai_responses"}:
        data["output"] = malformed
    elif provider == "anthropic":
        data["content"] = malformed
    else:
        data["candidates"][0]["content"]["parts"] = malformed
    history = init_message_history()
    with pytest.raises(IkaProviderResponseError):
        _parse_provider_round(provider, data, "fixture", message_history=history)
    assert "_pending_provider_data" not in history


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("live", [False, True])
def test_compaction_replays_reasoning_with_its_own_tool_exchange(provider, live):
    route = "openai" if provider == "zai" else provider
    history = init_message_history()
    history["first_input"]["message"] = "task"
    history["messages"]["old"] = {"message": "discard old observation", "tokens": 10}
    model_id, url = PROVIDERS[provider]
    model = BareBoneModel(model_id, "fixture", url, suppress_init_output=True)
    content, reason, calls, tokens, _ = _parse_provider_round(
        route, response_for(provider), model_id, message_history=history)
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "discard old observation"}]
    tools = [{"role": "tool", "tool_call_id": "call_1", "content": "approved"}]
    if provider == "anthropic":
        tools = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "approved"}]}]
    append_provider_tool_messages(route, messages, history, content, reason, calls, tools, ["approved"], tokens,
                                  format_gemini_results_fn=lambda calls, results: [
                                      {"functionResponse": {"name": "ask_user", "response": {"result": results[0]}}}])
    expected_exchange = deepcopy(messages[2:])
    _write_summary_to_history(history, "preserved facts", 2)
    compact_live_messages(messages)
    assert messages == [{"role": "user", "content": "task"}, *expected_exchange]
    saved_exchange = [json.loads(entry["message"]) for entry in history["messages"].values()]
    assert saved_exchange == expected_exchange
    payload = build_provider_request(route, model, messages if live else [], history)[2]
    wire = payload.get("input", payload.get("messages", payload.get("contents")))
    encoded = json.dumps(wire)
    assert "discard old observation" not in encoded
    assert encoded.count("approved") == 1
    # Both the signed/encrypted state and its call/result remain, once each.
    assert encoded.count(OPAQUE) == (2 if provider == "anthropic" else 1)
    assert encoded.count('"call_1"') == (0 if provider == "gemini" else 2)


def test_compaction_keeps_every_reasoning_item_in_a_native_call_group():
    reasons = [{"type": "reasoning", "id": str(i), "encrypted_content": "opaque-" + str(i)} for i in range(2)]
    call = {"type": "function_call", "call_id": "call", "name": "lookup", "arguments": "{}"}
    result = {"type": "function_call_output", "call_id": "call", "output": "ok"}
    messages = [{"role": "assistant", "content": "old"}, *deepcopy(reasons), call, result]
    compact_live_messages(messages)
    assert messages == [*reasons, call, result]
