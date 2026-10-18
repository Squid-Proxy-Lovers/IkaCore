"""Controlled offline retention fixtures, shared with the opt-in live evaluation."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from IkaModel.base import BareBoneModel
from IkaModel.summarization import async_summarise_message_history, run_summarization
from IkaTest.compaction_fixtures import retention_fixture


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("index", range(4))
def test_controlled_compaction_contains_facts_and_keeps_pending_structure(monkeypatch, asynchronous, index):
    facts, history = retention_fixture(index)
    before = copy.deepcopy(history)
    scripted = json.dumps(facts)
    response = httpx.Response(200, json={"choices": [{"message": {"content": scripted}}], "usage": {"total_tokens": 70}}, request=httpx.Request("POST", "https://example.test"))
    request = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr("IkaModel.summarization." + ("async_api_request_retry" if asynchronous else "api_request_retry"), request)
    model = BareBoneModel("gpt-4o", "test", "https://api.openai.com/v1/chat/completions", use_responses_api=False, suppress_init_output=True)
    result = asyncio.run(async_summarise_message_history(model, history)) if asynchronous else run_summarization(model, history)
    payload = request.call_args.args[2]
    serialized = json.dumps(payload)
    for value in facts.values():
        assert value in serialized
    assert "Check integrity before transfer" in serialized
    assert "pending tool calls" in payload["messages"][0]["content"]
    assert result == scripted
    assert history["summary"] == {"message": "[SUMMARY]\n" + scripted, "tokens": 70}
    assert history["system"] == before["system"] and history["first_input"] == before["first_input"]
    assert history["messages"]["user"] == before["messages"]["user"]
    assert history["messages"]["call"] == before["messages"]["call"]
    assert history["messages"]["result"] == before["messages"]["result"]
    assert history["messages"]["pending"] == before["messages"]["pending"]
    assert history["messages"]["pending-note"] == before["messages"]["pending-note"]
    assert not any(key.startswith("noise-") for key in history["messages"])
    assert facts["result"] not in json.dumps(history["messages"])
    assert facts["question"] not in json.dumps(history["messages"])
    # A pending assistant/tool call is retained verbatim. No orphan tool result.
    seen = set()
    for entry in history["messages"].values():
        if entry.get("type") not in {"assistant_with_tools", "tool"}:
            continue
        message = json.loads(entry["message"])
        seen.update(call["id"] for call in message.get("tool_calls", []))
        if message.get("role") == "tool":
            assert message["tool_call_id"] in seen


@pytest.mark.parametrize("asynchronous", [False, True])
def test_empty_summary_does_not_erase_history(monkeypatch, asynchronous):
    _, history = retention_fixture()
    history["messages"]["pending-note"]["reasoning_content"] = "UNIQUE_SEPARATELY_STORED_REASONING"
    before = copy.deepcopy(history)
    response = httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]}, request=httpx.Request("POST", "https://example.test"))
    request = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr("IkaModel.summarization." + ("async_api_request_retry" if asynchronous else "api_request_retry"), request)
    model = BareBoneModel("gpt-4o", "test", "https://api.openai.com/v1/chat/completions", suppress_init_output=True)
    if asynchronous:
        asyncio.run(async_summarise_message_history(model, history))
    else:
        run_summarization(model, history)
    assert history == before
    assert "UNIQUE_SEPARATELY_STORED_REASONING" in json.dumps(request.call_args.args[2])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_compaction_also_bounds_the_live_request_buffer(monkeypatch, asynchronous):
    from IkaModel.chat_interface import chat_runtime

    facts, history = retention_fixture()
    messages = [{"role": "user", "content": facts["goal"]}]
    messages.extend({"role": "assistant", "content": "old observation " + str(i)} for i in range(1000))
    messages.append({"role": "user", "content": facts["question"]})
    messages.append(json.loads(history["messages"]["call"]["message"]))
    messages.append(json.loads(history["messages"]["result"]["message"]))
    messages.append(json.loads(history["messages"]["pending"]["message"]))
    expected_tail = copy.deepcopy(messages[-4:])
    model = BareBoneModel("gpt-4o", "test", "https://api.openai.com/v1/chat/completions")
    model.context_budget = 100
    response = httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(facts)}}]}, request=httpx.Request("POST", "https://example.test"))
    request = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr("IkaModel.summarization." + ("async_api_request_retry" if asynchronous else "api_request_retry"), request)
    round_result = ("after", None, [], 0, {})
    monkeypatch.setattr(chat_runtime, "_request_provider_round", Mock(return_value=round_result))
    monkeypatch.setattr(chat_runtime, "_request_provider_round_async", AsyncMock(return_value=round_result))
    if asynchronous:
        result = asyncio.run(chat_runtime.async_chat(model, messages, history, client=object()))
    else:
        result = chat_runtime.chat(model, messages, history)
    assert result["content"] == "after"
    assert messages == expected_tail


@pytest.mark.parametrize("asynchronous", [False, True])
def test_context_overflow_retry_uses_compacted_live_buffer(monkeypatch, asynchronous):
    from IkaModel.chat_interface import chat_request
    from IkaModel.request_interface import IkaAPIError

    _, history = retention_fixture()
    messages = [{"role": "user", "content": "task"}, *[{"role": "assistant", "content": "old"} for _ in range(50)], {"role": "user", "content": "latest"}]
    seen = []
    def build():
        seen.append(copy.deepcopy(messages))
        return "https://example.test", {}, {}
    summary_response = httpx.Response(200, json={"choices": [{"message": {"content": "preserved"}}]}, request=httpx.Request("POST", "https://example.test"))
    success = httpx.Response(200, json={"ok": True})
    summary = AsyncMock(return_value=summary_response) if asynchronous else Mock(return_value=summary_response)
    request = AsyncMock(side_effect=[IkaAPIError("maximum context length"), success]) if asynchronous else Mock(side_effect=[IkaAPIError("maximum context length"), success])
    monkeypatch.setattr("IkaModel.summarization." + ("async_api_request_retry" if asynchronous else "api_request_retry"), summary)
    monkeypatch.setattr(chat_request._requests, "async_api_request_retry" if asynchronous else "api_request_retry", request)
    model = BareBoneModel("gpt-4o", "test", "https://api.openai.com/v1/chat/completions")
    if asynchronous:
        response = asyncio.run(chat_request._api_request_with_context_fallback_async(build, model, history, messages=messages))
    else:
        response = chat_request._api_request_with_context_fallback(build, model, history, messages=messages)
    assert response is success
    assert len(seen[0]) == 52 and seen[1] == [{"role": "user", "content": "latest"}]


@pytest.mark.parametrize("wire", ["openai", "anthropic", "gemini"])
def test_compact_live_preserves_system_and_complete_provider_tool_exchange(wire):
    from IkaModel.summarization import compact_live_messages

    if wire == "openai":
        exchange = [{"role": "assistant", "tool_calls": [{"id": "a"}]}, {"role": "tool", "tool_call_id": "a", "content": "result"}]
    elif wire == "anthropic":
        exchange = [{"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "result"}]}]
    else:
        exchange = [{"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "opaque"}]}, {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {"result": "ok"}}}]}]
    system = {"role": "system", "content": "constraints"}
    user = {"role": "user", "content": "latest user"}
    messages = [system, *[{"role": "assistant", "content": "noise"} for _ in range(10)], user, *exchange]
    expected = copy.deepcopy([system, user, *exchange])
    compact_live_messages(messages)
    assert messages == expected
    compact_live_messages(messages)
    assert messages == expected
    empty = []
    compact_live_messages(empty)
    assert empty == []


def test_compaction_preserves_native_parallel_call_group_and_encrypted_reasoning():
    from IkaModel.summarization import compact_live_messages

    user = {"role": "user", "content": "task"}
    opaque = {"type": "reasoning", "id": "reason", "encrypted_content": "opaque-byte-sequence"}
    calls = [{"type": "function_call", "call_id": name, "name": "lookup", "arguments": "{}"} for name in ("a", "b")]
    results = [{"type": "function_call_output", "call_id": name, "output": "ok"} for name in ("a", "b")]
    messages = [user, {"role": "assistant", "content": "discard"}, opaque, *calls, *results]
    expected = copy.deepcopy([user, opaque, *calls, *results])
    compact_live_messages(messages)
    assert messages == expected
    compact_live_messages(messages)
    assert messages == expected
