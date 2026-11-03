"""Golden two-request loops: opaque reasoning is transport state, not output."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_helpers_common import (
    append_provider_tool_messages,
    build_provider_request,
    parse_provider_response,
)
from IkaModel.chat_interface.chat_interface import async_chat, chat, init_message_history
from IkaModel.codex.codex_stream import async_collect_stream, collect_stream

OPAQUE = "opaque-encrypted-provider-data"
REASON = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": OPAQUE}
CALL = {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}
OUTPUT = [REASON, {"type": "message", "id": "msg_1", "role": "assistant",
                  "content": [{"type": "output_text", "text": "Checking.", "annotations": []}]}, CALL]
FINAL = {"output": [{"type": "message", "id": "msg_final", "role": "assistant",
                     "content": [{"type": "output_text", "text": "done"}]}],
         "usage": {"input_tokens": 4, "output_tokens": 2}}
URLS = {"codex": "https://chatgpt.com/backend-api/codex/responses",
        "openai_responses": "https://api.openai.com/v1/responses"}


def model_for(provider):
    model = BareBoneModel("gpt-5.6-sol", "fixture", URLS[provider], system_prompt="instructions", reasoning_effort="low")
    model.agent_tools = [AgentTool("lookup", "lookup", "lookup", ToolArgs("object", "args", properties={}))]
    return model


def sse(data):
    # The added event carries an incomplete cipher. Only the done item is replayable.
    events = [{"type": "response.output_item.added", "output_index": 0,
               "item": {**REASON, "encrypted_content": "incomplete-do-not-replay"}}]
    events += [{"type": "response.output_item.done", "output_index": i, "item": item}
               for i, item in enumerate(data["output"])]
    events += [{"type": "response.completed", "response": {**data, "status": "completed"}}]
    return "\n\n".join("data: " + json.dumps(e) for e in events) + "\n\n"


@pytest.mark.parametrize("provider", URLS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_two_turn_reasoning_payload(provider, asynchronous, capsys):
    requests = []
    first = {"output": deepcopy(OUTPUT), "usage": {"input_tokens": 3, "output_tokens": 5}}

    def respond(request):
        requests.append(json.loads(request.content))
        data = first if len(requests) == 1 else FINAL
        if provider == "codex":
            return httpx.Response(200, text=sse(data), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=data)

    model = model_for(provider)
    history = init_message_history()
    messages = [{"role": "user", "content": "task"}]
    kwargs = {"tool_executors": {"lookup": lambda args: "result"}, "max_tool_rounds": 2}
    if asynchronous:
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                return await async_chat(model, messages, history, client=client, **kwargs)
        result = asyncio.run(run())
    else:
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            result = chat(model, messages, history, client=client, **kwargs)
    assert result["content"] == "done"
    assert len(requests) == 2
    expected_user = {"type": "message", "role": "user", "content":
                     [{"type": "input_text", "text": "task"}] if provider == "codex" else "task"}
    assert requests[0]["input"] == [expected_user]
    expected = deepcopy(requests[0])
    expected["input"] += OUTPUT + [{"type": "function_call_output", "call_id": "call_1", "output": "result"}]
    assert requests[1] == expected
    assert requests[1]["include"] == ["reasoning.encrypted_content"]
    assert requests[1]["store"] is False
    assert OPAQUE in json.dumps(history)
    assert OPAQUE not in capsys.readouterr().out

    # A resumed call with a fresh live buffer must replay exactly one copy.
    resumed = build_provider_request(provider, model, [{"role": "user", "content": "next"}], history)[2]
    assert resumed["input"].count(REASON) == 1
    assert resumed["input"].index(REASON) < resumed["input"].index(CALL)


@pytest.mark.parametrize("provider", URLS)
def test_capture_all_reasoning_items_and_replay_final_turn(provider):
    from IkaModel.chat_interface.chat_response import _record_model_message

    history = init_message_history()
    raw = {"output": [REASON, {**REASON, "id": "rs_2", "summary": [{"type": "summary_text", "text": "summary"}]}]}
    parsed = parse_provider_response(provider, raw, "gpt-5.6-sol", history)
    _record_model_message(history, parsed[0], 0, parsed[1])
    payload = build_provider_request(provider, model_for(provider), [{"role": "user", "content": "next"}], history)[2]
    assert payload["input"][:2] == raw["output"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_codex_stream_retains_completed_encrypted_items(asynchronous):
    response = httpx.Response(200, text=sse({"output": OUTPUT}))
    result = asyncio.run(async_collect_stream(response)) if asynchronous else collect_stream(response)
    assert result["output"] == OUTPUT


@pytest.mark.parametrize("provider", URLS)
def test_direct_parse_history_contract(provider):
    history = init_message_history()
    content, reasoning, calls, tokens = parse_provider_response(provider, {"output": OUTPUT}, "gpt-5.6-sol", history)
    messages = []
    append_provider_tool_messages(provider, messages, history, content, reasoning, calls,
                                  [{"role": "tool", "tool_call_id": "call_1", "content": "result"}], ["result"], tokens)
    assert "_pending_provider_data" not in history
    assert messages[0]["provider_data"]["items"] == OUTPUT
    assert "provider_data" not in calls[0]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sse_merges_completed_envelope_with_partial_or_out_of_order_done_events(asynchronous):
    events = [
        {"type": "response.output_item.done", "output_index": 2, "item": CALL},
        {"type": "response.output_item.done", "output_index": 0, "item": REASON},
        {"type": "response.completed", "response": {"status": "completed", "output": OUTPUT}},
    ]
    response = httpx.Response(200, text="\n\n".join("data: " + json.dumps(e) for e in events) + "\n\n")
    result = asyncio.run(async_collect_stream(response)) if asynchronous else collect_stream(response)
    assert result["output"] == OUTPUT
