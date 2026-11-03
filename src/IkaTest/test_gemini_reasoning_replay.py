import asyncio
import json
import logging
from copy import deepcopy

import httpx
import pytest

from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_interface.chat_interface import async_chat, chat, init_message_history
from IkaModel.gemini.google import gemini_fill_payload

URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"


@pytest.mark.parametrize("sig_key", ["thoughtSignature", "thought_signature"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_gemini_keeps_each_signed_part_at_its_original_position(sig_key, asynchronous, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    parts = [
        {"text": "private thought summary", "thought": True, sig_key: "opaque-thought-signature"},
        {"text": "Checking.", sig_key: "opaque-text-signature"},
        {"functionCall": {"name": "lookup", "args": {"key": "first"}}, sig_key: "opaque-call-signature"},
        {"functionCall": {"name": "lookup", "args": {"key": "other"}}},
    ]
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        reply = deepcopy(parts) if len(requests) == 1 else [{"text": "done"}]
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": reply}}]})

    model = BareBoneModel("gemini-2.5-pro", "fixture", URL, parallel_tool_calls=True)
    model.agent_tools = [AgentTool("lookup", "lookup", "lookup", ToolArgs("object", "args", properties={"key": {"type": "string"}}, required=[]), parallel=True)]
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
    expected = deepcopy(requests[0])
    expected["contents"] += [
        {"role": "model", "parts": parts},
        {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {"result": "result"}}}] * 2},
    ]
    assert requests[1] == expected
    assert "opaque-call-signature" in json.dumps(history)
    output = capsys.readouterr()
    assert "opaque-" not in output.out + output.err + caplog.text
    resumed = gemini_fill_payload(model, [{"role": "user", "content": "next"}], history)
    assert resumed["contents"].count({"role": "model", "parts": parts}) == 1


def test_thought_text_is_not_user_visible_and_final_signature_is_persisted():
    from IkaModel.chat_helpers_common import parse_provider_response
    from IkaModel.chat_interface.chat_response import _record_model_message

    history = init_message_history()
    parts = [{"thought": True, "text": "private"}, {"text": "done", "thoughtSignature": "opaque-final"}]
    content, reasoning, calls, tokens = parse_provider_response(
        "gemini", {"candidates": [{"content": {"parts": parts}}]}, "gemini-2.5-pro", history)
    assert content == "done" and reasoning == "private" and not calls
    _record_model_message(history, content, tokens, reasoning)
    model = BareBoneModel("gemini-2.5-pro", "fixture", URL)
    payload = gemini_fill_payload(model, [{"role": "user", "content": "next"}], history)
    assert payload["contents"][0] == {"role": "model", "parts": parts}
