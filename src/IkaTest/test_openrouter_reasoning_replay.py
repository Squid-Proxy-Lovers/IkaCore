import asyncio
import json
import logging
from copy import deepcopy

import httpx
import pytest

from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_interface.chat_interface import async_chat, chat, init_message_history
from IkaModel.openrouter.openrouter import openrouter_fill_payload

DETAILS = [
    {"type": "reasoning.summary", "summary": "inspect", "index": 0, "id": "rs_1", "format": "anthropic-claude-v1"},
    {"type": "reasoning.text", "text": "thought", "signature": "opaque-signature", "index": 1, "id": "rs_2"},
    {"type": "reasoning.encrypted", "data": "opaque-cipher", "index": 2, "id": "rs_3"},
]
CALL = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
URL = "https://openrouter.ai/api/v1/chat/completions"


@pytest.mark.parametrize("fields", [{"reasoning_details": DETAILS},
                                   {"reasoning": "readable", "reasoning_details": DETAILS},
                                   {"reasoning_content": "readable", "reasoning_details": DETAILS}])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_openrouter_reasoning_details_survive_tool_loop(fields, asynchronous, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        message = {"content": "Checking.", "tool_calls": [CALL], **deepcopy(fields)}
        if len(requests) > 1:
            message = {"content": "done"}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    model = BareBoneModel("anthropic/claude-sonnet-4.5", "fixture", URL, reasoning_effort="high")
    model.agent_tools = [AgentTool("lookup", "lookup", "lookup", ToolArgs("object", "args", properties={}))]
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
    assistant = {"role": "assistant", "content": "Checking.", "tool_calls": [CALL], **fields}
    expected["messages"] += [assistant, {"role": "tool", "tool_call_id": "call_1", "content": "result"}]
    assert requests[1] == expected
    assert requests[1]["reasoning"] == {"effort": "high", "exclude": False}
    assert "reasoning_effort" not in requests[1]
    assert "opaque-cipher" in json.dumps(history)
    output = capsys.readouterr()
    assert "opaque-" not in output.out + output.err + caplog.text
    resumed = openrouter_fill_payload(model, [{"role": "user", "content": "next"}], history)
    assert resumed["messages"].count(assistant) == 1


def test_explicit_openrouter_effort_overrides_exclusion_for_tools():
    model = BareBoneModel("qwen/qwen3.6-plus", "fixture", URL, reasoning_effort="low")
    model.agent_tools = [AgentTool("lookup", "lookup", "lookup", ToolArgs("object", "args"))]
    assert openrouter_fill_payload(model, [], init_message_history())["reasoning"] == {"effort": "low", "exclude": False}


def test_openrouter_final_details_replay_without_a_tool_call():
    from IkaModel.chat_helpers_common import parse_provider_response
    from IkaModel.chat_interface.chat_response import _record_model_message

    history = init_message_history()
    data = {"choices": [{"message": {"content": "done", "reasoning_details": DETAILS}}]}
    content, reasoning, _, tokens = parse_provider_response("openrouter", data, "anthropic/claude", history)
    _record_model_message(history, content, tokens, reasoning)
    model = BareBoneModel("anthropic/claude", "fixture", URL)
    payload = openrouter_fill_payload(model, [{"role": "user", "content": "next"}], history)
    assert payload["messages"][0] == {"role": "assistant", "content": "done", "reasoning_details": DETAILS}
