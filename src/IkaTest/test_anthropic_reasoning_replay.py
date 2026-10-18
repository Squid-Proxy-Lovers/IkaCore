import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from IkaCore import IkaBaseAgent, ProviderConfig
from IkaModel.anthropic.claude import anthropic_fill_payload
from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_interface.chat_interface import async_chat, chat, init_message_history

BLOCKS = [
    {"type": "thinking", "thinking": "inspect fixture", "signature": "opaque-signature"},
    {"type": "text", "text": "Checking."},
    {"type": "redacted_thinking", "data": "opaque-redacted-data"},
    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
]
URL = "https://api.anthropic.com/v1/messages"


def thinking_model(**kwargs):
    model = BareBoneModel("claude-sonnet-4-5", "fixture", URL, thinking_budget_tokens=1024, max_tokens=4096, **kwargs)
    model.agent_tools = [AgentTool("lookup", "lookup", "lookup", ToolArgs("object", "args", properties={}), required=True)]
    return model


@pytest.mark.parametrize("asynchronous", [False, True])
def test_signed_and_redacted_blocks_replayed_verbatim(asynchronous, capsys):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        blocks = deepcopy(BLOCKS) if len(requests) == 1 else [{"type": "text", "text": "done"}]
        return httpx.Response(200, json={"content": blocks, "usage": {"input_tokens": 4, "output_tokens": 2}})

    history = init_message_history()
    model = thinking_model()
    model.forced_tool_name = "lookup"
    kwargs = {"tool_executors": {"lookup": lambda args: "result"}, "max_tool_rounds": 2}
    messages = [{"role": "user", "content": "task"}]
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
    expected["messages"] += [
        {"role": "assistant", "content": BLOCKS},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "result"}]},
    ]
    assert requests[1] == expected
    assert expected["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert expected["max_tokens"] > 1024
    assert expected["tool_choice"]["type"] == "auto"
    assert not {"temperature", "top_k", "top_p"} & expected.keys()
    encoded = json.dumps(history)
    assert "opaque-signature" in encoded and "opaque-redacted-data" in encoded
    output = capsys.readouterr()
    assert "opaque-signature" not in output.out + output.err
    assert "opaque-redacted-data" not in output.out + output.err
    resumed = anthropic_fill_payload(model, [{"role": "user", "content": "next"}], history)
    assert {"role": "assistant", "content": BLOCKS} in resumed["messages"]


@pytest.mark.parametrize("budget", [0, 1023, -1, True, "1024", 4096, 5000])
def test_invalid_thinking_budget_fails_before_network(budget):
    model = thinking_model()
    model.thinking_budget_tokens = budget
    with pytest.raises(ValueError, match="thinking_budget_tokens"):
        anthropic_fill_payload(model, [], init_message_history())


def test_haiku_cap_validates_effective_max_tokens():
    model = thinking_model()
    model.model_id = "claude-haiku-4-5"
    model.max_tokens = 20000
    model.thinking_budget_tokens = 4096
    with pytest.raises(ValueError, match="after model caps"):
        anthropic_fill_payload(model, [], init_message_history())


def test_thinking_is_explicit_and_config_reaches_cloned_model():
    agent = IkaBaseAgent("fixture", "fixture", "task", model_id="claude-sonnet-4-5", api_key="fixture",
                         provider=ProviderConfig(thinking_budget_tokens=1024, max_tokens=4096))
    clone = None
    try:
        model = agent.chat.get_barebone("instructions", [])
        assert model.thinking_budget_tokens == 1024
        clone = agent.clone_for_run()
        assert clone.config.provider.thinking_budget_tokens == 1024
        assert clone.chat.get_barebone("instructions", []).thinking_budget_tokens == 1024
        default = BareBoneModel("claude-sonnet-4-5", "fixture", URL)
        assert "thinking" not in anthropic_fill_payload(default, [], init_message_history())
    finally:
        agent.shutdown()
        if clone:
            clone.shutdown()
