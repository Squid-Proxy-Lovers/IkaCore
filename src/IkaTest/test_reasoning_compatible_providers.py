"""The current DeepSeek and Z.AI contracts require reasoning in tool requests."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_helpers_common import build_provider_request
from IkaModel.chat_interface.chat_interface import async_chat, chat, init_message_history

PROVIDERS = {
    "deepseek": ("deepseek-reasoner", "https://api.deepseek.com/chat/completions", "deepseek"),
    "zai": ("glm-4.7", "https://api.z.ai/api/coding/paas/v4/chat/completions", "openai"),
}
CALL = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_compatible_reasoning_tool_loop(provider, asynchronous):
    model_id, url, adapter = PROVIDERS[provider]
    model = BareBoneModel(model_id, "fixture", url, use_responses_api=False)
    model.agent_tools = [AgentTool("lookup", "lookup", "lookup", ToolArgs("object", "args", properties={}))]
    model.deepthinking = True
    history = init_message_history()
    messages = [{"role": "user", "content": "task"}]
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        message = {"content": "Checking.", "reasoning_content": "original unmodified thought", "tool_calls": [CALL]}
        if len(requests) > 1:
            message = {"content": "done", "reasoning_content": "final thought"}
        return httpx.Response(200, json={"choices": [{"message": message}]})

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
    assistant = {"role": "assistant", "content": "Checking.", "reasoning_content": "original unmodified thought",
                 "tool_calls": [CALL]}
    expected = deepcopy(requests[0])
    expected["messages"] += [assistant, {"role": "tool", "tool_call_id": "call_1", "content": "result"}]
    assert requests[1] == expected
    assert requests[1]["thinking"] == {"type": "enabled"}
    next_payload = build_provider_request(adapter, model, [{"role": "user", "content": "next"}], history)[2]
    assert next_payload["messages"].count(assistant) == 1
    assert {"role": "assistant", "content": "done", "reasoning_content": "final thought"} in next_payload["messages"]


def test_deepseek_no_tools_reasoning_is_accepted_but_not_used_by_api():
    # Current API docs: without tools, reasoning_content is optional and ignored.
    # Older R1-era advice prohibiting replay must not break modern thinking/tool use.
    from IkaModel.deepseek.deepseek import deepseek_fill_payload

    model = BareBoneModel("deepseek-reasoner", "fixture", PROVIDERS["deepseek"][1])
    messages = [{"role": "user", "content": "task"},
                {"role": "assistant", "content": "done", "reasoning_content": "prior thought"},
                {"role": "user", "content": "next"}]
    payload = deepseek_fill_payload(model, messages, init_message_history())
    assert "tools" not in payload
    assert payload["messages"][1]["reasoning_content"] == "prior thought"
