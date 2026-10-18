import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_interface.chat_request import init_message_history
from IkaModel.chat_interface.chat_runtime import async_chat, chat
from IkaModel.summarization import async_summarise_message_history, parse_summary_response, run_summarization

ROUTES = {
    "openai": ("gpt-4o", "https://api.openai.com/v1/chat/completions", "length"),
    "openai_responses": ("gpt-4o", "https://api.openai.com/v1/responses", "max_output_tokens"),
    "deepseek": ("deepseek-reasoner", "https://api.deepseek.com/chat/completions", "length"),
    "openrouter": ("openai/gpt-4o", "https://openrouter.ai/api/v1/chat/completions", "length"),
    "anthropic": ("claude-sonnet-4-5", "https://api.anthropic.com/v1/messages", "max_tokens"),
    "gemini": ("gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent", "MAX_TOKENS"),
    "codex": ("gpt-5.6-sol", "https://chatgpt.com/backend-api/codex/responses", "max_output_tokens"),
}


def response_data(provider, truncated, tools=False):
    text = "partial answer" if truncated else "checking"
    usage = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    call = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    if provider in {"codex", "openai_responses"}:
        output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
        if tools:
            output.append({"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"})
        return {"status": "incomplete" if truncated else "completed", "output": output, "usage": usage,
                "incomplete_details": {"reason": "max_output_tokens"} if truncated else None}
    if provider == "anthropic":
        content = [{"type": "text", "text": text}]
        if tools:
            content.append({"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}})
        return {"content": content, "usage": usage, "stop_reason": "max_tokens" if truncated else "tool_use"}
    if provider == "gemini":
        parts = [{"text": text}]
        if tools:
            parts.append({"functionCall": {"name": "lookup", "args": {}}})
        return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "MAX_TOKENS" if truncated else "STOP"}],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5}}
    message = {"role": "assistant", "content": text}
    if tools:
        message["tool_calls"] = [call]
    return {"choices": [{"message": message, "finish_reason": "length" if truncated else "tool_calls"}], "usage": usage}


def http_response(provider, data):
    if provider == "codex":
        return httpx.Response(200, text="event: response.incomplete\ndata: " + json.dumps({"type": "response.incomplete", "response": data}) + "\n\n")
    return httpx.Response(200, json=deepcopy(data))


@pytest.mark.parametrize("provider", [p for p in ROUTES if p != "codex"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("followup", [False, True])
def test_truncated_turn_is_interrupted_with_reason_and_never_executes_partial_tools(provider, asynchronous, followup):
    name, url, reason = ROUTES[provider]
    model = BareBoneModel(name, "fixture", url, use_responses_api=provider == "openai_responses")
    model.agent_tools = [AgentTool("lookup", "lookup", "lookup", ToolArgs("object", "args"))]
    calls = []
    executions = []

    def respond(request):
        calls.append(request)
        return http_response(provider, response_data(provider, not followup or len(calls) > 1, tools=True))

    kwargs = {"tool_executors": {"lookup": lambda args: executions.append(True) or "result"}}
    messages = [{"role": "user", "content": "task"}]
    if asynchronous:
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                return await async_chat(model, messages, init_message_history(), client=client, **kwargs)
        result = asyncio.run(run())
    else:
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            result = chat(model, messages, init_message_history(), client=client, **kwargs)
    assert len(calls) == (2 if followup else 1)
    assert len(executions) == (1 if followup else 0)
    assert result["content"] == "partial answer"
    assert result["interrupted"] is True
    assert result["interrupt_data"]["type"] == "provider_response_truncated"
    assert result["interrupt_data"]["reason"] == reason
    assert result["usage"]["total_tokens"] == (10 if followup else 5)


@pytest.mark.parametrize("provider", [p for p in ROUTES if p != "openai_responses"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_truncated_compaction_preserves_entire_history(provider, asynchronous):
    name, url, _ = ROUTES[provider]
    model = BareBoneModel(name, "fixture", url, use_responses_api=False)
    history = init_message_history()
    history["first_input"]["message"] = "task"
    history["messages"]["old"] = {"message": "UNIQUE IMPORTANT OBSERVATION", "tokens": 10}
    before = deepcopy(history)
    transport = httpx.MockTransport(lambda request: http_response(provider, response_data(provider, True)))
    if asynchronous:
        async def run():
            async with httpx.AsyncClient(transport=transport) as client:
                return await async_summarise_message_history(model, history, client=client)
        summary = asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as client:
            summary = run_summarization(model, history, client=client)
    assert summary == ""
    assert history == before


@pytest.mark.parametrize("asynchronous", [False, True])
def test_codex_incomplete_terminal_event_is_already_rejected(asynchronous):
    name, url, _ = ROUTES["codex"]
    model = BareBoneModel(name, "fixture", url)
    transport = httpx.MockTransport(lambda request: http_response("codex", response_data("codex", True)))
    if asynchronous:
        async def run():
            async with httpx.AsyncClient(transport=transport) as client:
                return await async_chat(model, [{"role": "user", "content": "task"}], client=client)
        with pytest.raises(RuntimeError, match=r"incomplete.*max_output_tokens"):
            asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as client, pytest.raises(RuntimeError, match=r"incomplete.*max_output_tokens"):
            chat(model, [{"role": "user", "content": "task"}], client=client)


def test_direct_responses_summary_parser_rejects_incomplete_envelope():
    with pytest.raises(RuntimeError, match=r"truncated.*max_output_tokens"):
        parse_summary_response("codex", httpx.Response(200, json=response_data("codex", True)))
