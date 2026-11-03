import asyncio

import pytest

from IkaModel.chat_interface import tool_execution_async as asynchronous
from IkaModel.chat_interface import tool_execution_sync as synchronous
from IkaModel.request_control import request_controls


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("kind", ["execution", "validation", "missing", "timeout"])
def test_every_direct_tool_error_obeys_output_cap(monkeypatch, async_mode, kind):
    name = "lookup" if kind in {"execution", "validation"} else "x" * 10000

    def fail(args):
        if kind == "execution":
            raise ValueError("x" * 10000)
        raise TimeoutError("timeout")

    executors = {} if kind == "missing" else {name: fail}
    if kind == "validation":
        def invalid(*args):
            raise ValueError("x" * 10000)
        monkeypatch.setattr(synchronous, "validate_tool_args", invalid)
        monkeypatch.setattr(asynchronous, "validate_tool_args", invalid)
    with request_controls(tool_result_max_chars=128):
        result = asyncio.run(asynchronous.async_execute_tool(name, {}, executors)) if async_mode else synchronous.execute_tool(name, {}, executors)
    assert len(result) <= 128


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("provider", ["openai", "openai_responses", "codex", "deepseek", "openrouter", "anthropic", "gemini"])
def test_planning_errors_are_capped_before_provider_formatting(async_mode, provider):
    calls = [{"id": "schema", "function": {"name": "lookup", "arguments": "{}"}}]
    metadata = {"lookup": {"parameters": {"required": ["query"], "properties": {
        "query": {"type": "string", "description": "x" * 10000}}}}}
    with request_controls(tool_result_max_chars=128):
        args = (calls, {"lookup": lambda args: "unused"}, provider)
        kwargs = {"tool_metadata": metadata}
        result = asyncio.run(asynchronous.async_execute_tool_calls(*args, **kwargs)) if async_mode else synchronous.execute_tool_calls(*args, **kwargs)
    assert len(result[1][0]) <= 128
    if provider == "gemini":
        assert len(result[0][0]["functionResponse"]["response"]["result"]) <= 128
    elif provider == "anthropic":
        assert len(result[0][0]["content"][0]["content"]) <= 128
    else:
        assert len(result[0][0]["content"]) <= 128
