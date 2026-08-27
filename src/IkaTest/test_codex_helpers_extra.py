import json
from unittest.mock import patch

import pytest

from IkaModel.codex.chat_helpers_codex import (
    _CodexRetryableStreamError,
    _collect_stream,
    _iter_sse,
    _parse_retry_after,
    _retry_codex_exception,
    append_codex_tool_messages,
    is_codex_url,
    parse_codex_response,
    request_codex,
)


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        yield from self._lines


def _frame(event, obj):
    return [f"event: {event}", f"data: {json.dumps(obj)}", ""]


def test_codex_url_accepts_a_scoped_gateway_endpoint():
    assert is_codex_url("http://172.30.0.1:18081/backend-api/codex/responses")
    assert not is_codex_url("https://api.openai.com/v1/responses")


def test_retry_after_falls_back_when_http_date_parser_raises():
    with patch("IkaModel.codex.chat_helpers_codex.email.utils.parsedate_to_datetime", side_effect=ValueError("bad")):
        assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", default=9) == 9.0


def test_iter_sse_skips_none_empty_done_and_non_json_lines():
    lines = [None, "", "event: response.completed", "data:", "data: [DONE]", "data: not-json", "data: {\"ok\": true}", ""]

    assert list(_iter_sse(_FakeStreamResponse(lines))) == [("response.completed", {"ok": True})]


def test_collect_stream_ignores_non_dict_objects_and_non_dict_envelopes_or_items():
    lines = (
        ["data: 1", ""]
        + _frame("response.created", {"type": "response.created", "response": "not-a-dict"})
        + _frame("response.in_progress", {"type": "response.in_progress", "response": {"id": "r1", "output": []}})
        + _frame("response.output_item.done", {"type": "response.output_item.done", "item": "not-a-dict"})
        + _frame("response.output_text.delta", {"type": "response.output_text.delta", "delta": 7})
        + _frame("response.completed", {"type": "response.completed", "response": {"id": "r1", "output": []}})
    )

    envelope = _collect_stream(_FakeStreamResponse(lines))

    assert envelope == {"id": "r1", "output": []}


def test_retry_codex_exception_raises_on_final_attempt_and_request_loop_handles_none_exhaustion():
    error = _CodexRetryableStreamError("transient")
    with pytest.raises(_CodexRetryableStreamError):
        _retry_codex_exception("message %s", backoff=1, attempt=1, max_retries=2, error=error)

    with patch("IkaModel.codex.chat_helpers_codex._request_codex_once", return_value=None):
        with pytest.raises(RuntimeError, match="without specific exception"):
            request_codex("u", {}, {"stream": False}, timeout=1, max_retries=2, wait_seconds=1)


def test_parse_codex_response_handles_reasoning_fallback_text_and_usage_fallback():
    content, reasoning, tool_calls, tokens = parse_codex_response(
        {
            "output": [
                {"type": "reasoning", "summary": [{"text": "think"}, {"text": "more"}, "bad"]},
                {"type": "function_call", "id": "fallback_id", "name": "lookup", "arguments": "{\"q\":\"x\"}"},
            ],
            "output_text": "fallback text",
            "usage": {"input_tokens": 2, "output_tokens": 3},
        },
        "codex",
    )

    assert content == "fallback text"
    assert reasoning == "think\nmore"
    assert tool_calls == [
        {
            "id": "fallback_id",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"},
        }
    ]
    assert tokens == 5


def test_append_codex_tool_messages_records_reasoning_tools_warning_and_tool_history():
    messages = []
    history = {"messages": {}}
    executed = [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}]
    tool_messages = [{"role": "tool", "content": "ok", "tool_call_id": "call_1"}]

    append_codex_tool_messages(
        messages,
        history,
        content="answer",
        reasoning_content="reasoning",
        executed_tool_call_list=executed,
        tool_messages=tool_messages,
        tokens=7,
        repeated_warning_msg="warning",
    )

    assert messages[0]["reasoning_content"] == "reasoning"
    assert messages[0]["tool_calls"] == executed
    assert messages[-1] == {"role": "user", "content": "warning"}
    assert {entry["type"] for entry in history["messages"].values()} == {"assistant_with_tools", "tool"}
    assert is_codex_url(None) is False
    assert is_codex_url("https://chatgpt.com/backend-api/codex/responses") is True
