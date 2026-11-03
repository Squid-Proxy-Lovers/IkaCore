import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from IkaModel.anthropic.chat_helpers_anthropic import append_anthropic_tool_messages
from IkaModel.base import AgentTool, ToolArgs
from IkaModel.codex import codex_responses as codex
from IkaModel.deepseek.chat_helpers_deepseek import (
    _unwrap_double_encoded_args,
    append_deepseek_tool_messages,
    build_deepseek_request,
    parse_deepseek_response,
)
from IkaModel.gemini.chat_helpers_gemini import append_gemini_tool_messages
from IkaModel.openrouter.chat_helpers_openrouter import (
    append_openrouter_tool_messages,
    build_openrouter_request,
    parse_openrouter_response,
)


def _history():
    return {
        "system": {"message": "system", "tokens": 0},
        "first_input": {"message": "first", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def _model(**overrides):
    defaults = {
        "model_id": "codex-mini-latest",
        "api_key": "test-key",
        "api_url": "https://example.test",
        "system_prompt": "system",
        "agent_tools": [],
        "parallel_tool_calls": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_deepseek_unwraps_only_nested_object_or_array_strings():
    wrapped = json.dumps(
        {
            "model": json.dumps({"product_type": "sku", "price": 3}),
            "items": json.dumps([{"id": 1}]),
            "number": "3",
            "plain": "text",
        }
    )

    unwrapped = json.loads(_unwrap_double_encoded_args(wrapped))

    assert unwrapped["model"] == {"product_type": "sku", "price": 3}
    assert unwrapped["items"] == [{"id": 1}]
    assert unwrapped["number"] == "3"
    assert unwrapped["plain"] == "text"
    assert _unwrap_double_encoded_args("") == ""
    assert _unwrap_double_encoded_args({"not": "a string"}) == {"not": "a string"}
    assert _unwrap_double_encoded_args("[1, 2]") == "[1, 2]"
    assert _unwrap_double_encoded_args('{"plain":"text"}') == '{"plain":"text"}'


def test_deepseek_parse_repairs_tool_arguments_and_append_handles_optional_fields():
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning_content": "thoughts",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "submit",
                                "arguments": json.dumps({"payload": json.dumps({"ok": True})}),
                            },
                        },
                        {"id": "bad", "function": "not-a-dict"},
                    ],
                }
            }
        ],
        "usage": {"total_tokens": 9},
    }

    content, reasoning, tool_calls, tokens = parse_deepseek_response(data, "deepseek-v4-flash")

    assert content == ""
    assert reasoning == "thoughts"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"payload": {"ok": True}}
    assert tokens == 9

    messages = []
    history = _history()
    append_deepseek_tool_messages(
        messages,
        history,
        content="answer",
        reasoning_content="thoughts",
        executed_tool_call_list=[tool_calls[0]],
        tool_messages=[{"role": "tool", "content": "ok", "tool_call_id": "call_1"}],
        tokens=9,
        repeated_warning_msg="do not repeat",
    )

    assert messages[0]["reasoning_content"] == "thoughts"
    assert messages[0]["tool_calls"] == [tool_calls[0]]
    assert messages[-1] == {"role": "user", "content": "do not repeat"}
    assert {entry["type"] for entry in history["messages"].values()} == {"assistant_with_tools", "tool"}


def test_deepseek_request_passes_filtered_agent_tools_and_auth_headers():
    model = _model(api_url="https://api.deepseek.com/chat/completions", agent_tools=[MagicMock(name="tool")])
    with patch("IkaModel.deepseek.chat_helpers_deepseek.agent_tools_for_payload", return_value=["filtered"]) as tools:
        with patch("IkaModel.deepseek.chat_helpers_deepseek.deepseek_fill_payload", return_value={"payload": True}) as fill:
            url, headers, payload = build_deepseek_request(model, [{"role": "user", "content": "hi"}], _history())

    assert url == "https://api.deepseek.com/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert payload == {"payload": True}
    tools.assert_called_once_with(model)
    fill.assert_called_once()
    assert fill.call_args.kwargs["agent_tools"] == ["filtered"]


def test_openrouter_parse_errors_reasoning_logging_and_append_paths(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="OpenRouter returned error"):
        parse_openrouter_response(
            {
                "error": {
                    "message": "upstream failed",
                    "metadata": {"provider_name": "ProviderX", "raw": "raw detail\n"},
                }
            },
            "openrouter/model",
        )
    with pytest.raises(ValueError, match="missing 'choices'"):
        parse_openrouter_response({}, "openrouter/model")

    log_path = tmp_path / "glm_reasoning.log"
    monkeypatch.setenv("GLM_REASONING_LOG", str(log_path))
    content, reasoning, tool_calls, tokens = parse_openrouter_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "reasoning": "glm reasoning",
                        "tool_calls": [{"id": "call_1"}],
                    }
                }
            ],
            "usage": {"total_tokens": 7, "cost": 0.02},
        },
        "z-ai/glm-4.5",
    )

    assert (content, reasoning, tool_calls, tokens) == ("answer", "glm reasoning", [{"id": "call_1"}], 7)
    assert "glm reasoning" in log_path.read_text()

    messages = []
    history = _history()
    append_openrouter_tool_messages(
        messages,
        history,
        content="answer",
        reasoning_content="reasoning",
        executed_tool_call_list=[{"id": "call_1"}],
        tool_messages=[{"role": "tool", "content": "ok", "tool_call_id": "call_1"}],
        tokens=7,
        repeated_warning_msg="warning",
    )
    assert messages[0]["reasoning_content"] == "reasoning"
    assert messages[-1] == {"role": "user", "content": "warning"}
    assert {entry["type"] for entry in history["messages"].values()} == {"assistant_with_tools", "tool"}


def test_openrouter_request_includes_optional_headers_and_provider_options():
    model = _model(
        api_url="https://openrouter.ai/api/v1/chat/completions",
        http_referer="https://app.example",
        x_title="IkaCore",
        openrouter_plugins=[{"id": "web"}],
        openrouter_response_format={"type": "json_object"},
    )
    with patch("IkaModel.openrouter.chat_helpers_openrouter.agent_tools_for_payload", return_value=["filtered"]) as tools:
        with patch("IkaModel.openrouter.chat_helpers_openrouter.openrouter_fill_payload", return_value={"payload": True}) as fill:
            url, headers, payload = build_openrouter_request(model, [{"role": "user", "content": "hi"}], _history())

    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["HTTP-Referer"] == "https://app.example"
    assert headers["X-Title"] == "IkaCore"
    assert payload == {"payload": True}
    tools.assert_called_once_with(model)
    assert fill.call_args.kwargs["plugins"] == [{"id": "web"}]
    assert fill.call_args.kwargs["response_format"] == {"type": "json_object"}
    assert fill.call_args.kwargs["agent_tools"] == ["filtered"]


def test_anthropic_and_gemini_appenders_tolerate_malformed_tool_arguments():
    malformed_call = {
        "id": "call_bad",
        "name": "lookup",
        "function": {"name": "lookup", "arguments": '{"query":'},
    }
    anthropic_messages = []
    anthropic_history = _history()
    append_anthropic_tool_messages(
        anthropic_messages,
        anthropic_history,
        content="",
        reasoning_content=None,
        executed_tool_call_list=[malformed_call],
        tool_messages=[{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_bad", "content": "bad"}]}],
        tokens=3,
        repeated_warning_msg="fix args",
    )

    assert anthropic_messages[0]["content"][1]["input"] == {}
    assert anthropic_messages[-1]["content"][0]["text"] == "fix args"
    assert {entry["type"] for entry in anthropic_history["messages"].values()} == {"assistant_with_tools", "tool"}

    gemini_messages = []
    gemini_history = _history()
    append_gemini_tool_messages(
        gemini_messages,
        gemini_history,
        content="",
        reasoning_content=None,
        executed_tool_call_list=[malformed_call],
        tool_results=["bad"],
        tokens=3,
        repeated_warning_msg="fix args",
        format_gemini_results_fn=lambda _calls, _results: [{"functionResponse": {"name": "lookup", "response": {}}}],
    )

    assert gemini_messages[0]["parts"][1]["functionCall"]["args"] == {}
    assert gemini_messages[-1] == {"role": "user", "parts": [{"text": "fix args"}]}
    assert {entry["type"] for entry in gemini_history["messages"].values()} == {"assistant_with_tools", "tool"}


def test_codex_content_wrappers_and_message_item_conversion_cover_mixed_shapes():
    assert codex._wrap_user_content("hi") == [{"type": "input_text", "text": "hi"}]
    assert codex._wrap_user_content(["hi", {"type": "input_text", "text": "there"}, 3]) == [
        {"type": "input_text", "text": "hi"},
        {"type": "input_text", "text": "there"},
        {"type": "input_text", "text": "3"},
    ]
    assert codex._wrap_assistant_content(["out", {"type": "output_text", "text": "there"}, 4]) == [
        {"type": "output_text", "text": "out"},
        {"type": "output_text", "text": "there"},
        {"type": "output_text", "text": "4"},
    ]

    assert codex._codex_items_for_message("raw")[0]["content"][0]["text"] == "raw"
    assert codex._codex_items_for_message({"role": "tool", "content": {"ok": True}, "tool_call_id": "call_1"}) == [
        {"type": "function_call_output", "call_id": "call_1", "output": '{"ok": true}'}
    ]
    assert codex._codex_items_for_message({"role": "assistant", "content": ""}) == []
    assert codex._codex_items_for_message(
        {"role": "assistant", "tool_calls": [_tool_call := {"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}]}
    ) == [{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}]
    assert codex._codex_items_for_message(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "anthropic_call", "content": "result"},
                {"type": "note", "value": 1},
            ],
        }
    )[0] == {"type": "function_call_output", "call_id": "anthropic_call", "output": "result"}
    assert _tool_call["id"] == "call_1"


def test_codex_payload_context_skips_duplicate_first_message_and_selects_tool_choice():
    model = _model(reasoning_effort="high", parallel_tool_calls=True)
    search = AgentTool("search", "search", "Search", ToolArgs("object", "payload"), required=True)
    submit = AgentTool("submit", "submit", "Submit", ToolArgs("object", "payload"), required=True)
    agent_end = AgentTool("agent_end", "agent_end", "End", ToolArgs("object", "payload"), required=True)
    history = _history()
    history["summary"]["message"] = "summary"

    payload = codex.codex_responses_fill_payload(
        model,
        [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
        history,
        agent_tools=[search, submit, agent_end],
    )

    assert payload["instructions"] == "system"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is True
    assert [item["content"][0]["text"] for item in payload["input"] if item.get("role") == "user"] == [
        "first",
        "second",
    ]

    model.forced_tool_name = "search"
    forced = codex.codex_responses_fill_payload(model, [{"role": "user", "content": "new"}], history, agent_tools=[search])
    assert forced["tool_choice"] == {"type": "function", "name": "search"}

    auto = codex.codex_responses_fill_payload(model, [{"role": "user", "content": "new"}], history, agent_tools=[agent_end])
    assert auto["tool_choice"] == "auto"
