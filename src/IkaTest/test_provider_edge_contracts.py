"""Provider edge shapes, malformed history and optional request fields."""

import importlib
import json
from unittest.mock import Mock

import pytest

from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_helpers_common import append_provider_tool_messages, build_provider_request, parse_provider_response
from IkaModel.chat_interface.chat_request import init_message_history
from IkaModel.codex import codex_responses
from IkaModel.codex.codex_stream import CodexRetryableStreamError, collect_stream
from IkaModel.openai import openai_responses


@pytest.mark.parametrize("provider", ["openai", "openai_responses", "anthropic", "deepseek", "gemini", "openrouter", "codex"])
def test_append_empty_tool_turn_with_warning_preserves_user_instruction(provider):
    messages, history = [], init_message_history()
    append_provider_tool_messages(provider, messages, history, "text", "reason", [], [], [], 0, "retry warning")
    assert "retry warning" in json.dumps(messages)
    assert "text" in json.dumps(messages)


@pytest.mark.parametrize("module", ["openai.chat_helpers_openai", "deepseek.chat_helpers_deepseek", "openrouter.chat_helpers_openrouter", "codex.chat_helpers_codex"])
def test_provider_token_coercion_has_stable_bool_and_invalid_defaults(module):
    helper = importlib.import_module("IkaModel." + module)
    assert helper._token_count(True) == 1
    assert helper._token_count("bad") == 0


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
@pytest.mark.parametrize("raw", ["{bad", "[]", {"value": 1}])
def test_provider_tool_argument_object_decoder_rejects_nonobjects(provider, raw):
    helper = importlib.import_module(f"IkaModel.{provider}.chat_helpers_{provider}")
    expected = raw if isinstance(raw, dict) else {}
    assert helper._decode_tool_args_object(raw) == expected


@pytest.mark.parametrize("raw", ["", "[1]", '{"broken":"{bad","scalar":1,"empty":""}'])
def test_deepseek_nested_argument_repair_keeps_unparseable_values(raw):
    from IkaModel.deepseek.chat_helpers_deepseek import _unwrap_double_encoded_args
    assert _unwrap_double_encoded_args(raw) == raw


def test_gemini_function_signatures_keep_both_wire_spellings():
    from IkaModel.gemini.chat_helpers_gemini import append_gemini_tool_messages, parse_gemini_response
    for spelling in ("thoughtSignature", "thought_signature"):
        data = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "fixture", "args": {}}, spelling: "opaque"}, {"inlineData": {}}]}}]}
        _, _, calls, _ = parse_gemini_response(data, "gemini")
        assert calls[0]["gemini_thought_signature_key"] == spelling
        messages = []
        append_gemini_tool_messages(messages, {}, "", None, calls, [], 0)
        assert messages[0]["parts"][1][spelling] == "opaque"
        assert len(messages) == 1  # no formatter means no invented result


def test_openrouter_reasoning_log_write_failure_preserves_response(monkeypatch, tmp_path):
    from IkaModel.openrouter.chat_helpers_openrouter import parse_openrouter_response
    monkeypatch.setenv("GLM_REASONING_LOG", str(tmp_path))
    response = {"choices": [{"message": {"content": "answer", "reasoning": "reason"}}], "usage": {"total_tokens": True}}
    assert parse_openrouter_response(response, "glm-fixture")[:2] == ("answer", "reason")


@pytest.mark.parametrize("provider", ["unsupported"])
def test_unknown_provider_dispatch_is_explicit(provider):
    with pytest.raises(ValueError, match="Unsupported provider"):
        build_provider_request(provider, None, [], {})
    assert parse_provider_response(provider, {}, "fixture") == ("", None, [], 0)
    messages = []
    append_provider_tool_messages(provider, messages, {}, "", None, [], [], [], 0)
    assert messages == []


def test_openai_response_url_and_output_text_fallback():
    from IkaModel.openai.chat_helpers_openai import build_openai_responses_request, parse_openai_responses_response
    for url, expected in [("https://proxy.test/v1/chat/completions", "https://proxy.test/v1/responses"), ("https://proxy.test", "https://proxy.test/v1/responses")]:
        model = BareBoneModel("gpt-4o", "test", url, suppress_init_output=True)
        assert build_openai_responses_request(model, [], {})[0] == expected
    assert parse_openai_responses_response({"output": [{"type": "other"}], "output_text": "fallback"}, "fixture")[0] == "fallback"


@pytest.mark.parametrize("module", [openai_responses, codex_responses])
def test_responses_content_wrapping_preserves_mixed_block_order(module):
    if module is codex_responses:
        assert module._wrap_user_content(["text", {"type": "input_image", "image_url": "x"}, 7]) == [{"type": "input_text", "text": "text"}, {"type": "input_image", "image_url": "x"}, {"type": "input_text", "text": "7"}]
        assert module._wrap_assistant_content(["text", 7]) == [{"type": "output_text", "text": "text"}, {"type": "output_text", "text": "7"}]
        assert module._wrap_assistant_content(7) == [{"type": "output_text", "text": "7"}]
    else:
        assert module._responses_items_for_message(7)[0]["content"] == "7"
        assert module._responses_items_for_message({"role": "developer", "content": "policy"})[0]["role"] == "developer"
        assert module.openai_responses_fill_payload(BareBoneModel("gpt-4o", "test", "url", suppress_init_output=True), [], None)["input"] == []


@pytest.mark.parametrize("provider", ["openai", "openai_responses", "anthropic"])
def test_multiple_required_tools_and_explicit_tool_override(provider):
    tool_list = [AgentTool(str(i), "tool" + str(i), "fixture", ToolArgs("object", "args"), required=True) for i in range(2)]
    model = BareBoneModel("claude-sonnet-4" if provider == "anthropic" else "gpt-4o", "test", "https://example.test", suppress_init_output=True)
    model.agent_tools = tool_list
    payload = build_provider_request(provider, model, [{"role": "user", "content": "task"}], init_message_history())[2]
    assert payload["tool_choice"] in ("required", {"type": "required", "disable_parallel_tool_use": True}, {"type": "required"})
    model.forced_tool_name = "tool1"
    payload = build_provider_request(provider, model, [], init_message_history())[2]
    assert "tool1" in json.dumps(payload["tool_choice"])


@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
def test_legacy_history_shapes_ignore_bad_json_and_preserve_nontext_entries(provider):
    model = BareBoneModel("claude-haiku-4" if provider == "anthropic" else "deepseek-chat", "test", "url", suppress_init_output=True)
    history = init_message_history()
    history["messages"] = {"bad1": {"type": "assistant_with_tools", "message": "{"}, "bad2": {"type": "tool", "message": "{"}, "plain": {"message": 7}, "user": {"type": "hitl_input", "message": 8}}
    payload = build_provider_request(provider, model, [{"content": "bare"}, {"other": 7}, 99], history)[2]
    assert "bare" in json.dumps(payload) and "99" in json.dumps(payload)
    assert payload["messages"][0]["content"] == "7"


@pytest.mark.parametrize("status,error", [("failed", RuntimeError), ("incomplete", RuntimeError), ("in_progress", CodexRetryableStreamError)])
def test_codex_completed_event_with_nonterminal_status_is_never_accepted(status, error):
    import httpx
    raw = {"type": "response.completed", "response": {"status": status}}
    response = httpx.Response(200, text="data: " + json.dumps(raw) + "\n\n")
    with pytest.raises(error):
        collect_stream(response)


def test_defensive_message_wrappers_keep_scalar_content_and_required_tool_choice():
    from IkaModel.anthropic.claude import _message_content, _without_blank_text
    from IkaModel.deepseek.deepseek import _message_content as deepseek_content
    from IkaModel.openai.openai import _history_message, _normalise_openai_message
    from IkaModel.tool_schema import _ensure_object_root

    assert _without_blank_text({"role": "assistant", "content": None}) == {"role": "assistant", "content": None}
    assert _message_content(7) == deepseek_content(7) == "7"
    assert _normalise_openai_message(7) == {"role": "user", "content": "7"}
    assert _history_message({"system": "bad"}, "system") == ""
    assert openai_responses._responses_items_for_message(7)[0]["content"] == "7"
    assert codex_responses._wrap_user_content(7)[0]["text"] == "7"
    assert codex_responses._default_message_history()["messages"] == {}
    assert codex_responses._codex_items_for_message({"role": "developer", "content": "policy"})[0]["content"][0]["text"] == "policy"
    parameters = {"type": "object", "properties": {"input": {"type": "string"}}}
    assert _ensure_object_root(parameters) is parameters
    with pytest.raises(ValueError, match="Name must"):
        AgentTool.validate("invalid name!")
    model = BareBoneModel("gpt-4o", "test", "url")
    model.agent_tools = [AgentTool("f", "fixture", "test", ToolArgs("object", "args"), required=True)]
    assert openai_responses.openai_responses_fill_payload(model, [], None)["tool_choice"] == {"type": "function", "name": "fixture"}


def test_anthropic_defensive_output_caps_are_enforced(monkeypatch):
    from IkaModel.anthropic import chat_helpers_anthropic as helper
    model = BareBoneModel("claude-haiku-4", "test", "url")
    for raw in ({}, {"max_tokens": 9000}):
        monkeypatch.setattr(helper, "anthropic_fill_payload", lambda *a, raw=raw, **kw: dict(raw))
        assert helper.build_anthropic_request(model, [], {})[2]["max_tokens"] == 4096


def test_openrouter_fallback_tool_choice_and_token_key(monkeypatch):
    from IkaModel.openrouter import openrouter as helper
    model = BareBoneModel("vendor/custom", "test", "url")
    for raw in ({"tool_choice": "required"}, {"tool_choice": "none"}):
        monkeypatch.setattr(helper, "openai_fill_payload", lambda *a, raw=raw, **kw: dict(raw))
        payload = helper.openrouter_fill_payload(model, [])
        assert payload["tool_choice"] == ("none" if raw["tool_choice"] == "none" else "auto")
        assert payload["max_tokens"] == model.max_tokens


def test_codex_unknown_slug_http_error_and_final_exception_preservation(monkeypatch):
    import httpx

    from IkaModel.codex import chat_helpers_codex as helper
    warning = Mock()
    monkeypatch.setattr(helper.LOG, "warning", warning)
    model = BareBoneModel("future-unlisted-model", "test", "https://chatgpt.com/backend-api/codex/responses")
    assert helper.build_codex_request(model, [], {})[2]["model"] == "future-unlisted-model"
    warning.assert_called_once()
    failure = httpx.ConnectError("synthetic transport failure")
    monkeypatch.setattr(helper, "_request_codex_once", Mock(side_effect=failure))
    monkeypatch.setattr(helper, "_retry_codex_exception", Mock())
    with pytest.raises(httpx.ConnectError) as caught:
        helper._request_codex_with_retries("url", {}, {}, 1, 1, 0)
    assert caught.value is failure


def test_codex_empty_completed_envelope_is_retryable_truncation():
    import httpx
    with pytest.raises(CodexRetryableStreamError, match=r"before response\.completed"):
        collect_stream(httpx.Response(200, text='data: {"type":"response.completed","response":{}}\n\n'))


def test_base_resource_missing_in_isolated_module_uses_fallback(monkeypatch):
    import builtins
    import runpy

    import IkaModel.base as base
    original = builtins.open
    identity = base.BareBoneModel
    def missing_resource(*a):
        raise FileNotFoundError("resource removed")
    def opening(path, *a, **kw):
        if str(path).endswith("summary_prompt"):
            raise FileNotFoundError("resource removed")
        return original(path, *a, **kw)
    monkeypatch.setattr("IkaModel.resource_loader.read_embedded_resource", missing_resource)
    monkeypatch.setattr(builtins, "open", opening)
    isolated = runpy.run_path(base.__file__, run_name="IkaModel._isolated_base")
    assert isolated["SUMMARY_PROMPT"].startswith("Please summarize")
    assert base.BareBoneModel is identity
