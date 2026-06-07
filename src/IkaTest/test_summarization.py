import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from IkaModel import summarization as s
from IkaModel.base import BareBoneModel


def _model(**overrides):
    defaults = {
        "model_id": "gpt-4o",
        "api_key": "test-key",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "suppress_init_output": True,
        "use_responses_api": False,
    }
    defaults.update(overrides)
    return BareBoneModel(**defaults)


def _history():
    return {
        "system": {"message": "system", "tokens": 0},
        "first_input": {"message": "first input", "tokens": 0},
        "summary": {"message": "[SUMMARY]\nold", "tokens": 2},
        "messages": {
            "1": {"message": "assistant: one", "tokens": 3},
            "2": {"message": "user: two", "tokens": 0},
        },
    }


def _response(data):
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def test_prompt_loading_and_prompt_kind_selection_fallbacks():
    assert s._load_prompt("definitely_missing_prompt.txt", "fallback") == "fallback"
    assert s._get_prompts_for_kind("force_answer") == (s.FORCE_ANSWER_SYSTEM, s.FORCE_ANSWER_USER_PREFIX)
    assert s._get_prompts_for_kind("what_remains") == (s.WHAT_REMAINS_SYSTEM, s.WHAT_REMAINS_USER_PREFIX)
    assert s._get_prompts_for_kind("unknown") == (s.SUMMARY_PROMPT, s.DEFAULT_SUMMARY_USER_PREFIX)


def test_summary_payload_builders_cover_provider_specific_shapes():
    deepseek, deepseek_headers = s.create_summary_payload(
        "deepseek",
        "deepseek-chat",
        "key",
        "conversation",
        system_prompt="sys",
        user_prompt_prefix="prefix:",
    )
    assert deepseek["stream"] is False
    assert deepseek["temperature"] == 0.3
    assert deepseek["messages"][0] == {"role": "system", "content": "sys"}
    assert deepseek["messages"][1]["content"] == "prefix:conversation"
    assert deepseek_headers["Authorization"] == "Bearer key"

    with patch("IkaModel.summarization.uses_openai_max_completion_tokens", return_value=True):
        with patch("IkaModel.summarization.supports_custom_temperature", return_value=False):
            openai, _ = s.create_summary_payload("openai_responses", "o3-mini", "key", "conversation")
    assert "max_completion_tokens" in openai
    assert "temperature" not in openai

    anthropic, anthropic_headers = s.create_summary_payload("anthropic", "claude", "key", "conversation")
    assert anthropic["system"] == s.SUMMARY_PROMPT
    assert anthropic["messages"][0]["role"] == "user"
    assert anthropic_headers["x-api-key"] == "key"

    gemini, gemini_headers = s.create_summary_payload("gemini", "gemini", "key", "conversation")
    assert gemini["contents"][0]["parts"][0]["text"].startswith(s.SUMMARY_PROMPT)
    assert gemini_headers["x-goog-api-key"] == "key"

    with pytest.raises(ValueError, match="Unsupported provider"):
        s.create_summary_payload("unsupported", "model", "key", "conversation")


def test_summary_response_parsers_and_token_extractors_cover_all_providers():
    assert s.parse_summary_response("openai", _response({"choices": [{"message": {"content": "openai summary"}}]})) == "openai summary"
    assert s.parse_summary_response(
        "anthropic",
        _response({"content": [{"type": "text", "text": "one"}, {"type": "tool_use", "text": "ignored"}, {"type": "text", "text": "two"}]}),
    ) == "onetwo"
    assert s.parse_summary_response(
        "gemini",
        _response({"candidates": [{"content": {"parts": [{"text": "gemini summary"}]}}]}),
    ) == "gemini summary"
    with pytest.raises(ValueError, match="Unsupported provider"):
        s.parse_summary_response("unsupported", _response({}))

    assert s.extract_summary_tokens("openai", {"usage": {"total_tokens": 7}}) == 7
    assert s.extract_summary_tokens("openai_responses", {"usage": {"total_tokens": 8}}) == 8
    assert s.extract_summary_tokens("anthropic", {"usage": {"input_tokens": 2, "output_tokens": 3}}) == 5
    assert s.extract_summary_tokens("gemini", {"usageMetadata": {"totalTokenCount": 6}}) == 6
    assert s.extract_summary_tokens("codex", {"usage": {"input_tokens": 4, "output_tokens": 5}}) == 9
    assert s.extract_summary_tokens("unsupported", {"usage": {"total_tokens": 7}}) == 0


def test_summary_target_resolution_and_history_helpers():
    assert s._resolve_summary_target(_model(), "openai", use_same_model=False) == s.get_summary_model("openai")

    codex_model = SimpleNamespace(model_id="gpt-5.4-mini", api_url="")
    model_name, api_url = s._resolve_summary_target(codex_model, "codex", use_same_model=True)
    assert model_name == "gpt-5.4-mini"
    assert "codex/responses" in api_url

    responses_model = SimpleNamespace(model_id="gpt-4o", api_url="https://api.openai.com/v1/responses")
    assert s._resolve_summary_target(responses_model, "openai", use_same_model=True) == (
        "gpt-4o",
        s.OPENAI_CHAT_COMPLETIONS_URL,
    )

    history = _history()
    assert s.get_conversation_text(history).splitlines() == [
        "first input",
        "[SUMMARY]",
        "old",
        "assistant: one",
        "user: two",
    ]
    s._write_summary_to_history(history, "new summary", 11)
    assert history["summary"] == {"message": "[SUMMARY]\nnew summary", "tokens": 11}
    assert history["messages"] == {}


def test_run_summarization_success_writes_history_and_supports_no_history_short_circuit():
    model = _model()
    empty = {"system": {"message": "", "tokens": 0}, "first_input": {"message": "", "tokens": 0}, "summary": {"message": "", "tokens": 0}, "messages": {}}
    assert s.run_summarization(model, empty) == ""

    history = _history()
    response = _response({"choices": [{"message": {"content": "fresh summary"}}], "usage": {"total_tokens": 12}})

    with patch("IkaModel.summarization.api_request_retry", return_value=response) as retry:
        summary = s.run_summarization(model, history, prompt_kind="what_remains", write_to_history=True)

    assert summary == "fresh summary"
    assert history["summary"] == {"message": "[SUMMARY]\nfresh summary", "tokens": 12}
    assert history["messages"] == {}
    retry.assert_called_once()
    assert "DONE" in retry.call_args.args[2]["messages"][0]["content"]


def test_run_summarization_returns_empty_for_missing_target_http_and_unexpected_errors():
    model = _model()
    history = _history()

    with patch("IkaModel.summarization.get_summary_model", return_value=(None, None)):
        assert s.run_summarization(model, history, use_same_model=False) == ""

    with patch("IkaModel.summarization.api_request_retry", side_effect=httpx.HTTPError("network")):
        assert s.run_summarization(model, history, write_to_history=False) == ""

    bad_response = _response({"unexpected": True})
    with patch("IkaModel.summarization.api_request_retry", return_value=bad_response):
        assert s.run_summarization(model, history, write_to_history=False) == ""


def test_summarise_message_history_is_default_wrapper():
    model = _model()
    history = _history()
    with patch("IkaModel.summarization.run_summarization", return_value="wrapped") as run:
        assert s.summarise_message_history(model, history, client="client") == "wrapped"
    run.assert_called_once_with(
        model,
        history,
        prompt_kind="default",
        write_to_history=True,
        client="client",
    )


def test_async_summarise_message_history_success_and_error_paths():
    async def run_case():
        model = _model()
        empty = {"system": {"message": "", "tokens": 0}, "first_input": {"message": "", "tokens": 0}, "summary": {"message": "", "tokens": 0}, "messages": {}}
        assert await s.async_summarise_message_history(model, empty) == ""

        history = _history()
        response = _response({"choices": [{"message": {"content": "async summary"}}], "usage": {"total_tokens": 10}})
        with patch("IkaModel.summarization.async_api_request_retry", return_value=response):
            summary = await s.async_summarise_message_history(model, history, prompt_kind="force_answer")

        missing_history = _history()
        with patch("IkaModel.summarization.get_summary_model", return_value=(None, None)):
            missing = await s.async_summarise_message_history(model, missing_history, use_same_model=False)

        http_history = _history()
        with patch("IkaModel.summarization.async_api_request_retry", side_effect=httpx.HTTPError("network")):
            http_error = await s.async_summarise_message_history(model, http_history, write_to_history=False)

        unexpected_history = _history()
        bad_response = _response({"unexpected": True})
        with patch("IkaModel.summarization.async_api_request_retry", return_value=bad_response):
            unexpected = await s.async_summarise_message_history(model, unexpected_history, write_to_history=False)

        return summary, history, missing, http_error, unexpected

    summary, history, missing, http_error, unexpected = asyncio.run(run_case())

    assert summary == "async summary"
    assert history["summary"] == {"message": "[SUMMARY]\nasync summary", "tokens": 10}
    assert missing == ""
    assert http_error == ""
    assert unexpected == ""
