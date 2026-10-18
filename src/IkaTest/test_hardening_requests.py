"""Behavioral failure-path contracts for retries, metadata and chat requests."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from IkaModel import model_metadata as metadata
from IkaModel import request_interface as requests
from IkaModel.base import BareBoneModel
from IkaModel.chat_interface import chat_request as chat
from IkaModel.runtime_errors import IkaContextWindowError, IkaRequestCancelled


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", [httpx.ReadTimeout("read"), httpx.ConnectError("connect"), OSError("OS")])
def test_retry_transport_failures_recover_and_exhaust_with_typed_error(monkeypatch, asynchronous, failure):
    monkeypatch.setattr(requests, "sleep_with_controls", lambda *a: None)
    monkeypatch.setattr(requests, "async_sleep_with_controls", AsyncMock())
    success = httpx.Response(200, json={"ok": True})
    transport = AsyncMock(side_effect=[failure, success]) if asynchronous else Mock(side_effect=[failure, success])
    monkeypatch.setattr(requests, "controlled_async_post" if asynchronous else "controlled_post", transport)
    def invoke():
        if asynchronous:
            return asyncio.run(requests.async_api_request_retry("https://example.test", {}, {}, client=SimpleNamespace(), max_retries=2))
        return requests.api_request_retry("https://example.test", {}, {}, max_retries=2)
    assert invoke().json() == {"ok": True}
    transport.side_effect = [failure, failure]
    expected = requests.IkaTimeoutError if isinstance(failure, httpx.TimeoutException) else requests.IkaHTTPError if isinstance(failure, httpx.HTTPError) else OSError
    with pytest.raises(expected):
        invoke()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_zero_attempts_and_defensive_last_error_path(monkeypatch, asynchronous):
    monkeypatch.setattr(requests, "effective_max_retries", lambda n: n)
    monkeypatch.setattr(requests, "sleep_with_controls", lambda *a: None)
    monkeypatch.setattr(requests, "async_sleep_with_controls", AsyncMock())
    def invoke(n):
        if asynchronous:
            return asyncio.run(requests._async_retry_loop("url", {}, {}, 1, SimpleNamespace(), n, 0))
        return requests.api_request_retry("url", {}, {}, max_retries=n)
    with pytest.raises(requests.IkaAPIError, match="0 attempts"):
        invoke(0)
    # Exercise loop exhaustion independently of policy helpers that normally
    # raise on the final response: preserve the last failure object exactly.
    failure = requests.IkaAPIError("last")
    monkeypatch.setattr(requests, "_response_retry_delay_and_error", lambda *a: (0, failure))
    monkeypatch.setattr(requests, "controlled_post", lambda *a: httpx.Response(503))
    monkeypatch.setattr(requests, "controlled_async_post", AsyncMock(return_value=httpx.Response(503)))
    with pytest.raises(requests.IkaAPIError, match="last") as caught:
        invoke(1)
    assert caught.value is failure


@pytest.mark.parametrize("asynchronous", [False, True])
def test_debug_logs_redact_credentials_and_accept_text_responses(monkeypatch, caplog, asynchronous):
    caplog.set_level(logging.DEBUG, logger=requests.__name__)
    response = httpx.Response(200, text="ordinary text")
    monkeypatch.setattr(requests, "controlled_post", lambda *a: response)
    monkeypatch.setattr(requests, "controlled_async_post", AsyncMock(return_value=response))
    url, headers = "https://example.test/path?token=secret-token&label=ok", {"Authorization": "Bearer hidden", "x-api-key": "secret-key", "Accept": "json"}
    if asynchronous:
        asyncio.run(requests.async_api_request_retry(url, headers, {}, client=SimpleNamespace()))
    else:
        requests.api_request_retry(url, headers, {})
    assert "ordinary text" in caplog.text and "label=ok" in caplog.text
    assert "secret-token" not in caplog.text and "secret-key" not in caplog.text and "Bearer hidden" not in caplog.text
    assert requests._redact_url("http://[bad") == "http://[bad"
    assert requests._int_value(True) == 1 and requests._int_value("1") == 0


@pytest.mark.parametrize("body,expected", [({"error": "RATE LIMIT reached"}, True), ("rate_limit text", True), ("ordinary", False)])
def test_legacy_rate_limit_detection_uses_json_and_text(body, expected):
    response = httpx.Response(400, json=body) if isinstance(body, dict) else httpx.Response(400, text=body)
    assert requests._is_rate_limit_error(response) is expected


@pytest.mark.parametrize("loader,value,error", [
    ("_require_mapping", None, "non-empty"), ("_require_mapping", {}, "non-empty"),
    ("_load_tokenmax_mapping", {"": 1}, "non-empty"), ("_load_tokenmax_mapping", {"model": 0}, "positive"),
    ("_load_model_costs", {"": [1, 2, 3]}, "non-empty"), ("_load_model_costs", {"model": "bad"}, "3-item"),
    ("_load_model_costs", {"model": [1]}, "3-item"), ("_load_model_costs", {"model": [-1, 0, 0]}, "negative"),
    ("_load_string_set", {}, "list"), ("_load_api_urls", {"codex": "wrong"}, "out of sync"),
])
def test_invalid_metadata_fails_with_actionable_error(loader, value, error):
    key = {"_load_tokenmax_mapping": "tokenmax_mapping", "_load_model_costs": "model_costs", "_load_api_urls": "api_url_by_provider"}.get(loader, "fixture")
    args = ({"fixture": value}, "fixture") if loader in {"_require_mapping", "_load_string_set"} else ({key: value},)
    with pytest.raises(RuntimeError, match=error):
        getattr(metadata, loader)(*args)


@pytest.mark.parametrize("raw,error", [("{", "invalid"), ("[]", "JSON object")])
def test_metadata_resource_parse_errors(monkeypatch, raw, error):
    monkeypatch.setattr("IkaModel.resource_loader.read_embedded_resource", lambda *a: raw)
    with pytest.raises(RuntimeError, match=error):
        metadata._load_model_metadata()


def test_metadata_resource_fallbacks_and_vendor_resolution(monkeypatch):
    def missing(*a, **kw):
        raise FileNotFoundError("resource")
    monkeypatch.setattr("IkaModel.resource_loader.read_embedded_resource", missing)
    assert metadata._load_model_metadata()["tokenmax_mapping"]
    monkeypatch.setattr(metadata, "files", lambda *a: SimpleNamespace(joinpath=lambda *a: SimpleNamespace(read_text=missing)))
    with pytest.raises(RuntimeError, match="missing model metadata"):
        metadata._load_model_metadata()
    assert metadata.get_provider_for_model("unknown", "https://api.openai.com/custom", False) == "openai"
    assert metadata.get_provider_for_model("unknown", "https://api.openai.com/custom", True) == "openai_responses"
    assert metadata.get_max_tokens_for_model("llama-unlisted") == 131072
    assert metadata.get_max_tokens_for_model("qwen-unlisted") == 131072
    assert metadata.get_max_tokens_for_model("unknown") == metadata.DEFAULT_MAX_TOKENS
    assert metadata.get_api_url_for_model("claude-unlisted") == metadata.API_URL_BY_PROVIDER["anthropic"]
    assert metadata.resolve_model_cost("anthropic.claude-sonnet-4@vendor") == metadata.MODEL_COSTS["claude-sonnet-4"]
    assert "claude-sonnet-4" in metadata._model_cost_candidates("claude-sonnet-4:vendor")
    assert metadata.resolve_model_cost("claude-sonnet-4-custom-suffix") == metadata.MODEL_COSTS["claude-sonnet-4"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("phase", ["cancel", "other_http", "summary", "summary_cancel", "rebuild", "value"])
def test_chat_context_recovery_error_contracts(monkeypatch, asynchronous, phase):
    model = BareBoneModel("gpt-4o", "test", "https://api.openai.com/v1/chat/completions", suppress_init_output=True)
    overflow = requests.IkaAPIError("maximum context length")
    first = IkaRequestCancelled("stop") if phase == "cancel" else requests.IkaAPIError("other") if phase == "other_http" else ValueError("bad") if phase == "value" else overflow
    req = AsyncMock(side_effect=first) if asynchronous else Mock(side_effect=first)
    monkeypatch.setattr(requests, "async_api_request_retry" if asynchronous else "api_request_retry", req)
    summary = IkaRequestCancelled("stop") if phase == "summary_cancel" else ValueError("summary") if phase == "summary" else None
    summarizer = AsyncMock(side_effect=summary) if asynchronous else Mock(side_effect=summary)
    monkeypatch.setattr(chat._summaries, "async_summarise_message_history" if asynchronous else "summarise_message_history", summarizer)
    build = Mock(side_effect=[("url", {}, {}), ValueError("rebuild")])
    expected = IkaRequestCancelled if phase in {"cancel", "summary_cancel"} else requests.IkaAPIError if phase == "other_http" else IkaContextWindowError
    with pytest.raises(expected):
        if asynchronous:
            asyncio.run(chat._api_request_with_context_fallback_async(build, model, {}, client=SimpleNamespace()))
        else:
            chat._api_request_with_context_fallback(build, model, {})
    assert build.call_count == (2 if phase == "rebuild" else 1)


def test_chat_accounting_non_numeric_tokens_are_estimated():
    assert chat._token_count(True) == 1
    assert chat._token_count("many") == 0
    assert chat.get_total_tokens({"system": {"tokens": "bad", "message": "12345678"}}) == 2
    assert chat._parse_provider_round("openai", {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 9}}, "gpt-4o")[3] == 9
