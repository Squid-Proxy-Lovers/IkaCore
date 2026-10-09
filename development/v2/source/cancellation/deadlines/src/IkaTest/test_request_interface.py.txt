import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from IkaModel import request_interface as ri


def test_context_retry_override_applies_and_restores(monkeypatch):
    observed = []

    def fake_request_codex(
        _api_url,
        _headers,
        _payload,
        timeout,
        max_retries,
        wait_seconds,
    ):
        observed.append((timeout, max_retries, wait_seconds))
        return object()

    monkeypatch.setattr(
        "IkaModel.codex.chat_helpers_codex.request_codex",
        fake_request_codex,
    )

    previous = ri.set_request_max_retries(1)
    try:
        ri.api_request_retry(
            "https://chatgpt.com/backend-api/codex/responses",
            {},
            {},
            max_retries=3,
            wait_seconds=7,
            timeout=11,
        )
    finally:
        ri.set_request_max_retries(previous)

    ri.api_request_retry(
        "https://chatgpt.com/backend-api/codex/responses",
        {},
        {},
        max_retries=3,
        wait_seconds=7,
        timeout=11,
    )

    assert observed == [(11, 1, 7), (11, 3, 7)]
from IkaModel.request_interface import (
    IkaAPIError,
    _error_text_from_response,
    _rate_limit_wait_time,
    _response_json_preview,
    _response_text_preview,
    api_request_retry,
    async_api_request_retry,
)


def test_redaction_and_error_classifiers_cover_sensitive_values_and_context_errors():
    assert ri._redact_headers(
        {
            "Authorization": "Bearer secret",
            "x-api-key": "secret",
            "safe": "value",
        }
    ) == {"Authorization": "Bearer ***", "x-api-key": "***", "safe": "value"}
    assert ri._redact_url("https://example.test/path?api_key=secret&ok=1#frag") == (
        "https://example.test/path?api_key=%2A%2A%2A&ok=1#frag"
    )
    assert ri._redact_url(None) == b""

    assert ri._is_rate_limit_error(httpx.Response(429))
    assert ri._is_rate_limit_error(httpx.Response(500, json={"error": "rate_limit exceeded"}))
    assert ri._is_rate_limit_error(httpx.Response(500, content=b"rate limit exceeded"))
    assert not ri._is_rate_limit_error(httpx.Response(500, content=b"server exploded"))

    assert ri._is_context_length_error(RuntimeError("maximum context length exceeded"))
    assert ri._is_context_length_error(RuntimeError("requested 100 tokens, maximum is 50"))
    assert not ri._is_context_length_error(RuntimeError("temporary server error"))


def test_response_preview_helpers_handle_json_and_text_bodies():
    json_response = httpx.Response(400, json={"error": {"message": "bad"}})
    text_response = httpx.Response(500, content=b"plain error body")

    assert '"message": "bad"' in _response_json_preview(json_response)
    assert _response_json_preview(text_response) is None
    assert _response_text_preview(text_response, 5) == "plain"
    assert _error_text_from_response(json_response, 200).startswith("{")
    assert _error_text_from_response(text_response, 200) == "plain error body"


def test_rate_limit_wait_time_uses_retry_after_or_clamped_backoff():
    assert _rate_limit_wait_time(httpx.Response(429, headers={"retry-after": "12"}), 10, 0) == (12, True)
    assert _rate_limit_wait_time(httpx.Response(429, headers={"retry-after": "not-int"}), 1, 0) == (30, False)
    assert _rate_limit_wait_time(httpx.Response(429), 200, 3) == (300, False)


def test_retry_delay_helpers_return_or_raise_typed_errors():
    with patch("IkaModel.request_interface.get_cli_output", return_value=MagicMock()):
        wait, err = ri._response_retry_delay_and_error(
            httpx.Response(429, content=b"slow down"),
            wait_seconds=1,
            attempt=0,
            max_retries=2,
        )
    assert wait == 30
    assert isinstance(err, ri.IkaRateLimitError)
    assert err.status_code == 429

    with pytest.raises(ri.IkaRateLimitError, match="after 1 attempts"):
        ri._response_retry_delay_and_error(
            httpx.Response(429, json={"error": "slow down"}),
            wait_seconds=1,
            attempt=0,
            max_retries=1,
        )

    wait, err = ri._timeout_retry_delay_and_error(timeout=0.5, wait_seconds=2, attempt=0, max_retries=2)
    assert wait == 2
    assert isinstance(err, ri.IkaTimeoutError)
    with pytest.raises(ri.IkaTimeoutError):
        ri._timeout_retry_delay_and_error(timeout=0.5, wait_seconds=2, attempt=1, max_retries=2)

    wait, err = ri._http_retry_delay_and_error(httpx.ConnectError("down"), wait_seconds=3, attempt=0, max_retries=2)
    assert wait == 3
    assert isinstance(err, ri.IkaHTTPError)
    with pytest.raises(ri.IkaHTTPError, match="after 2 attempts"):
        ri._http_retry_delay_and_error(httpx.ConnectError("down"), wait_seconds=3, attempt=1, max_retries=2)

    original = RuntimeError("unexpected")
    wait, err = ri._unexpected_retry_delay_and_error(original, wait_seconds=4, attempt=0, max_retries=2)
    assert (wait, err) == (4, original)
    with pytest.raises(RuntimeError, match="unexpected"):
        ri._unexpected_retry_delay_and_error(original, wait_seconds=4, attempt=1, max_retries=2)


def test_sync_api_retry_retries_server_error_then_succeeds():
    responses = [
        httpx.Response(500, json={"error": "temporary"}),
        httpx.Response(200, json={"ok": True}),
    ]

    with patch("IkaModel.request_interface.httpx.post", side_effect=responses) as post:
        with patch("IkaModel.request_interface.time.sleep") as sleep:
            response = api_request_retry(
                "https://example.test",
                {"Authorization": "Bearer secret"},
                {"input": "x"},
                max_retries=2,
                wait_seconds=1,
            )

    assert response.json() == {"ok": True}
    assert post.call_count == 2
    sleep.assert_called_once_with(1)


def test_sync_api_retry_raises_typed_error_after_final_server_error():
    with patch(
        "IkaModel.request_interface.httpx.post",
        return_value=httpx.Response(500, json={"error": "still bad"}),
    ):
        with pytest.raises(IkaAPIError, match="after 1 attempts") as exc_info:
            api_request_retry("https://example.test", {}, {}, max_retries=1)

    assert exc_info.value.status_code == 500
    assert "still bad" in str(exc_info.value)


def test_sync_api_retry_dispatches_codex_and_retries_http_errors():
    codex_response = httpx.Response(200, json={"ok": "codex"})
    with patch("IkaModel.codex.chat_helpers_codex.request_codex", return_value=codex_response) as request_codex:
        assert api_request_retry(
            "https://chatgpt.com/backend-api/codex/responses",
            {},
            {"input": []},
            max_retries=2,
            wait_seconds=1,
            timeout=5,
        ) is codex_response
    request_codex.assert_called_once()

    responses = [httpx.ConnectError("temporary"), httpx.Response(200, json={"ok": True})]
    with patch("IkaModel.request_interface.httpx.post", side_effect=responses):
        with patch("IkaModel.request_interface.time.sleep") as sleep:
            response = api_request_retry("https://example.test", {}, {}, max_retries=2, wait_seconds=1)

    assert response.json() == {"ok": True}
    sleep.assert_called_once_with(1)

    with patch("IkaModel.request_interface.httpx.post", side_effect=ValueError("bad client")):
        with pytest.raises(ValueError, match="bad client"):
            api_request_retry("https://example.test", {}, {}, max_retries=1)


def test_async_api_retry_honors_retry_after_and_reuses_caller_client():
    calls = {"count": 0}

    def handler(_request):
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"error": "slow down"})
        return httpx.Response(200, json={"ok": True})

    async def run_case():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        sleep = AsyncMock()
        try:
            with patch("IkaModel.request_interface.asyncio.sleep", sleep):
                response = await async_api_request_retry(
                    "https://example.test",
                    {},
                    {},
                    max_retries=2,
                    wait_seconds=1,
                    client=client,
                )
            assert client.is_closed is False
            assert response.json() == {"ok": True}
            sleep.assert_awaited_once_with(1)
        finally:
            await client.aclose()

    asyncio.run(run_case())


def test_async_api_retry_dispatches_codex_closes_owned_client_and_raises_last_error():
    codex_response = httpx.Response(200, json={"ok": "codex"})

    async def run_codex():
        with patch("IkaModel.codex.chat_helpers_codex.request_codex", return_value=codex_response) as request_codex:
            response = await async_api_request_retry(
                "https://chatgpt.com/backend-api/codex/responses",
                {},
                {"input": []},
                max_retries=2,
                wait_seconds=1,
                timeout=5,
            )
        request_codex.assert_called_once()
        return response

    assert asyncio.run(run_codex()) is codex_response

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout
            self.closed = False

        async def post(self, *_args, **_kwargs):
            raise httpx.ConnectError("down")

        async def aclose(self):
            self.closed = True

    created = []

    def make_client(timeout):
        client = FakeAsyncClient(timeout)
        created.append(client)
        return client

    async def run_owned_client_error():
        with patch("IkaModel.request_interface.httpx.AsyncClient", side_effect=make_client):
            with patch("IkaModel.request_interface.asyncio.sleep", AsyncMock()):
                with pytest.raises(ri.IkaHTTPError, match="after 1 attempts"):
                    await async_api_request_retry("https://example.test", {}, {}, max_retries=1, timeout=0.5)

    asyncio.run(run_owned_client_error())
    assert created[0].closed is True


def test_agent_tools_for_payload_caches_filters_and_compatibility_shims_restore():
    unlimited = type("Tool", (), {"name": "always", "limit_calls": 0})()
    limited = type("Tool", (), {"name": "limited", "limit_calls": 2})()
    model = type("Model", (), {})()
    model.agent_tools = [unlimited, limited]
    model._tool_call_counts = {"limited": 1}

    first = ri.agent_tools_for_payload(model)
    second = ri.agent_tools_for_payload(model)
    assert first is second
    assert first == [unlimited, limited]

    model._tool_call_counts = {"limited": 2}
    filtered = ri.agent_tools_for_payload(model)
    assert filtered == [unlimited]
    assert ri.model_for_payload(model).agent_tools == [unlimited]
    assert model.agent_tools == [unlimited, limited]

    ri._apply_tools_filter_for_payload(model)
    assert model.agent_tools == [unlimited]
    ri._restore_tools_after_payload(model)
    assert model.agent_tools == [unlimited, limited]
