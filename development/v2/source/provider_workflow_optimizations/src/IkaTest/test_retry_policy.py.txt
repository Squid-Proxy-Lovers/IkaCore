import time
from unittest.mock import patch

import httpx
import pytest

from IkaModel.request_interface import IkaAPIError, api_request_retry
from IkaModel.retry_policy import (
    MAX_BACKOFF_S,
    MAX_HINT_S,
    backoff_seconds,
    is_retryable_status,
    retry_delay,
    retry_hint_seconds,
)


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 529])
def test_transient_statuses_are_retryable(status):
    assert is_retryable_status(status)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_permanent_statuses_are_not_retryable(status):
    assert not is_retryable_status(status)


def test_hint_formats():
    assert retry_hint_seconds({"retry-after-ms": "500"}) == 0.5
    assert retry_hint_seconds({"Retry-After": "2"}) == 2.0
    assert retry_hint_seconds({"retry-after": "0.8"}) == 0.8
    future = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(time.time() + 30))
    assert 25 <= retry_hint_seconds({"retry-after": future}) <= 31
    assert retry_hint_seconds({"x-ratelimit-reset-requests": "1.5s", "x-ratelimit-reset-tokens": "6m0s"}) == 360.0
    assert retry_hint_seconds({"x-ratelimit-reset-tokens": "250ms"}) == 0.25
    gemini = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "25s"}]}}
    assert retry_hint_seconds({}, gemini) == 25.0
    assert retry_hint_seconds({}) is None
    assert retry_hint_seconds({"retry-after": "soon", "x-ratelimit-reset-requests": "later"}) is None


def test_backoff_is_exponential_jittered_and_capped():
    for attempt, base in ((0, 1.0), (1, 2.0), (2, 4.0), (10, MAX_BACKOFF_S)):
        for _ in range(50):
            assert 0.75 * base <= backoff_seconds(attempt) <= base
    assert retry_delay(0, 3.0) == 3.0
    assert retry_delay(0, 10_000.0) == MAX_HINT_S


@pytest.mark.parametrize("status", [400, 401, 404])
def test_permanent_errors_fail_on_first_attempt_without_sleeping(status):
    with patch("IkaModel.request_interface.httpx.post", return_value=httpx.Response(status, json={"error": "no"})) as post:
        with patch("IkaModel.request_interface.time.sleep") as sleep:
            with pytest.raises(IkaAPIError) as exc_info:
                api_request_retry("https://example.test", {}, {}, max_retries=3)
    assert exc_info.value.status_code == status
    assert post.call_count == 1
    sleep.assert_not_called()


def test_rate_limit_honours_millisecond_hint():
    responses = [httpx.Response(429, headers={"retry-after-ms": "250"}, json={"error": "rate limit"}),
                 httpx.Response(200, json={"ok": True})]
    with patch("IkaModel.request_interface.httpx.post", side_effect=responses):
        with patch("IkaModel.request_interface.time.sleep") as sleep, \
             patch("IkaModel.request_interface.get_cli_output"):
            assert api_request_retry("https://example.test", {}, {}, max_retries=3).json() == {"ok": True}
    sleep.assert_called_once_with(0.25)


def test_programmer_errors_are_not_retried():
    with patch("IkaModel.request_interface.httpx.post", side_effect=TypeError("bad payload")) as post:
        with patch("IkaModel.request_interface.time.sleep") as sleep:
            with pytest.raises(TypeError):
                api_request_retry("https://example.test", {}, {}, max_retries=3)
    assert post.call_count == 1
    sleep.assert_not_called()
