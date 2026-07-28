"""
Codex provider — retry classifier + retry-after parsing tests.

Covers:
  - _classify_stream_error recognizes transient vs caller-error variants.
  - _parse_retry_after handles integer seconds, float seconds, HTTP-date
    (both GMT and naive -0000 zone), garbage, missing.
  - request_codex retry loop:
      * 200 returns; status > 0 not retried
      * 401 raises immediately with body
      * 4xx other (400) raises immediately
      * 429 retried with Retry-After honored
      * 5xx retried with exponential backoff
      * persistent 5xx exhausts and raises
      * httpx.ReadTimeout retried
      * transient stream-level response.failed retried
      * non-transient stream-level response.failed bubbles
"""
import email.utils
import time
from unittest.mock import patch

import httpx
import pytest

from IkaModel.codex.chat_helpers_codex import (
    _classify_stream_error,
    _CodexRetryableStreamError,
    _parse_retry_after,
    request_codex,
)

# ----------------------------------------------------------------------
# _classify_stream_error
# ----------------------------------------------------------------------

@pytest.mark.parametrize("field", ["type", "code"])
@pytest.mark.parametrize("identifier", [
    "server_error",
    "internal_error",
    "engine_error",
    "rate_limit_exceeded",
    "rate_limit_error",
    "overloaded_error",
    "model_overloaded",
    "server_is_overloaded",
])
def test_classify_stream_error_accepts_transient_identifier_under_either_key(
    field,
    identifier,
):
    assert _classify_stream_error({field: identifier})


def test_classify_stream_error_normalizes_case_and_outer_whitespace():
    assert _classify_stream_error(
        {"code": "  SERVER_IS_OVERLOADED  "}
    )


@pytest.mark.parametrize("err", [
    {"type": "invalid_request_error"},
    {"type": "context_length_exceeded"},
    {"code": "context_length_exceeded"},
    {"code": "insufficient_quota"},
    {
        "type": "invalid_request_error",
        "message": "request mentions server_is_overloaded",
    },
    None,
    "just a string",
    {},
])
def test_classify_stream_error_rejects_non_transient_or_unknown_errors(err):
    assert not _classify_stream_error(err)


# ----------------------------------------------------------------------
# _parse_retry_after
# ----------------------------------------------------------------------

class TestParseRetryAfter:

    def test_integer_seconds(self):
        assert _parse_retry_after("30", default=99) == 30.0

    def test_float_seconds(self):
        assert _parse_retry_after("12.5", default=99) == 12.5

    def test_zero_seconds(self):
        assert _parse_retry_after("0", default=99) == 0.0

    def test_negative_seconds_clamped_to_zero(self):
        assert _parse_retry_after("-5", default=99) == 0.0

    def test_missing_falls_back(self):
        assert _parse_retry_after(None, default=7) == 7.0
        assert _parse_retry_after("", default=7) == 7.0

    def test_garbage_falls_back(self):
        assert _parse_retry_after("not-a-number", default=7) == 7.0

    def test_http_date_gmt(self):
        future = email.utils.formatdate(time.time() + 5, usegmt=True)
        v = _parse_retry_after(future, default=99)
        assert 3.0 <= v <= 7.0

    def test_http_date_naive_zone(self):
        """RFC 5322 ``-0000`` zone — formatdate's default form. Naive datetime
        must be treated as UTC, not local."""
        future = email.utils.formatdate(time.time() + 5)
        v = _parse_retry_after(future, default=99)
        assert 3.0 <= v <= 7.0

    def test_past_http_date_returns_zero(self):
        past = email.utils.formatdate(time.time() - 60, usegmt=True)
        assert _parse_retry_after(past, default=99) == 0.0


# ----------------------------------------------------------------------
# request_codex retry loop dispatch (mocked transport)
# ----------------------------------------------------------------------

class _FakeStreamCtx:
    """Mock just enough of the httpx.stream(...) context manager."""

    def __init__(self, status, body=b"", events=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self._events = events or []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self._body

    def iter_lines(self):
        for line in self._events:
            yield line


def _completed_event_stream():
    """A valid minimal SSE stream that exits with response.completed."""
    return [
        'event: response.created',
        'data: {"type":"response.created","response":{"id":"r","model":"x","output":[],"usage":{"total_tokens":2}}}',
        '',
        'event: response.completed',
        'data: {"type":"response.completed","response":{"id":"r","model":"x","output":[],"usage":{"total_tokens":2}}}',
        '',
    ]


class TestRequestCodexRetryLoop:

    def test_200_succeeds_immediately(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            return _FakeStreamCtx(200, events=_completed_event_stream())

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            resp = request_codex("u", {}, {}, timeout=1, max_retries=3, wait_seconds=1)
        assert resp.status_code == 200
        assert calls["n"] == 1

    def test_401_raises_no_retry(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            return _FakeStreamCtx(401, body=b'{"detail":"bad token"}')

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            with pytest.raises(RuntimeError, match="401"):
                request_codex("u", {}, {}, timeout=1, max_retries=3, wait_seconds=1)
        assert calls["n"] == 1

    def test_400_raises_no_retry(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            return _FakeStreamCtx(400, body=b'{"detail":"bad input"}')

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            with pytest.raises(RuntimeError, match="400"):
                request_codex("u", {}, {}, timeout=1, max_retries=3, wait_seconds=1)
        assert calls["n"] == 1

    def test_5xx_retried_then_succeeds(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeStreamCtx(503, body=b'{"error":"unavailable"}')
            return _FakeStreamCtx(200, events=_completed_event_stream())

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            resp = request_codex("u", {}, {}, timeout=1, max_retries=3, wait_seconds=1)
        assert resp.status_code == 200
        assert calls["n"] == 2

    def test_persistent_5xx_exhausts_retries(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            return _FakeStreamCtx(502, body=b'{"error":"bad gateway"}')

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            with pytest.raises(RuntimeError, match="502.*after 3 attempts"):
                request_codex("u", {}, {}, timeout=1, max_retries=3, wait_seconds=1)
        assert calls["n"] == 3

    def test_429_honors_retry_after(self):
        calls = {"n": 0}
        sleeps = []

        def side_effect(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeStreamCtx(429, body=b'{}', headers={"retry-after": "7"})
            return _FakeStreamCtx(200, events=_completed_event_stream())

        with patch("httpx.stream", side_effect=side_effect), \
             patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
            resp = request_codex("u", {}, {}, timeout=1, max_retries=3, wait_seconds=99)
        assert resp.status_code == 200
        # Honored 7s from header, not 99s default.
        assert 7.0 in sleeps

    def test_transient_stream_failure_retried(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeStreamCtx(200, events=[
                    'event: response.failed',
                    'data: {"type":"response.failed","error":{"type":"server_error"}}',
                    '',
                ])
            return _FakeStreamCtx(200, events=_completed_event_stream())

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            resp = request_codex("u", {}, {}, timeout=1, max_retries=3, wait_seconds=1)
        assert resp.status_code == 200
        assert calls["n"] == 2

    def test_server_is_overloaded_then_success_retries_and_discards_partial_output(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeStreamCtx(200, events=[
                    'event: response.created',
                    'data: {"type":"response.created","response":{"id":"partial","output":[]}}',
                    '',
                    'event: response.output_text.delta',
                    'data: {"type":"response.output_text.delta","delta":"discard me"}',
                    '',
                    'event: error',
                    'data: {"type":"error","code":"server_is_overloaded","message":"busy"}',
                    '',
                ])
            return _FakeStreamCtx(200, events=_completed_event_stream())

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            resp = request_codex(
                "u",
                {},
                {},
                timeout=1,
                max_retries=3,
                wait_seconds=1,
            )

        assert calls["n"] == 2
        assert resp.json()["id"] == "r"
        assert resp.json()["output"] == []

    def test_persistent_server_is_overloaded_uses_three_total_attempts(self):
        calls = {"n": 0}
        sleeps = []

        def side_effect(*a, **kw):
            calls["n"] += 1
            return _FakeStreamCtx(200, events=[
                'event: response.failed',
                'data: {"type":"response.failed","response":{"status":"failed","error":{"code":"server_is_overloaded"}}}',
                '',
            ])

        with patch("httpx.stream", side_effect=side_effect), patch(
            "time.sleep",
            side_effect=lambda seconds: sleeps.append(seconds),
        ):
            with pytest.raises(_CodexRetryableStreamError):
                request_codex(
                    "u",
                    {},
                    {},
                    timeout=1,
                    max_retries=3,
                    wait_seconds=1,
                )

        assert calls["n"] == 3
        assert sleeps == [1, 2]

    def test_truncated_http_200_stream_retries(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeStreamCtx(200, events=[
                    'event: response.created',
                    'data: {"type":"response.created","response":{"id":"partial","status":"in_progress","output":[]}}',
                    '',
                ])
            return _FakeStreamCtx(200, events=_completed_event_stream())

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            resp = request_codex(
                "u",
                {},
                {},
                timeout=1,
                max_retries=3,
                wait_seconds=1,
            )

        assert calls["n"] == 2
        assert resp.json()["id"] == "r"

    def test_explicit_incomplete_stream_does_not_retry(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            return _FakeStreamCtx(200, events=[
                'event: response.incomplete',
                'data: {"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}',
                '',
            ])

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            with pytest.raises(RuntimeError, match="incomplete") as exc_info:
                request_codex(
                    "u",
                    {},
                    {},
                    timeout=1,
                    max_retries=3,
                    wait_seconds=1,
                )

        assert not isinstance(exc_info.value, _CodexRetryableStreamError)
        assert calls["n"] == 1

    def test_non_transient_stream_failure_bubbles(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            return _FakeStreamCtx(200, events=[
                'event: response.failed',
                'data: {"type":"response.failed","error":{"type":"invalid_request_error"}}',
                '',
            ])

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            with pytest.raises(RuntimeError) as exc_info:
                request_codex("u", {}, {}, timeout=1, max_retries=3, wait_seconds=1)
        assert not isinstance(exc_info.value, _CodexRetryableStreamError)
        assert calls["n"] == 1   # no retry

    def test_read_timeout_retried(self):
        calls = {"n": 0}

        def side_effect(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("slow")
            return _FakeStreamCtx(200, events=_completed_event_stream())

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            resp = request_codex("u", {}, {}, timeout=1, max_retries=3, wait_seconds=1)
        assert resp.status_code == 200
        assert calls["n"] == 2

    def test_forces_stream_true_in_payload(self):
        captured = {}

        def side_effect(method, url, *, headers, json, timeout):
            captured.update(json)
            return _FakeStreamCtx(200, events=_completed_event_stream())

        with patch("httpx.stream", side_effect=side_effect), patch("time.sleep"):
            request_codex("u", {}, {"model": "gpt-5.5", "stream": False}, timeout=1,
                          max_retries=1, wait_seconds=1)
        assert captured["stream"] is True
