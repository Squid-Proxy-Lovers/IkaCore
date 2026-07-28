"""
Codex provider — SSE collection (_collect_stream + _iter_sse) tests.

Covers:
  - response.completed envelope is preserved (id, model, usage).
  - response.output_item.done events populate the final output[] array
    even when response.completed.response.output is empty (codex backend
    quirk).
  - response.output_text.delta is used as a last-resort fallback.
  - response.failed with retryable error type raises _CodexRetryableStreamError.
  - response.failed with non-retryable error raises plain RuntimeError.
  - Missing envelope raises RuntimeError.
  - _CodexResponseShim's .json() returns the collected envelope, exposes
    .status_code / .headers / .raise_for_status.
"""
import json

import pytest

from IkaModel.codex.chat_helpers_codex import (
    _CodexResponseShim,
    _CodexRetryableStreamError,
    _collect_stream,
    _iter_sse,
)


class _FakeStreamResponse:
    """Provides .iter_lines() like an httpx streaming Response."""

    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        for line in self._lines:
            yield line


def _frame(event, obj):
    """Build the two-line SSE frame followed by the blank-line terminator."""
    return [f"event: {event}", f"data: {json.dumps(obj)}", ""]


# ----------------------------------------------------------------------
# _iter_sse — frame parser
# ----------------------------------------------------------------------

class TestIterSSE:

    def test_parses_event_and_data(self):
        lines = _frame("response.completed", {"type": "response.completed", "v": 1})
        got = list(_iter_sse(_FakeStreamResponse(lines)))
        assert len(got) == 1
        event_name, obj = got[0]
        assert event_name == "response.completed"
        assert obj == {"type": "response.completed", "v": 1}

    def test_ignores_done_sentinel(self):
        lines = ["data: [DONE]", ""]
        assert list(_iter_sse(_FakeStreamResponse(lines))) == []

    def test_skips_unparseable_data(self):
        lines = ["event: x", "data: <not-json>", ""]
        assert list(_iter_sse(_FakeStreamResponse(lines))) == []

    def test_blank_line_resets_event(self):
        lines = (
            _frame("response.created", {"type": "response.created"})
            + _frame("response.completed", {"type": "response.completed"})
        )
        events = [name for name, _ in _iter_sse(_FakeStreamResponse(lines))]
        assert events == ["response.created", "response.completed"]


# ----------------------------------------------------------------------
# _collect_stream — happy path & quirks
# ----------------------------------------------------------------------

class TestCollectStream:

    def test_envelope_with_collected_items(self):
        """codex backend ships response.completed with output:[] — the items
        arrive via output_item.done events. Collector must graft them."""
        lines = (
            _frame("response.created", {
                "type": "response.created",
                "response": {"id": "r1", "model": "gpt-5.5", "output": []},
            })
            + _frame("response.output_item.added", {
                "type": "response.output_item.added",
                "item": {"type": "message", "role": "assistant", "content": []},
            })
            + _frame("response.output_item.done", {
                "type": "response.output_item.done",
                "item": {
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
            })
            + _frame("response.completed", {
                "type": "response.completed",
                "response": {"id": "r1", "model": "gpt-5.5", "output": [],
                             "usage": {"total_tokens": 5}},
            })
        )
        envelope = _collect_stream(_FakeStreamResponse(lines))
        assert envelope["id"] == "r1"
        assert envelope["model"] == "gpt-5.5"
        assert envelope["usage"] == {"total_tokens": 5}
        # Grafted in from output_item.done events
        assert len(envelope["output"]) == 1
        assert envelope["output"][0]["content"][0]["text"] == "hello"

    def test_function_call_items_collected(self):
        lines = (
            _frame("response.created", {
                "type": "response.created",
                "response": {"id": "r1", "model": "x", "output": []},
            })
            + _frame("response.output_item.done", {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call", "call_id": "call_abc",
                    "name": "my_tool", "arguments": '{"a":1}',
                },
            })
            + _frame("response.completed", {
                "type": "response.completed",
                "response": {"id": "r1", "model": "x", "output": []},
            })
        )
        envelope = _collect_stream(_FakeStreamResponse(lines))
        fc = envelope["output"][0]
        assert fc["type"] == "function_call"
        assert fc["call_id"] == "call_abc"
        assert fc["name"] == "my_tool"

    def test_delta_fallback_when_no_done_items(self):
        """Defensive: if output_item.done never arrives, synthesize from deltas."""
        lines = (
            _frame("response.created", {
                "type": "response.created",
                "response": {"id": "r1", "model": "x", "output": []},
            })
            + _frame("response.output_text.delta", {
                "type": "response.output_text.delta", "delta": "Hel",
            })
            + _frame("response.output_text.delta", {
                "type": "response.output_text.delta", "delta": "lo!",
            })
            + _frame("response.completed", {
                "type": "response.completed",
                "response": {"id": "r1", "model": "x", "output": []},
            })
        )
        envelope = _collect_stream(_FakeStreamResponse(lines))
        text = envelope["output"][0]["content"][0]["text"]
        assert text == "Hello!"

    def test_retryable_failed_raises_retryable_error(self):
        lines = _frame("response.failed", {
            "type": "response.failed",
            "error": {"type": "server_error", "message": "hiccup"},
        })
        with pytest.raises(_CodexRetryableStreamError):
            _collect_stream(_FakeStreamResponse(lines))

    def test_official_nested_failed_error_is_retryable(self):
        lines = _frame("response.failed", {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {
                    "code": "server_is_overloaded",
                    "message": "busy",
                },
            },
        })
        with pytest.raises(_CodexRetryableStreamError):
            _collect_stream(_FakeStreamResponse(lines))

    def test_top_level_error_after_partial_output_is_retryable(self):
        lines = (
            _frame("response.created", {
                "type": "response.created",
                "response": {"id": "r1", "status": "in_progress", "output": []},
            })
            + _frame("response.output_text.delta", {
                "type": "response.output_text.delta",
                "delta": "partial",
            })
            + _frame("error", {
                "type": "error",
                "code": "server_is_overloaded",
                "message": "busy",
            })
        )
        with pytest.raises(_CodexRetryableStreamError):
            _collect_stream(_FakeStreamResponse(lines))

    def test_sse_event_name_is_used_when_json_type_is_absent(self):
        lines = (
            _frame("response.created", {
                "response": {"id": "r1", "output": []},
            })
            + _frame("response.completed", {
                "response": {
                    "id": "r1",
                    "status": "completed",
                    "output": [],
                },
            })
        )
        assert _collect_stream(_FakeStreamResponse(lines))["id"] == "r1"

    def test_non_retryable_failed_raises_plain_runtime_error(self):
        lines = _frame("response.failed", {
            "type": "response.failed",
            "error": {"type": "invalid_request_error", "message": "bad"},
        })
        with pytest.raises(RuntimeError) as exc_info:
            _collect_stream(_FakeStreamResponse(lines))
        assert not isinstance(exc_info.value, _CodexRetryableStreamError)

    def test_no_envelope_raises(self):
        """Stream that ends without created / in_progress / completed."""
        with pytest.raises(
            _CodexRetryableStreamError,
            match="before response.completed",
        ):
            _collect_stream(_FakeStreamResponse([]))

    def test_created_without_completed_is_retryable_truncation(self):
        lines = _frame("response.created", {
            "type": "response.created",
            "response": {
                "id": "r1",
                "status": "in_progress",
                "output": [],
            },
        })
        with pytest.raises(
            _CodexRetryableStreamError,
            match="after a response envelope",
        ):
            _collect_stream(_FakeStreamResponse(lines))

    @pytest.mark.parametrize("status", ["failed", "error"])
    def test_completed_event_rejects_explicit_failure_status(self, status):
        lines = (
            _frame("response.created", {
                "type": "response.created",
                "response": {"id": "r1", "output": []},
            })
            + _frame("response.completed", {
                "type": "response.completed",
                "response": {
                    "id": "r1",
                    "status": status,
                    "error": {"type": "invalid_request_error"},
                    "output": [],
                },
            })
        )
        with pytest.raises(RuntimeError) as exc_info:
            _collect_stream(_FakeStreamResponse(lines))
        assert not isinstance(
            exc_info.value,
            _CodexRetryableStreamError,
        )

    def test_incomplete_event_is_explicit_non_retryable_terminal(self):
        lines = _frame("response.incomplete", {
            "type": "response.incomplete",
            "response": {
                "id": "r1",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        })
        with pytest.raises(RuntimeError, match="incomplete") as exc_info:
            _collect_stream(_FakeStreamResponse(lines))
        assert not isinstance(
            exc_info.value,
            _CodexRetryableStreamError,
        )


# ----------------------------------------------------------------------
# _CodexResponseShim
# ----------------------------------------------------------------------

class TestResponseShim:

    def test_json_returns_data(self):
        data = {"id": "r1", "output": [], "usage": {"total_tokens": 3}}
        shim = _CodexResponseShim(data=data)
        assert shim.json() == data

    def test_status_and_headers(self):
        shim = _CodexResponseShim(
            data={}, status_code=200,
            headers={"x-codex-active-limit": "premium"},
        )
        assert shim.status_code == 200
        assert "x-codex-active-limit" in shim.headers

    def test_raise_for_status_on_400(self):
        shim = _CodexResponseShim(data={}, status_code=400)
        with pytest.raises(Exception):
            shim.raise_for_status()

    def test_raise_for_status_on_200(self):
        shim = _CodexResponseShim(data={}, status_code=200)
        # Should not raise
        shim.raise_for_status()
