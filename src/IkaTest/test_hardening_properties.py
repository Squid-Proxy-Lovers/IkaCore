"""Generative contracts: framing, structured state, malformed responses and arguments."""

import asyncio
import copy
import json
import tempfile
from pathlib import Path

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from IkaCore.snapshot_privacy import sanitized_state
from IkaCore.snapshot_store import SnapshotStore
from IkaModel.chat_interface.chat_request import _parse_provider_round, _response_json_or_raise
from IkaModel.chat_interface.tool_execution_sync import _decode_tool_arguments, validate_tool_args
from IkaModel.codex.codex_stream import async_collect_stream, collect_stream
from IkaModel.runtime_errors import IkaProviderResponseError

json_scalars = st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(max_size=60)
json_values = st.recursive(json_scalars, lambda x: st.lists(x, max_size=5) | st.dictionaries(st.text(max_size=12), x, max_size=5), max_leaves=20)


class Chunks(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, data, cuts):
        points = [0, *sorted({n % (len(data) + 1) for n in cuts}), len(data)]
        self.chunks = [data[a:b] for a, b in zip(points, points[1:], strict=False)]

    def __iter__(self):
        yield from self.chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


@settings(max_examples=70, deadline=None, derandomize=True)
@given(st.text(max_size=100), st.lists(st.integers(min_value=0, max_value=10000), max_size=60), st.sampled_from(["\n", "\r\n", "\r"]))
def test_sse_arbitrary_byte_boundaries_preserve_unicode_and_terminal_output(text, cuts, newline):
    events = [": heartbeat", "data: {malformed", "data: [DONE]", "data: null",
              "data: " + json.dumps({"type": "response.output_text.delta", "delta": text}, ensure_ascii=False),
              "data: " + json.dumps({"type": "response.completed", "response": {"status": "completed", "id": "ok"}})]
    raw = (newline.join(events) + newline * 2).encode()
    sync = collect_stream(httpx.Response(200, stream=Chunks(raw, cuts)))
    async def run():
        return await async_collect_stream(httpx.Response(200, stream=Chunks(raw, cuts)))
    asynchronous = asyncio.run(run())
    assert sync == asynchronous
    assert sync["id"] == "ok"
    assert sync["output"][0]["content"][0]["text"] == text


@settings(max_examples=45, deadline=None, derandomize=True)
@given(json_values)
def test_snapshot_state_roundtrip_is_immutable_and_chain_is_append_only(value):
    state = {"value": value, "counter": 1}
    original = copy.deepcopy(state)
    expected = sanitized_state(original)
    with tempfile.TemporaryDirectory() as directory:
        store = SnapshotStore(str(Path(directory) / "state.db"))
        run = store.create_run("agent", "property")
        frame = store.create_frame(run, "agent", "property")
        first = store.create_snapshot(run, frame, "first", state)
        second = store.create_snapshot(run, frame, "second", {"counter": 2}, prev_snapshot_id=first)
        assert store.load_snapshot(first)["state_json"] == expected
        assert state == original
        assert [s["snapshot_id"] for s in store.get_snapshot_chain(second)] == [first, second]
        assert store.load_snapshot(first)["state_json"]["value"] == expected["value"]


@settings(max_examples=80, derandomize=True)
@given(st.dictionaries(st.text(max_size=20), json_values, max_size=8))
def test_tool_json_decode_roundtrip_and_validation_do_not_mutate(arguments):
    before = copy.deepcopy(arguments)
    decoded, error = _decode_tool_arguments("fixture", json.dumps(arguments))
    assert error is None and decoded == arguments
    validated = validate_tool_args("fixture", decoded)
    assert set(validated) == set(arguments)
    assert arguments == before
    for key, value in arguments.items():
        if not isinstance(value, str):
            assert validated[key] == value


@settings(max_examples=50, derandomize=True)
@given(st.text(max_size=100))
def test_truncated_json_never_becomes_valid_tool_arguments(text):
    raw = json.dumps({"value": text})[:-1]
    decoded, error = _decode_tool_arguments("fixture", raw)
    assert decoded is None
    assert "Malformed JSON" in json.loads(error)["error"]
    with pytest.raises(IkaProviderResponseError, match="valid JSON"):
        _response_json_or_raise(httpx.Response(200, text=raw), "openai")


@pytest.mark.parametrize("provider", ["openai", "openai_responses", "anthropic", "deepseek", "gemini", "codex", "openrouter"])
@settings(max_examples=40, derandomize=True)
@given(st.sampled_from([None, [], "", 7, True]))
def test_malformed_provider_envelopes_have_typed_errors(provider, malformed):
    with pytest.raises(IkaProviderResponseError, match="JSON must be an object"):
        _response_json_or_raise(httpx.Response(200, content=json.dumps(malformed)), provider)
    # A missing choices/candidates collection is either a valid empty response or
    # a typed parse error, never an unwrapped indexing/attribute failure.
    try:
        result = _parse_provider_round(provider, {}, "fixture")
    except IkaProviderResponseError:
        pass
    else:
        assert result[0] == "" and result[2] == []


@pytest.mark.parametrize("provider", ["openai", "openai_responses", "anthropic", "deepseek", "gemini", "codex", "openrouter"])
@settings(max_examples=80, derandomize=True)
@given(st.dictionaries(st.sampled_from(["choices", "content", "candidates", "output", "usage"]), json_values, max_size=5))
def test_nested_malformed_provider_shapes_never_escape_typed_parse_errors(provider, envelope):
    try:
        result = _parse_provider_round(provider, envelope, "fixture")
    except IkaProviderResponseError:
        return
    assert isinstance(result[0], str) and isinstance(result[2], list) and isinstance(result[3], int)
