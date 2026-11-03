import asyncio
import json

import httpx
import pytest

from IkaModel.codex.codex_stream import async_collect_stream, collect_stream, iter_sse


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_codex_multiline_sse_data_dispatches_on_blank_line(asynchronous, newline):
    lines = [": comment", "id: ignored", "retry: 1000", "event: response.completed",
             'data: {"type":"response.completed",', 'data: "response":{"status":"completed","output":[]}}',
             "", "data: [DONE]", "", ""]
    response = httpx.Response(200, text=newline.join(lines))
    result = asyncio.run(async_collect_stream(response)) if asynchronous else collect_stream(response)
    assert result == {"status": "completed", "output": []}


def test_sse_data_fields_without_colons_and_event_buffers_reset():
    response = httpx.Response(200, text='event: ignored\n\nevent: first\ndata\ndata: {"n":1}\n\n: comment\ndata: {"n":2}\n\ndata: {"unfinished":3}')
    assert list(iter_sse(response)) == [("first", {"n": 1}), (None, {"n": 2})]
    assert json.loads('\n{"n":1}') == {"n": 1}
