"""Native streaming, retries, cancellation, and ownership with offline transports."""

import asyncio
import json

import httpx
import pytest

from IkaModel.codex import async_request
from IkaModel.request_control import request_controls
from IkaModel.request_interface import api_request_retry, async_api_request_retry
from IkaModel.runtime_errors import IkaRequestCancelled, IkaRequestDeadlineExceeded

URL = 'https://example.invalid/backend-api/codex/responses'
COMPLETE = b'event: response.completed\ndata: {"response":{"status":"completed","output":[]}}\n\n'


@pytest.mark.parametrize('asynchronous', [False, True])
def test_codex_reuses_supplied_client_and_preserves_payload(asynchronous):
    seen = []
    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=COMPLETE, headers={'x-codex-fixture': 'retained'})
    payload = {'stream': False}
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            for _ in range(2):
                response = await async_api_request_retry(URL, {}, payload, client=client)
                assert response.headers['x-codex-fixture'] == 'retained' and not client.is_closed
    if asynchronous:
        asyncio.run(run())
    else:
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            for _ in range(2):
                assert api_request_retry(URL, {}, payload, client=client).json()['status'] == 'completed'
                assert not client.is_closed
    assert seen == [{'stream': True}, {'stream': True}] and payload == {'stream': False}


@pytest.mark.parametrize('first', [408, 409, 429, 500, 'incomplete', 'transport', 'transient'])
def test_native_codex_retries_transient_failures(first, monkeypatch):
    calls, sleeps = [], []
    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            if first == 'transport':
                raise httpx.ConnectError('fixture')
            if first == 'incomplete':
                return httpx.Response(200, content=b': keepalive\ndata: broken\ndata: [DONE]\ndata: []\n\n')
            if first == 'transient':
                return httpx.Response(200, content=b'data: {"type":"error","error":{"code":"server_error"}}\n\n')
            return httpx.Response(first, json={'error': 'fixture'}, headers={'retry-after': '0.5'})
        return httpx.Response(200, content=COMPLETE)
    async def sleep(delay, sleeper):
        sleeps.append(delay)
    monkeypatch.setattr(async_request, 'async_sleep_with_controls', sleep)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            assert (await async_api_request_retry(URL, {}, {}, client=client)).json()['status'] == 'completed'
    asyncio.run(run())
    assert len(calls) == 2 and len(sleeps) == 1
    if first == 429:
        assert sleeps[0] >= 0.5


@pytest.mark.parametrize('status', [400, 401, 429, 500])
def test_native_codex_terminal_errors_close_owned_client(status, monkeypatch):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status, json={'error': 'fixture'})))
    monkeypatch.setattr(async_request, 'new_async_client', lambda timeout: client)
    with pytest.raises(RuntimeError, match=str(status)):
        asyncio.run(async_api_request_retry(URL, {}, {}, max_retries=1))
    assert client.is_closed


@pytest.mark.parametrize('mode', ['deadline', 'request_cancel', 'external_cancel'])
def test_native_codex_cancellation_closes_stream_but_keeps_shared_client(mode):
    async def run():
        started, closed = asyncio.Event(), asyncio.Event()
        callbacks = []
        cancelled = False
        class BlockingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                await asyncio.Event().wait()
                yield b''
            async def aclose(self):
                closed.set()
        def register(callback):
            callbacks.append(callback)
            return lambda: callbacks.remove(callback)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=BlockingStream()))) as client:
            with request_controls(timeout=0.02 if mode == 'deadline' else None, cancel_checker=lambda: cancelled, abort_registrar=register):
                task = asyncio.create_task(async_api_request_retry(URL, {}, {}, client=client))
                await started.wait()
                if mode == 'external_cancel':
                    task.cancel()
                elif mode == 'request_cancel':
                    cancelled = True
                    callbacks[0]()
                expected = {'deadline': IkaRequestDeadlineExceeded, 'request_cancel': IkaRequestCancelled, 'external_cancel': asyncio.CancelledError}[mode]
                with pytest.raises(expected):
                    await task
            assert closed.is_set() and not client.is_closed and not callbacks
    asyncio.run(run())


def test_native_codex_exhausted_stream_error_and_empty_budget():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b''))) as client:
            with pytest.raises(RuntimeError, match=r'before response\.completed'):
                await async_api_request_retry(URL, {}, {}, client=client, max_retries=1)
            with pytest.raises(RuntimeError, match='without specific exception'):
                await async_api_request_retry(URL, {}, {}, client=client, max_retries=0)
    asyncio.run(run())


def test_sync_codex_releases_retry_response_before_backoff(monkeypatch):
    responses = []
    def respond(request):
        response = httpx.Response(429, json={'error': 'fixture'}) if not responses else httpx.Response(200, content=COMPLETE)
        responses.append(response)
        return response
    def sleep(delay, sleeper):
        assert responses[0].is_closed
        assert not client.is_closed
    monkeypatch.setattr('IkaModel.codex.chat_helpers_codex.sleep_with_controls', sleep)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert api_request_retry(URL, {}, {}, client=client).status_code == 200
    assert len(responses) == 2
