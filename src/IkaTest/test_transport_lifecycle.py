import asyncio
import json
import threading
from concurrent.futures import Future, TimeoutError
from unittest.mock import Mock

import httpx
import pytest

from IkaModel import runtime_options
from IkaModel.codex.stream_transport import open_stream
from IkaModel.request_control import request_controls
from IkaModel.request_lifecycle import async_sleep_with_controls, await_with_controls, request_scope, wait_for_tool
from IkaModel.request_transport import controlled_post, new_async_client
from IkaModel.runtime_errors import IkaRequestCancelled

URL = 'https://example.invalid'


@pytest.fixture
def owned_clients(monkeypatch):
    original = httpx.Client
    created = []
    def create(**kwargs):
        client = original(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={'ok': True})), **kwargs)
        created.append(client)
        return client
    monkeypatch.setattr(httpx, 'Client', create)
    return created


def test_scoped_posts_and_streams_share_owned_client_and_close_it(owned_clients):
    with runtime_options(reuse_connections=True, shared_ssl_context=True):
        assert controlled_post(URL, {}, {}, 1, None).json() == {'ok': True}
        with open_stream(URL, {}, {}, None) as response:
            assert json.loads(response.read()) == {'ok': True}
        assert len(owned_clients) == 1 and not owned_clients[0].is_closed
        with runtime_options(reuse_connections=True):
            controlled_post(URL, {}, {}, 1, None)
            assert len(owned_clients) == 1
    assert owned_clients[0].is_closed


def test_deadline_owned_client_is_closed_after_success(owned_clients):
    with runtime_options(shared_ssl_context=True), request_controls(timeout=1):
        assert controlled_post(URL, {}, {}, 1, None).status_code == 200
    assert len(owned_clients) == 1 and owned_clients[0].is_closed


def test_shared_ssl_option_reaches_unpooled_post_and_stream(monkeypatch):
    from IkaModel.http_configuration import shared_ssl_context
    post, stream = Mock(return_value=httpx.Response(200)), Mock(return_value='stream')
    monkeypatch.setattr(httpx, 'post', post)
    monkeypatch.setattr(httpx, 'stream', stream)
    with runtime_options(shared_ssl_context=True):
        controlled_post(URL, {}, {}, 1, None)
        assert open_stream(URL, {}, {}, 1) == 'stream'
        assert post.call_args.kwargs['verify'] is shared_ssl_context()
        assert stream.call_args.kwargs['verify'] is shared_ssl_context()
        async def own_async():
            client = new_async_client(1)
            await client.aclose()
            assert client.is_closed
        asyncio.run(own_async())


def test_external_async_cancellation_is_preserved():
    async def run():
        entered = asyncio.Event()
        async def operation():
            entered.set()
            await asyncio.Event().wait()
        with request_controls(cancel_checker=lambda: False):
            task = asyncio.create_task(await_with_controls(operation))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
    asyncio.run(run())


def test_async_retry_sleep_observes_cancel_and_zero_delay_never_sleeps():
    cancelled = threading.Event()
    calls = []
    async def sleeper(delay):
        calls.append(delay)
        cancelled.set()
    async def run():
        with request_controls(cancel_checker=cancelled.is_set):
            await async_sleep_with_controls(0, sleeper)
            assert calls == []
            with pytest.raises(IkaRequestCancelled):
                await async_sleep_with_controls(30, sleeper)
    asyncio.run(run())
    assert len(calls) == 1 and calls[0] <= 0.05


def test_tool_wait_distinguishes_result_tool_failure_and_own_timeout():
    completed, failed, pending = Future(), Future(), Future()
    completed.set_result('done')
    error = TimeoutError('raised by tool')
    failed.set_exception(error)
    with request_controls(cancel_checker=lambda: False):
        assert wait_for_tool(completed, None) == 'done'
        with pytest.raises(TimeoutError) as raised:
            wait_for_tool(failed, 1)
        assert raised.value is error
        with pytest.raises(TimeoutError):
            wait_for_tool(pending, 0)


def test_abort_cleanup_failures_do_not_replace_request_failure():
    callbacks = []
    def fail():
        raise OSError('already closed')
    def register(callback):
        callbacks.append(callback)
        return fail
    with request_controls(abort_registrar=register), pytest.raises(ValueError, match='request failed'), request_scope(fail):
        callbacks[0]()
        raise ValueError('request failed')
