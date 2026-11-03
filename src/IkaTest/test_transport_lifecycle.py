import asyncio
import json
import socket
import threading
from concurrent.futures import Future, TimeoutError
from unittest.mock import Mock

import httpx
import pytest

from IkaModel import runtime_options
from IkaModel.codex.stream_transport import open_stream
from IkaModel.request_control import request_controls
from IkaModel.request_interface import api_request_retry, async_api_request_retry
from IkaModel.request_lifecycle import async_sleep_with_controls, await_with_controls, request_scope, wait_for_tool
from IkaModel.request_transport import controlled_post, new_async_client
from IkaModel.runtime_errors import IkaRequestCancelled, IkaRequestDeadlineExceeded

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


@pytest.mark.parametrize('codex', [False, True])
@pytest.mark.parametrize('mode', ['deadline', 'cancel'])
def test_sync_abort_closes_only_response_and_preserves_caller_client(codex, mode):
    released, cancelled = threading.Event(), threading.Event()
    callbacks, requests = [], []
    body = b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n' if codex else b'{"ok":true}'
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            if mode == 'cancel':
                cancelled.set()
                callbacks[0]()
            assert released.wait(1), 'the individual response was not aborted'
            yield body
        def close(self):
            released.set()
    class Transport(httpx.MockTransport):
        def close(self):
            released.set()
            super().close()
    def respond(request):
        requests.append(request)
        return httpx.Response(200, stream=Stream()) if len(requests) == 1 else httpx.Response(200, content=body)
    def register(callback):
        callbacks.append(callback)
        return lambda: callbacks.remove(callback)
    url = URL + ('/backend-api/codex/responses' if codex else '/v1/messages')
    with httpx.Client(transport=Transport(respond)) as client:
        error = IkaRequestDeadlineExceeded if mode == 'deadline' else IkaRequestCancelled
        with request_controls(timeout=0.03 if mode == 'deadline' else None,
                              cancel_checker=cancelled.is_set, abort_registrar=register), pytest.raises(error):
            api_request_retry(url, {}, {}, client=client)
        assert released.wait(1) and not callbacks
        assert not client.is_closed
        assert api_request_retry(url, {}, {}, client=client).status_code == 200


@pytest.mark.parametrize('mode', ['deadline', 'cancel'])
def test_async_post_abort_preserves_caller_client(mode):
    async def run():
        started, closed = asyncio.Event(), asyncio.Event()
        callbacks = []
        cancelled = False
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                await asyncio.Event().wait()
                yield b''
            async def aclose(self):
                closed.set()
        def register(callback):
            callbacks.append(callback)
            return lambda: callbacks.remove(callback)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))) as client:
            with request_controls(timeout=0.02 if mode == 'deadline' else None,
                                  cancel_checker=lambda: cancelled, abort_registrar=register):
                task = asyncio.create_task(async_api_request_retry(URL, {}, {}, client=client))
                await started.wait()
                if mode == 'cancel':
                    cancelled = True
                    callbacks[0]()
                with pytest.raises(IkaRequestDeadlineExceeded if mode == 'deadline' else IkaRequestCancelled):
                    await task
            assert closed.is_set() and not client.is_closed and not callbacks
    asyncio.run(run())


def test_sync_request_clamps_each_timeout_without_changing_caller_configuration():
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={})
    with httpx.Client(timeout=900, transport=httpx.MockTransport(respond)) as client:
        with request_controls(timeout=.2):
            controlled_post(URL, {}, {}, 2, client)
        timeouts = requests[0].extensions['timeout']
        assert set(timeouts) == {'connect', 'read', 'write', 'pool'}
        assert all(0 < value <= .2 for value in timeouts.values())
        assert client.timeout == httpx.Timeout(900)


@pytest.mark.parametrize('error_type', [httpx.ReadError, httpx.ReadTimeout])
def test_sync_transport_errors_preserve_their_type_without_an_execution_abort(error_type):
    def respond(request):
        raise error_type('transport failed')
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with request_controls(cancel_checker=lambda: False), pytest.raises(error_type):
            controlled_post(URL, {}, {}, 2, client)


@pytest.mark.parametrize('multiplexed', [False, True])
def test_sync_abort_never_accesses_transport_sockets(multiplexed):
    cancelled, released = threading.Event(), threading.Event()
    callbacks = []
    sock = Mock(spec=socket.socket)
    sock.shutdown.side_effect = OSError('already closed')
    network = Mock()
    network.get_extra_info.return_value = sock
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            cancelled.set()
            callbacks[0]()
            assert released.wait(1)
            yield b'{}'
        def close(self):
            released.set()
    def respond(request):
        trace = request.extensions['trace']
        trace('connection.connect_tcp.complete', {'return_value': network})
        trace('connection.start_tls.complete', {'return_value': network})
        trace(('http2' if multiplexed else 'http11') + '.send_request_headers.started', {'request': request})
        return httpx.Response(200, stream=Stream(), extensions={
            'http_version': b'HTTP/2' if multiplexed else b'HTTP/1.1', 'network_stream': network,
        })
    def register(callback):
        callbacks.append(callback)
        return lambda: callbacks.remove(callback)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with request_controls(cancel_checker=cancelled.is_set, abort_registrar=register), pytest.raises(IkaRequestCancelled):
            controlled_post(URL, {}, {}, 2, client)
        assert not client.is_closed and not callbacks
    network.get_extra_info.assert_not_called()
    sock.shutdown.assert_not_called()


@pytest.mark.parametrize('phase', ['transport', 'hook', 'client'])
def test_slow_user_code_cannot_hold_the_sync_caller(monkeypatch, phase):
    import time

    import IkaModel.request_transport as transport
    entered, release = threading.Event(), threading.Event()
    def stall(*args, **kwargs):
        entered.set()
        assert release.wait(2)
    class Borrowed(httpx.MockTransport):
        def close(self):
            raise AssertionError('caller-owned custom transport was closed')
    def respond(request):
        if phase == 'transport':
            stall()
        return httpx.Response(200, json={})
    client = httpx.Client(transport=Borrowed(respond), event_hooks={'response': [stall] if phase == 'hook' else []})
    def create(timeout):
        stall()
        return client
    if phase == 'client':
        monkeypatch.setattr(transport, 'new_sync_client', create)
    try:
        started = time.monotonic()
        with request_controls(timeout=.04), pytest.raises(IkaRequestDeadlineExceeded):
            controlled_post(URL, {}, {}, 10, None if phase == 'client' else client)
        assert entered.is_set() and time.monotonic() - started < .25
        assert not client.is_closed
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name.startswith('ika-http-'):
                thread.join(2)
                assert not thread.is_alive()
        # The factory case creates an owned client; the test deliberately gives
        # it a transport whose close raises to check that late cleanup is consumed.
        if phase != 'client':
            client._transport = httpx.MockTransport(respond)
            client.close()


@pytest.mark.parametrize('error', [ValueError('hook failed'), KeyboardInterrupt(), SystemExit(7)])
def test_sync_worker_preserves_exception_identity(error):
    def respond(request):
        raise error
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with request_controls(cancel_checker=lambda: False), pytest.raises(type(error)) as raised:
            controlled_post(URL, {}, {}, 10, client)
        assert raised.value is error


def test_closed_caller_client_fails_without_reopening():
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    client.close()
    with request_controls(timeout=1), pytest.raises(RuntimeError, match='client has been closed'):
        controlled_post(URL, {}, {}, 10, client)


def test_private_pool_copies_configuration_and_preserves_trace():
    from IkaModel.request_transport import _RequestTransport, _SyncRequestAbort
    events = []
    with httpx.HTTPTransport(verify=False, proxy='http://localhost:9999', retries=2,
                             local_address='127.0.0.1', socket_options=[(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]) as source:
        abort = _SyncRequestAbort()
        private = _RequestTransport(source, abort)
        pool, original = private.transport._pool, source._pool
        assert type(pool) is type(original) and pool is not original
        for key, value in original.__dict__.items():
            if key not in {'_connections', '_requests', '_optional_thread_lock', '_max_keepalive_connections'}:
                assert getattr(pool, key) is value
        assert pool._requests is not original._requests and pool._connections is not original._connections
        assert pool._optional_thread_lock is not original._optional_thread_lock
        assert pool._max_keepalive_connections == 0
        # A hook's trace is composed with ours, rather than silently replaced.
        request = httpx.Request('POST', URL, extensions={'trace': lambda event, info: events.append(event)})
        original_handle = private.transport.handle_request
        private.transport.handle_request = lambda request: httpx.Response(200)
        try:
            private.handle_request(request)
            request.extensions['trace']('http11.send_request_headers.started', {'request': request})
            assert events == ['http11.send_request_headers.started']
        finally:
            private.transport.handle_request = original_handle
            private.close()


def test_connection_arriving_after_abort_is_shutdown_and_closed():
    from IkaModel.request_transport import _SyncRequestAbort
    abort = _SyncRequestAbort()
    abort.abort()
    duplicate = Mock(spec=socket.socket)
    duplicate.shutdown.side_effect = OSError('already closed')
    stream = Mock()
    stream.get_extra_info.return_value.dup.return_value = duplicate
    abort.own_connection('connection.connect_tcp.complete', {'return_value': stream})
    duplicate.shutdown.assert_called_once_with(socket.SHUT_RDWR)
    duplicate.close.assert_called_once()
    assert not abort.sockets


def test_custom_response_arriving_after_abort_is_closed():
    from IkaModel.request_transport import _SyncRequestAbort
    closed = threading.Event()
    class Stream(httpx.SyncByteStream):
        def close(self):
            closed.set()
    abort = _SyncRequestAbort()
    abort.abort()
    abort.bind_response(httpx.Response(200, stream=Stream()))
    assert closed.is_set()


def test_controlled_client_retains_mounts_auth_cookies_redirects_and_hooks():
    seen, hooks = [], []
    def respond(request):
        seen.append(request)
        if request.url.path == '/start':
            return httpx.Response(302, headers={'Location': '/finish', 'Set-Cookie': 'first=1; Path=/'})
        return httpx.Response(200, json={}, headers={'Set-Cookie': 'second=2; Path=/'})
    mounted = httpx.MockTransport(respond)
    with httpx.Client(base_url=URL, headers={'X-Client': 'preserved'}, cookies={'initial': 'yes'},
                      auth=('user', 'password'), params={'query': 'value'}, follow_redirects=True,
                      event_hooks={'response': [lambda response: hooks.append(response.status_code)]},
                      mounts={'https://': mounted, 'http://': None}) as client:
        original_pool = client._transport._pool
        with request_controls(timeout=1):
            assert controlled_post('/start', {'X-Request': 'also-preserved'}, {}, 10, client).json() == {}
        assert hooks == [302, 200] and len(seen) == 2
        assert all(request.headers['X-Client'] == 'preserved' and request.headers['X-Request'] == 'also-preserved'
                   and request.headers['Authorization'].startswith('Basic ') for request in seen)
        assert seen[0].url.params['query'] == 'value'
        assert 'initial=yes' in seen[0].headers['Cookie'] and 'first=1' in seen[1].headers['Cookie']
        assert client.cookies.get('first') == '1' and client.cookies.get('second') == '2'
        assert client._transport._pool is original_pool and not client.is_closed


@pytest.mark.parametrize('async_requests', [False, True])
def test_retry_sleep_never_schedules_past_the_execution_deadline(async_requests):
    from IkaModel.request_lifecycle import sleep_with_controls
    intervals = []
    def sleep(delay):
        intervals.append(delay)
        raise ValueError('stop before sleeping')
    async def async_sleep(delay):
        sleep(delay)
    async def run():
        await async_sleep_with_controls(10, async_sleep)
    with request_controls(timeout=.03), pytest.raises(ValueError, match='stop before sleeping'):
        if async_requests:
            asyncio.run(run())
        else:
            sleep_with_controls(10, sleep)
    assert len(intervals) == 1 and 0 < intervals[0] <= .03


@pytest.mark.parametrize('error_type', [OSError, RuntimeError, httpx.ReadError])
def test_custom_abort_cleanup_errors_do_not_escape_background_thread(error_type):
    released = threading.Event()
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            assert released.wait(2)
            yield b'{}'
        def close(self):
            released.set()
            raise error_type('close failed after releasing the reader')
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))) as client:
        with request_controls(timeout=.03), pytest.raises(IkaRequestDeadlineExceeded):
            controlled_post(URL, {}, {}, 10, client)
        for thread in threading.enumerate():
            if thread.name.startswith('ika-http-'):
                thread.join(2)
                assert not thread.is_alive()
        assert released.is_set() and not client.is_closed


def test_custom_response_cancelled_before_httpx_adopts_it_is_closed():
    from IkaModel.request_transport import _RequestTransport, _SyncRequestAbort
    cancelled, closed = threading.Event(), threading.Event()
    class Stream(httpx.SyncByteStream):
        def close(self):
            closed.set()
    def respond(request):
        cancelled.set()
        return httpx.Response(200, stream=Stream())
    with httpx.MockTransport(respond) as source:
        transport = _RequestTransport(source, _SyncRequestAbort())
        with request_controls(cancel_checker=cancelled.is_set), pytest.raises(IkaRequestCancelled):
            transport.handle_request(httpx.Request('POST', URL))
        assert closed.is_set(), 'a transport response escaped ownership during cancellation'
