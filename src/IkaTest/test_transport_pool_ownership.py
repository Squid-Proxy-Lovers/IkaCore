"""HTTP/HTTPS pool reuse, including a preloaded response releasing its socket."""

import asyncio
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from IkaModel.request_control import request_controls
from IkaModel.request_transport import controlled_async_post, controlled_post
from IkaModel.runtime_errors import IkaRequestCancelled, IkaRequestDeadlineExceeded

pytestmark = pytest.mark.loopback_network


@pytest.fixture
def pooled_server(tmp_path):
    servers = []
    def make(*, tls=False, phase='body'):
        entered, release = threading.Event(), threading.Event()
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def handle(self):
                try:
                    super().handle()
                except ConnectionError:
                    pass  # Closing an aborted request can reset its keepalive connection.
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                blocked = self.path == '/block'
                try:
                    if blocked and phase == 'headers':
                        entered.set()
                        release.wait(5)
                    self.send_response(200)
                    self.send_header('Content-Length', '2')
                    self.end_headers()
                    self.wfile.flush()
                    if blocked and phase == 'body':
                        entered.set()
                        release.wait(5)
                    self.wfile.write(b'{}')
                    self.wfile.flush()
                except OSError:
                    pass
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        if tls:
            key, cert = tmp_path / 'key.pem', tmp_path / 'cert.pem'
            subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                '-keyout', str(key), '-out', str(cert), '-days', '1', '-subj', '/CN=localhost'],
                check=True, capture_output=True)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=.01), daemon=True)
        thread.start()
        servers.append((httpd, thread, release))
        return f'{"https" if tls else "http"}://127.0.0.1:{httpd.server_port}', entered, release
    yield make
    for httpd, thread, release in servers:
        release.set()
        httpd.shutdown()
        thread.join(2)
        httpd.server_close()


@pytest.mark.parametrize('tls', [False, True])
@pytest.mark.parametrize('phase', ['headers', 'body'])
@pytest.mark.parametrize('mode', ['cancel', 'deadline'])
def test_sync_socket_and_pool(pooled_server, tls, phase, mode):
    url, entered, release = pooled_server(tls=tls, phase=phase)
    cancelled, done = threading.Event(), threading.Event()
    callbacks, errors = [], []
    def register(callback):
        callbacks.append(callback)
        return lambda: callbacks.remove(callback)
    with httpx.Client(verify=False, trust_env=False, limits=httpx.Limits(max_connections=1)) as client:
        def run():
            try:
                with request_controls(timeout=.5 if mode == 'deadline' else None,
                        cancel_checker=cancelled.is_set, abort_registrar=register):
                    controlled_post(url + '/block', {}, {}, .3 if mode == 'cancel' else 3, client)
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            assert entered.wait(2)
            if mode == 'cancel':
                cancelled.set()
                callbacks[0]()
            assert done.wait(1.5)
            expected = IkaRequestCancelled if mode == 'cancel' else IkaRequestDeadlineExceeded
            assert len(errors) == 1 and isinstance(errors[0], expected), errors
            assert not client.is_closed
            for _ in range(3):
                assert client.post(url + '/ok', json={}).json() == {}
        finally:
            release.set()
            thread.join(3)


@pytest.mark.parametrize('tls', [False, True])
@pytest.mark.parametrize('phase', ['headers', 'body'])
@pytest.mark.parametrize('mode', ['cancel', 'deadline'])
def test_async_socket_and_pool(pooled_server, tls, phase, mode):
    url, entered, release = pooled_server(tls=tls, phase=phase)
    callbacks = []
    cancelled = threading.Event()
    def register(callback):
        callbacks.append(callback)
        return lambda: callbacks.remove(callback)
    async def run():
        async with httpx.AsyncClient(verify=False, trust_env=False, limits=httpx.Limits(max_connections=1)) as client:
            async def request():
                with request_controls(timeout=.5 if mode == 'deadline' else None,
                        cancel_checker=cancelled.is_set, abort_registrar=register):
                    return await controlled_async_post(url + '/block', {}, {}, 3, client)
            task = asyncio.create_task(request())
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                if mode == 'cancel':
                    cancelled.set()
                    callbacks[0]()
                expected = IkaRequestCancelled if mode == 'cancel' else IkaRequestDeadlineExceeded
                with pytest.raises(expected):
                    await asyncio.wait_for(task, 1.5)
                assert not client.is_closed
                for _ in range(3):
                    assert (await client.post(url + '/ok', json={})).json() == {}
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize('tls', [False, True])
def test_preloaded_response_does_not_abort_another_request(pooled_server, monkeypatch, tls):
    url, entered, release = pooled_server(tls=tls)
    cancelled = threading.Event()
    callbacks, errors = [], []
    original_read = httpx.Response.read
    def register(callback):
        callbacks.append(callback)
        return lambda: callbacks.remove(callback)
    def preload(response):
        if response.request.url.path == '/ok':
            response.read()  # Supported response hook: consumes body and releases pooled connection.
    with httpx.Client(verify=False, trust_env=False, event_hooks={'response': [preload]},
            limits=httpx.Limits(max_connections=1)) as client:
        second = None
        def run_other():
            try:
                client.post(url + '/block', json={})
            except BaseException as exc:
                errors.append(exc)
        def read_then_switch(response):
            nonlocal second
            if response.request.url.path != '/ok' or not response.is_closed:
                return original_read(response)
            # Pause the content-cache read, after the body was consumed by the hook.
            second = threading.Thread(target=run_other, daemon=True)
            second.start()
            assert entered.wait(2)
            cancelled.set()
            callbacks[0]()
            release.set()
            second.join(2)
            return original_read(response)
        monkeypatch.setattr(httpx.Response, 'read', read_then_switch)
        try:
            with request_controls(cancel_checker=cancelled.is_set, abort_registrar=register):
                with pytest.raises(IkaRequestCancelled):
                    controlled_post(url + '/ok', {}, {}, 3, client)
            assert not errors
        finally:
            release.set()
            if second is not None:
                second.join(3)
