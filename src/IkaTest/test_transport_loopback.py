"""Real sockets must wake on request abort; response.close mocks cannot establish this."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from IkaModel.request_control import request_controls
from IkaModel.request_interface import api_request_retry
from IkaModel.runtime_errors import IkaRequestCancelled, IkaRequestDeadlineExceeded

pytestmark = pytest.mark.loopback_network


@pytest.fixture
def blocked_server():
    servers = []
    def create(phase, *, trickle=False):
        entered, release = threading.Event(), threading.Event()
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def send_headers(self, length):
                self.send_response(200)
                self.send_header('Content-Length', str(length))
                self.end_headers()
                self.wfile.flush()
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                if self.path == '/ok':
                    self.send_headers(2)
                    self.wfile.write(b'{}')
                    return
                if phase == 'body':
                    self.send_headers(10 if trickle else 2)
                entered.set()
                try:
                    if trickle:
                        release.wait(.4)
                        self.wfile.write(b'x')
                        self.wfile.flush()
                    release.wait(3)
                    if phase == 'headers':
                        self.send_headers(2)
                    self.wfile.write(b'{}')
                    self.wfile.flush()
                except OSError:
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
        thread.start()
        servers.append((server, thread, release))
        return f'http://127.0.0.1:{server.server_port}', entered
    yield create
    for server, thread, release in servers:
        release.set()
        server.shutdown()
        thread.join(2)
        server.server_close()


@pytest.mark.parametrize('phase', ['headers', 'body'])
@pytest.mark.parametrize('mode', ['deadline', 'cancel'])
@pytest.mark.parametrize('owned', [False, True])
def test_real_socket_abort_wakes_headers_and_body_and_preserves_client(blocked_server, monkeypatch, phase, mode, owned):
    url, entered = blocked_server(phase)
    cancelled, done = threading.Event(), threading.Event()
    callbacks, errors, clients = [], [], []
    def register(callback):
        callbacks.append(callback)
        return lambda: callbacks.remove(callback)
    def make_client(timeout):
        client = httpx.Client(timeout=timeout, trust_env=False)
        clients.append(client)
        return client
    monkeypatch.setattr('IkaModel.request_transport.new_sync_client', make_client)
    client = None if owned else make_client(2)
    def work():
        try:
            with request_controls(timeout=.2 if mode == 'deadline' else None,
                                  cancel_checker=cancelled.is_set, abort_registrar=register):
                api_request_retry(url + '/blocked', {}, {}, timeout=2, client=client, max_retries=1)
        except Exception as error:
            errors.append(error)
        finally:
            done.set()
    worker = threading.Thread(target=work, daemon=True)
    started = time.monotonic()
    worker.start()
    try:
        assert entered.wait(1)
        assert len(callbacks) == 1
        if mode == 'cancel':
            started = time.monotonic()
            cancelled.set()
            callbacks[0]()
        assert done.wait(.6), 'request remained blocked after abort'
        elapsed = time.monotonic() - started
        assert elapsed < (.6 if mode == 'deadline' else .3)
        expected = IkaRequestDeadlineExceeded if mode == 'deadline' else IkaRequestCancelled
        assert len(errors) == 1 and isinstance(errors[0], expected)
        assert not callbacks
        assert clients[0].is_closed == owned
        if client is not None:
            assert api_request_retry(url + '/ok', {}, {}, client=client).json() == {}
    finally:
        worker.join(3)
        for made in clients:
            made.close()


def test_trickled_body_cannot_restart_the_execution_deadline(blocked_server):
    url, _ = blocked_server('body', trickle=True)
    with httpx.Client(trust_env=False) as client:
        started = time.monotonic()
        with request_controls(timeout=.5), pytest.raises(IkaRequestDeadlineExceeded):
            api_request_retry(url + '/blocked', {}, {}, timeout=2, client=client, max_retries=1)
        assert time.monotonic() - started < .7
        assert not client.is_closed
        assert api_request_retry(url + '/ok', {}, {}, client=client).json() == {}


def test_reused_connection_header_wait_is_bounded_by_remaining_deadline(blocked_server):
    url, _ = blocked_server('headers')
    with httpx.Client(trust_env=False) as client:
        api_request_retry(url + '/ok', {}, {}, client=client)
        started = time.monotonic()
        with request_controls(timeout=.2), pytest.raises(IkaRequestDeadlineExceeded):
            api_request_retry(url + '/blocked', {}, {}, timeout=2, client=client, max_retries=1)
        assert time.monotonic() - started < .5
        assert not client.is_closed
