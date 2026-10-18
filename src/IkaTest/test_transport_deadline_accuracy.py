"""Hard sync HTTP budgets, exclusive connections and prompt worker reclamation."""

import gc
import json
import socket
import socketserver
import ssl
import subprocess
import threading
import time
from contextlib import contextmanager

import httpx
import pytest

from IkaModel.request_control import request_controls
from IkaModel.request_interface import api_request_retry
from IkaModel.request_transport import controlled_post
from IkaModel.runtime_errors import IkaRequestCancelled, IkaRequestDeadlineExceeded

pytestmark = pytest.mark.loopback_network


@pytest.fixture(scope='module')
def tls_context(tmp_path_factory):
    root = tmp_path_factory.mktemp('deadline-tls')
    key, cert = root / 'key.pem', root / 'cert.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(cert), '-days', '1', '-subj', '/CN=localhost'],
                   check=True, capture_output=True)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


@contextmanager
def deadline_server(tls_context, scenario, tls):
    entered, release = threading.Event(), threading.Event()
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                if scenario == 'handshake':
                    entered.set()
                    release.wait(5)
                    return
                while self.rfile.readline() not in {b'\r\n', b''}:
                    pass
                entered.set()
                if scenario == 'headers':
                    self.wfile.write(b'HTTP/1.1 200 OK\r\nX-Trickle: ')
                    for _ in range(60):
                        self.wfile.write(b'x')
                        if release.wait(.03):
                            return
                elif scenario == 'body':
                    self.wfile.write(b'HTTP/1.1 200 OK\r\nContent-Length: 60\r\n\r\n')
                    for _ in range(60):
                        self.wfile.write(b'x')
                        if release.wait(.03):
                            return
                elif scenario == 'sse':
                    self.wfile.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n'
                                     b'Transfer-Encoding: chunked\r\n\r\n')
                    data = b'data: {"type":"response.created","response":{"status":"in_progress"}}\n\n'
                    self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
                    release.wait(5)
                else:
                    release.wait(5)  # TCP accepts, but never sends response headers.
            except (OSError, ssl.SSLError):
                pass
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
    server = Server(('127.0.0.1', 0), Handler)
    if tls and scenario != 'handshake':
        server.socket = tls_context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01))
    worker.start()
    scheme = 'https' if tls else 'http'
    url = f'{scheme}://127.0.0.1:{server.server_address[1]}'
    if scenario == 'sse':
        url += '/backend-api/codex/responses'
    try:
        yield url, entered
    finally:
        release.set()
        server.shutdown()
        worker.join(2)
        server.server_close()  # Join every server handler, too.
        assert not worker.is_alive()


def join_request_workers():
    for thread in threading.enumerate():
        if thread.name in {'ika-http-request', 'ika-http-close'}:
            thread.join(1.5)
            assert not thread.is_alive(), f'leaked request worker: {thread.name}'
    gc.collect()


@pytest.mark.parametrize('tls,scenario', [(tls, scenario) for tls in [False, True]
                                         for scenario in ['headers', 'body', 'accept', 'sse']]
                         + [(True, 'handshake')])
@pytest.mark.parametrize('mode', ['deadline', 'cancel', 'checker'])
def test_sync_wall_clock_budget_and_worker_cleanup(tls_context, monkeypatch, tls, scenario, mode):
    cancelled = threading.Event()
    callbacks, duplicates, timing = [], [], {}
    original_dup = socket.socket.dup
    def duplicate(sock):
        result = original_dup(sock)
        duplicates.append(result)
        return result
    monkeypatch.setattr(socket.socket, 'dup', duplicate)
    def register(callback):
        callbacks.append(callback)
        return lambda: callbacks.remove(callback)
    with deadline_server(tls_context, scenario, tls) as (url, entered):
        with httpx.Client(verify=False, trust_env=False) as client:
            def cancel():
                assert entered.wait(2)
                time.sleep(.05)  # Let the client enter the deliberately stalled I/O phase.
                timing['cancelled'] = time.monotonic()
                cancelled.set()
                if mode == 'cancel':
                    callbacks[0]()
            trigger = None
            if mode != 'deadline':
                trigger = threading.Thread(target=cancel)
                trigger.start()
            try:
                started = time.monotonic()
                with request_controls(timeout=.2 if mode == 'deadline' else None,
                                      cancel_checker=cancelled.is_set,
                                      abort_registrar=None if mode == 'checker' else register):
                    with pytest.raises(IkaRequestDeadlineExceeded if mode == 'deadline' else IkaRequestCancelled):
                        api_request_retry(url, {}, {}, timeout=10, client=client, max_retries=1)
                elapsed = time.monotonic() - (started if mode == 'deadline' else timing['cancelled'])
                print('TIMING', json.dumps({'scenario': scenario, 'tls': tls, 'mode': mode, 'elapsed': elapsed}))
                assert elapsed < (.4 if mode == 'deadline' else .2)
                assert not callbacks and not client.is_closed
                join_request_workers()  # Before the stalled server is released.
                assert duplicates and all(sock.fileno() == -1 for sock in duplicates)
            finally:
                if trigger is not None:
                    trigger.join(3)
                    assert not trigger.is_alive()


def test_uncontrolled_request_has_no_worker_or_pool_copy(monkeypatch):
    import IkaModel.request_transport as transport
    def unexpected(*args, **kwargs):
        raise AssertionError('ordinary request entered the controlled path')
    monkeypatch.setattr(transport, 'controlled_sync_request', unexpected)
    monkeypatch.setattr(transport.copy, 'copy', unexpected)
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={'ok': True}))) as client:
        assert controlled_post('https://example.invalid', {}, {}, 1, client).json() == {'ok': True}


def test_dns_can_finish_later_without_delaying_the_sync_caller(tls_context, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = socket.getaddrinfo
    def resolve(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)
    monkeypatch.setattr(socket, 'getaddrinfo', resolve)
    with deadline_server(tls_context, 'accept', False) as (url, _), httpx.Client(trust_env=False) as client:
        try:
            started = time.monotonic()
            with request_controls(timeout=.04), pytest.raises(IkaRequestDeadlineExceeded):
                controlled_post(url, {}, {}, 10, client)
            assert entered.is_set() and time.monotonic() - started < .25
            assert not client.is_closed
        finally:
            release.set()
            join_request_workers()
