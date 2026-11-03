import os
import socket
import sys
from pathlib import Path

import pytest

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def pytest_configure(config):
    config.addinivalue_line('filterwarnings', 'error::ResourceWarning')
    config.addinivalue_line('filterwarnings', 'error::pytest.PytestUnraisableExceptionWarning')
    config.addinivalue_line('markers', 'loopback_network: allow connections only to a local 127.0.0.1 test server')


@pytest.fixture(autouse=True)
def offline_network_guard(monkeypatch, request):
    if os.getenv("IKACORE_LIVE_PROVIDER_TESTS") == "1":
        return
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    loopback = request.node.get_closest_marker('loopback_network') is not None
    def connect(sock, address, *, original):
        if loopback and sock.family == socket.AF_INET and isinstance(address, tuple) and address[0] == '127.0.0.1':
            return original(sock, address)
        raise AssertionError("Offline tests must use a mock transport; a real network connection was attempted")
    monkeypatch.setattr(socket.socket, "connect", lambda sock, address: connect(sock, address, original=original_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", lambda sock, address: connect(sock, address, original=original_connect_ex))
