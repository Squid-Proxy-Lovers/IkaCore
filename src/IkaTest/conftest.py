import os
import socket
import sys
from pathlib import Path

import pytest

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


@pytest.fixture(autouse=True)
def offline_network_guard(monkeypatch):
    if os.getenv("IKACORE_LIVE_PROVIDER_TESTS") == "1":
        return
    def deny_connection(*args, **kwargs):
        raise AssertionError("Offline tests must use a mock transport; a real network connection was attempted")
    monkeypatch.setattr(socket.socket, "connect", deny_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_connection)
