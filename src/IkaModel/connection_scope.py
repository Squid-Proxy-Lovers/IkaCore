"""Execution-owned sync clients; environment proxies remain managed by httpx."""

# pyright: strict

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .http_configuration import shared_ssl_context

_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy")


@dataclass
class _ConnectionState:
    clients: dict[tuple[str, tuple[Optional[str], ...], bool], httpx.Client] = field(default_factory=lambda: {})
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False


_state: ContextVar[Optional[_ConnectionState]] = ContextVar("ika_connection_scope", default=None)


@contextmanager
def connection_scope() -> Generator[None, None, None]:
    if _state.get() is not None:
        yield
        return
    state = _ConnectionState()
    token = _state.set(state)
    try:
        yield
    finally:
        _state.reset(token)
        with state.lock:
            state.closed = True
            clients = list(state.clients.values())
            state.clients.clear()
        for client in clients:
            client.close()


def scoped_client(api_url: str, headers: dict[str, str], timeout: float, reuse_ssl: bool) -> Optional[httpx.Client]:
    state = _state.get()
    if state is None:
        return None
    identity = json.dumps([api_url, sorted((key.lower(), value) for key, value in headers.items())])
    digest = hashlib.sha256(identity.encode()).hexdigest()
    key = (digest, tuple(os.environ.get(name) for name in _ENV), reuse_ssl)
    with state.lock:
        if state.closed:
            raise RuntimeError("execution connection scope is closed")
        client = state.clients.get(key)
        if client is None or client.is_closed:
            client = httpx.Client(timeout=timeout, verify=shared_ssl_context() if reuse_ssl else True,
                                  limits=httpx.Limits(max_connections=None, max_keepalive_connections=64))
            state.clients[key] = client
        return client
