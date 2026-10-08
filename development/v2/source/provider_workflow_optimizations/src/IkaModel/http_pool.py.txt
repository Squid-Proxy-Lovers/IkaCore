"""Process-wide HTTP connection pools shared by every IkaCore client.

Each agent (and each clone used for subagents and workflow nodes) still owns
its own ``httpx.Client``: owners close their clients exactly as before. What
is shared is the transport underneath, i.e. the pool of keep-alive
connections, so a request to a provider reuses an open TLS connection instead
of paying a fresh TCP + TLS handshake (two network round trips) per agent,
per Codex request, per summary call or per async chat round.

Clients reach the pool through thin wrapper transports whose ``close`` is a
no-op, so closing one client never tears down connections other clients use.
The pools honour the same SSL settings as :mod:`IkaModel.http_config`, are
rebuilt after ``fork`` (sockets must not be shared across processes) and
after the SSL environment changes, and place no cap on concurrent
connections so parallel agents never queue behind each other. Async pools are
per event loop, since connections cannot move between loops.
"""

# pyright: strict

from __future__ import annotations

import asyncio
import os
import threading
import weakref
from typing import Optional

import httpx

from .http_config import shared_ssl_context

_LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=64)

_lock = threading.Lock()
_sync_pool: Optional[httpx.HTTPTransport] = None
_sync_pool_key: Optional[tuple[int, int]] = None
_async_pools: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[tuple[int, int], httpx.AsyncHTTPTransport]]" = (
    weakref.WeakKeyDictionary()
)
_default_client: Optional[httpx.Client] = None


def _pool_key() -> tuple[int, int]:
    return os.getpid(), id(shared_ssl_context())


class _SharedTransport(httpx.BaseTransport):
    def __init__(self, pool: httpx.HTTPTransport) -> None:
        self._pool = pool

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._pool.handle_request(request)

    def close(self) -> None:  # the pool outlives any one client
        pass


class _SharedAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, pool: httpx.AsyncHTTPTransport) -> None:
        self._pool = pool

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._pool.handle_async_request(request)

    async def aclose(self) -> None:  # the pool outlives any one client
        pass


def _sync_transport() -> httpx.HTTPTransport:
    global _sync_pool, _sync_pool_key
    key = _pool_key()
    with _lock:
        if _sync_pool is None or _sync_pool_key != key:
            _sync_pool = httpx.HTTPTransport(verify=shared_ssl_context(), limits=_LIMITS)
            _sync_pool_key = key
        return _sync_pool


def _async_transport() -> httpx.AsyncHTTPTransport:
    loop = asyncio.get_running_loop()
    key = _pool_key()
    with _lock:
        cached = _async_pools.get(loop)
        if cached is None or cached[0] != key:
            cached = (key, httpx.AsyncHTTPTransport(verify=shared_ssl_context(), limits=_LIMITS))
            _async_pools[loop] = cached
        return cached[1]


def new_pooled_client(timeout: float) -> httpx.Client:
    """A caller-owned sync client whose connections come from the shared pool."""
    return httpx.Client(timeout=timeout, transport=_SharedTransport(_sync_transport()))


def new_pooled_async_client(timeout: float) -> httpx.AsyncClient:
    """A caller-owned async client pooled per running event loop."""
    return httpx.AsyncClient(timeout=timeout, transport=_SharedAsyncTransport(_async_transport()))


def default_pooled_client() -> httpx.Client:
    """Shared client for call sites that have no client of their own (e.g. summaries)."""
    global _default_client
    with _lock:
        client = _default_client
    if client is None or client.is_closed or getattr(client, "_ika_pool_key", None) != _pool_key():
        client = new_pooled_client(timeout=900.0)
        setattr(client, "_ika_pool_key", _pool_key())
        with _lock:
            _default_client = client
    return client


__all__ = ["default_pooled_client", "new_pooled_async_client", "new_pooled_client"]
