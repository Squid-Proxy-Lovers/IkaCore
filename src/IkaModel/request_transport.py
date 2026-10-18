"""Opt-in request lifecycle handling without changing ordinary client behavior."""

# pyright: strict
# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
import socket
import threading
from collections.abc import Callable
from functools import partial
from typing import Any, Optional, TypeVar

import httpx

from IkaCore.agent_runtime_payloads import JsonDict

from .request_control import check_request_controls, effective_timeout, request_lifecycle_active
from .request_lifecycle import _quiet_close, await_with_controls, run_sync_with_controls
from .runtime_errors import IkaRequestControlError
from .runtime_policy import current_runtime_options

T = TypeVar("T")


class _SyncRequestAbort:
    """Only sockets from private pools may be interrupted.

    Duplicate descriptors retain the request's socket identity across TLS wrapping
    and response-hook pool release. They never refer to the caller's pool. Closing
    an HTTPX client/response alone does not reliably wake a blocked recv on all OSes.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sockets: list[socket.socket] = []
        self.stream_closers: list[Callable[[], None]] = []
        self.response: Optional[httpx.Response] = None
        self.aborted = False

    def trace(self, event: str, info: dict[str, Any]) -> None:
        if event.endswith(".started"):
            check_request_controls()
        request = info.get("request")
        if request is not None:
            timeouts = request.extensions.get("timeout", {})
            request.extensions["timeout"] = {key: effective_timeout(value) for key, value in timeouts.items()}

    def own_connection(self, event: str, info: dict[str, Any]) -> None:
        connected = event.endswith((".connect_tcp.complete", ".connect_unix_socket.complete"))
        if connected or event.endswith(".start_tls.complete"):
            stream = info["return_value"]
            # A trace callback can fail before httpcore attaches this stream to
            # its connection. Retain cleanup ownership through that hand-off.
            with self.lock:
                self.stream_closers.append(stream.close)
        if connected:
            # A dup also interrupts a TLS handshake while wrap_socket owns the
            # original descriptor, before start_tls.complete can expose it.
            duplicate = info["return_value"].get_extra_info("socket").dup()
            with self.lock:
                if self.aborted:
                    self._shutdown(duplicate)
                    duplicate.close()
                else:
                    self.sockets.append(duplicate)
        self.trace(event, info)

    @staticmethod
    def _shutdown(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def bind_response(self, response: httpx.Response) -> None:
        with self.lock:
            self.response = response
            aborted = self.aborted
        if aborted:
            response.close()
        try:
            check_request_controls()
        except IkaRequestControlError:
            # HTTPX has not adopted this transport response yet, so its normal
            # send/context-manager cleanup cannot see it when we raise here.
            self._close_response(response)
            raise

    def abort(self) -> None:
        with self.lock:
            self.aborted = True
            for sock in self.sockets:
                self._shutdown(sock)
            response = self.response
            self.response = None
        # Custom transports have no owned socket. Their response close may run
        # arbitrary user code, so it cannot be allowed to hold up the caller.
        if response is not None and not response.is_closed:
            threading.Thread(target=self._close_response, args=(response,), name="ika-http-close", daemon=True).start()

    @staticmethod
    def _close_response(response: httpx.Response) -> None:
        try:
            response.close()
        except (httpx.HTTPError, OSError, RuntimeError):
            pass

    def clear(self) -> None:
        with self.lock:
            sockets, closers = self.sockets, self.stream_closers
            self.sockets, self.stream_closers = [], []
            self.response = None
        try:
            for close in closers:
                _quiet_close(close)
        finally:
            for sock in sockets:
                sock.close()


class _RequestTransport(httpx.BaseTransport):
    """Clone HTTPX/httpcore 1.x configuration, never live pool state.

    HTTPX has no public clone API. Both supported HTTPX versions use httpcore
    1.x, whose pools keep mutable request/connection state in these three fields.
    Shallow copying preserves TLS, proxy (including mounts), HTTP/2, UDS, local
    address, retry and socket-option configuration without rebuilding SSL contexts.
    Custom transports are borrowed and are never closed or inspected for sockets.
    """

    def __init__(self, source: httpx.BaseTransport, abort: _SyncRequestAbort) -> None:
        self.abort = abort
        self.owned = type(source) is httpx.HTTPTransport
        self.transport = copy.copy(source) if self.owned else source
        if self.owned:
            pool = copy.copy(getattr(source, "_pool"))
            pool._connections = []
            pool._requests = []
            pool._optional_thread_lock = type(pool._optional_thread_lock)()
            pool._max_keepalive_connections = 0
            setattr(self.transport, "_pool", pool)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        previous = request.extensions.get("trace")
        def trace(event: str, info: dict[str, Any]) -> None:
            if self.owned:
                self.abort.own_connection(event, info)
            else:
                self.abort.trace(event, info)
            if previous is not None:
                previous(event, info)
        request.extensions["trace"] = trace
        response = self.transport.handle_request(request)
        if not self.owned:
            self.abort.bind_response(response)
        return response

    def close(self) -> None:
        if self.owned:
            self.transport.close()


def controlled_sync_request(operation: Callable[[httpx.Client], T], timeout: Optional[float],
                            client: Optional[httpx.Client]) -> T:
    """Run one controlled HTTP operation with request-exclusive connections."""
    abort = _SyncRequestAbort()
    def work() -> T:
        try:
            check_request_controls()
            if client is None:
                with new_sync_client(effective_timeout(timeout)) as source:
                    return send(source)
            if client.is_closed:
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            return send(client)
        finally:
            abort.clear()

    def send(source: httpx.Client) -> T:
        check_request_controls()
        # Retain client subclasses, auth, hooks, defaults, cookies and routing.
        # Only transport state and the client's lifecycle become request-owned.
        isolated = copy.copy(source)
        isolated._transport = _RequestTransport(source._transport, abort)
        isolated._mounts = {pattern: None if transport is None else _RequestTransport(transport, abort)
                            for pattern, transport in source._mounts.items()}
        try:
            return operation(isolated)
        finally:
            isolated.close()

    return run_sync_with_controls(work, abort.abort)


def controlled_post(
    api_url: str, headers: dict[str, str], payload: JsonDict, timeout: float, client: Optional[httpx.Client]
) -> httpx.Response:
    if not request_lifecycle_active():
        options = current_runtime_options()
        if client is None and options.reuse_connections:
            from .connection_scope import scoped_client
            client = scoped_client(api_url, headers, timeout, options.shared_ssl_context)
        if client is None and options.shared_ssl_context:
            from .http_configuration import shared_ssl_context
            return httpx.post(api_url, headers=headers, json=payload, timeout=timeout, verify=shared_ssl_context())
        post = client.post if client is not None else httpx.post
        return post(api_url, headers=headers, json=payload, timeout=timeout)
    return controlled_sync_request(partial(_read_post, api_url, headers, payload, timeout), timeout, client)


def _read_post(api_url: str, headers: dict[str, str], payload: JsonDict, timeout: float,
               owned: httpx.Client) -> httpx.Response:
    with owned.stream("POST", api_url, headers=headers, json=payload,
                      timeout=effective_timeout(timeout)) as response:
        response.read()
        check_request_controls()
        return response


async def controlled_async_post(
    api_url: str, headers: dict[str, str], payload: JsonDict, timeout: float, client: httpx.AsyncClient
) -> httpx.Response:
    if not request_lifecycle_active():
        return await client.post(api_url, headers=headers, json=payload)
    response = await await_with_controls(lambda: client.post(
        api_url, headers=headers, json=payload, timeout=effective_timeout(timeout)
    ))
    check_request_controls()
    return response


def new_async_client(timeout: float) -> httpx.AsyncClient:
    if current_runtime_options().shared_ssl_context:
        from .http_configuration import shared_ssl_context
        return httpx.AsyncClient(timeout=timeout, verify=shared_ssl_context())
    return httpx.AsyncClient(timeout=timeout)


def new_sync_client(timeout: Optional[float]) -> httpx.Client:
    if current_runtime_options().shared_ssl_context:
        from .http_configuration import shared_ssl_context
        return httpx.Client(timeout=timeout, verify=shared_ssl_context())
    return httpx.Client(timeout=timeout)
