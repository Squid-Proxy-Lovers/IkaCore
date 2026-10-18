"""Opt-in request lifecycle handling without changing ordinary client behavior."""

# pyright: strict

from __future__ import annotations

import socket
import threading
from typing import Any, Optional

import httpx

from IkaCore.agent_runtime_payloads import JsonDict

from .request_control import check_request_controls, effective_timeout, request_lifecycle_active
from .request_lifecycle import await_with_controls, request_scope
from .runtime_policy import current_runtime_options


class _SyncRequestAbort:
    """Abort one HTTP/1 connection; never close a caller's connection pool."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.stream: Any = None
        self.socket: Optional[socket.socket] = None
        self.response: Optional[httpx.Response] = None

    def _bind_stream(self, stream: Any) -> None:
        if stream is not None:
            sock = stream.get_extra_info("socket")
            if isinstance(sock, socket.socket):
                self.socket = sock

    def trace(self, event: str, info: dict[str, Any]) -> None:
        # HTTPX's trace/network_stream extensions are optional on custom transports.
        if event.endswith("response_closed.started"):
            self.clear()
            return
        if event.endswith(".started"):
            check_request_controls()
        with self.lock:
            if event in {"connection.connect_tcp.complete", "connection.connect_unix_socket.complete",
                         "connection.start_tls.complete"}:
                self.stream = info["return_value"]
            elif event.startswith("http11."):
                self._bind_stream(self.stream)
            elif event.startswith("http2."):
                self.socket = None  # A multiplexed connection belongs to other requests too.
        request = info.get("request")
        if request is not None:
            timeouts = request.extensions.get("timeout", {})
            request.extensions["timeout"] = {key: effective_timeout(value) for key, value in timeouts.items()}

    def bind_response(self, response: httpx.Response) -> None:
        check_request_controls()
        with self.lock:
            self.response = response
            if response.extensions.get("http_version") != b"HTTP/2":
                self._bind_stream(response.extensions.get("network_stream"))

    def close(self) -> None:
        with self.lock:
            if self.socket is not None:
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.socket = None
            response = self.response
        if response is not None:
            response.close()

    def clear(self) -> None:
        with self.lock:
            self.stream, self.socket, self.response = None, None, None


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
    if client is None:
        with new_sync_client(effective_timeout(timeout)) as owned:
            return controlled_post(api_url, headers, payload, timeout, owned)
    abort = _SyncRequestAbort()
    with request_scope(abort.close):
        try:
            with client.stream("POST", api_url, headers=headers, json=payload,
                               timeout=effective_timeout(timeout), extensions={"trace": abort.trace}) as response:
                abort.bind_response(response)
                response.read()
                check_request_controls()
                return response
        except (httpx.HTTPError, OSError):
            check_request_controls()
            raise
        finally:
            abort.clear()


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
