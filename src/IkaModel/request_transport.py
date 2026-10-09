"""Opt-in request lifecycle handling without changing ordinary client behavior."""

# pyright: strict

from __future__ import annotations

from typing import Optional

import httpx

from IkaCore.agent_runtime_payloads import JsonDict

from .request_control import check_request_controls, effective_timeout, request_lifecycle_active
from .request_lifecycle import await_with_controls, request_scope
from .runtime_policy import current_runtime_options


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
    with request_scope(client.close):
        response = client.post(api_url, headers=headers, json=payload, timeout=effective_timeout(timeout))
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
