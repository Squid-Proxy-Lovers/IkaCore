"""Select a scoped transport only when the caller opts into reuse."""

# pyright: strict

from typing import Any

import httpx

from IkaCore.agent_runtime_payloads import JsonDict

from ..request_control import request_lifecycle_active
from ..runtime_policy import current_runtime_options


def open_stream(api_url: str, headers: dict[str, str], payload: JsonDict, timeout: float | None) -> Any:
    options = current_runtime_options()
    if options.reuse_connections and not request_lifecycle_active():
        from ..connection_scope import scoped_client
        client = scoped_client(api_url, headers, timeout if timeout is not None else 900.0, options.shared_ssl_context)
        if client is not None:
            return client.stream("POST", api_url, headers=headers, json=payload, timeout=timeout)
    if options.shared_ssl_context:
        from ..http_configuration import shared_ssl_context
        return httpx.stream("POST", api_url, headers=headers, json=payload, timeout=timeout, verify=shared_ssl_context())
    return httpx.stream("POST", api_url, headers=headers, json=payload, timeout=timeout)
