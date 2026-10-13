"""Native async Codex transport; caller-owned clients retain their connection pools."""

# pyright: strict
# pyright: reportPrivateUsage=false

import asyncio
from typing import Optional

import httpx

from IkaCore.agent_runtime_payloads import JsonDict

from ..request_control import check_request_controls, effective_max_retries, effective_timeout
from ..request_lifecycle import async_sleep_with_controls, await_with_controls
from ..request_transport import new_async_client
from ..retry_policy import backoff_seconds
from .chat_helpers_codex import _codex_rate_headers, _CodexResponseShim, _streaming_codex_payload, codex_retry_delay
from .codex_stream import CodexRetryableStreamError, async_collect_stream


async def _attempt(client: httpx.AsyncClient, url: str, headers: dict[str, str], payload: JsonDict,
                   timeout: float, attempt: int, attempts: int, wait: float) -> _CodexResponseShim | float:
    async with client.stream("POST", url, headers=headers, json=payload, timeout=effective_timeout(timeout)) as response:
        if response.status_code == 200:
            return _CodexResponseShim(await async_collect_stream(response), headers=_codex_rate_headers(response))
        body = (await response.aread()).decode("utf-8", "replace")
        return codex_retry_delay(response, body, attempt, attempts, wait)


async def async_request_codex(url: str, headers: dict[str, str], payload: JsonDict,
                              timeout: float = 900.0, max_retries: int = 3, wait_seconds: float = 1.0,
                              client: Optional[httpx.AsyncClient] = None) -> _CodexResponseShim:
    check_request_controls()
    attempts = effective_max_retries(max_retries)
    owned = client is None
    transport = client if client is not None else new_async_client(timeout)
    normalized = _streaming_codex_payload(payload)
    try:
        for attempt in range(attempts):
            try:
                result = await await_with_controls(
                    lambda attempt=attempt: _attempt(transport, url, headers, normalized, timeout, attempt, attempts, wait_seconds))
                if isinstance(result, _CodexResponseShim):
                    return result
                delay = result
            except (CodexRetryableStreamError, httpx.HTTPError):
                check_request_controls()
                if attempt == attempts - 1:
                    raise
                delay = backoff_seconds(attempt, wait_seconds)
            await async_sleep_with_controls(delay, asyncio.sleep)
        raise RuntimeError("codex request failed without specific exception")
    finally:
        if owned:
            await transport.aclose()
