import asyncio
from contextlib import nullcontext

import httpx
import pytest

from IkaModel.request_control import request_controls
from IkaModel.request_interface import api_request_retry, async_api_request_retry


@pytest.mark.parametrize("controlled", [False, True])
def test_borrowed_clients_apply_request_timeout_without_changing_defaults(controlled):
    seen = []

    def respond(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={})

    async def run():
        async with httpx.AsyncClient(timeout=None, transport=httpx.MockTransport(respond)) as client:
            with request_controls(cancel_checker=lambda: False) if controlled else nullcontext():
                await async_api_request_retry("https://fixture.test", {}, {}, timeout=0.01, client=client)
            assert client.timeout.read is None
            assert not client.is_closed

    with httpx.Client(timeout=None, transport=httpx.MockTransport(respond)) as client:
        with request_controls(cancel_checker=lambda: False) if controlled else nullcontext():
            api_request_retry("https://fixture.test", {}, {}, timeout=0.01, client=client)
        assert client.timeout.read is None
        assert not client.is_closed
    asyncio.run(run())
    assert seen == [dict(connect=0.01, read=0.01, write=0.01, pool=0.01)] * 2
