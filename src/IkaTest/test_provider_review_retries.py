import asyncio

import httpx
import pytest

from IkaModel.request_interface import api_request_retry, async_api_request_retry


@pytest.mark.parametrize("error_type,expected", [
    (httpx.UnsupportedProtocol, 1), (httpx.LocalProtocolError, 1),
    (httpx.InvalidURL, 1), (httpx.TooManyRedirects, 1),
    (httpx.ConnectError, 3), (httpx.ReadError, 3), (httpx.RemoteProtocolError, 3),
    (httpx.ProxyError, 3), (httpx.PoolTimeout, 3),
])
@pytest.mark.parametrize("url", ["https://fixture.test", "https://fixture.test/backend-api/codex/responses"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_only_transient_http_failures_are_retried(error_type, expected, asynchronous, url):
    seen = []

    def fail(request):
        seen.append(request)
        raise error_type("fixture failure")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            await async_api_request_retry(url, {}, {}, client=client, wait_seconds=0)

    if asynchronous:
        with pytest.raises(Exception):
            asyncio.run(run())
    else:
        with httpx.Client(transport=httpx.MockTransport(fail)) as client, pytest.raises(Exception):
            api_request_retry(url, {}, {}, client=client, wait_seconds=0)
    assert len(seen) == expected
