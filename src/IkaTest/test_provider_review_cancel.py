import asyncio
import time

import httpx
import pytest

from IkaModel.request_control import request_controls
from IkaModel.request_interface import async_api_request_retry
from IkaModel.request_lifecycle import await_with_controls
from IkaModel.runtime_errors import IkaRequestCancelled


@pytest.mark.loopback_network
def test_checker_only_cancellation_interrupts_real_socket_and_drains_request():
    async def run():
        cancelled = False
        disconnected = asyncio.Event()
        handlers = []
        loop = asyncio.get_running_loop()

        async def handle(reader, writer):
            nonlocal cancelled
            handlers.append(asyncio.current_task())
            await reader.readuntil(b"\r\n\r\n")
            def cancel():
                nonlocal cancelled
                cancelled = True
            loop.call_later(0.02, cancel)
            try:
                await asyncio.wait_for(reader.read(), timeout=0.3)
            except TimeoutError:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                disconnected.set()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server, httpx.AsyncClient(trust_env=False) as client:
            started = time.monotonic()
            with request_controls(cancel_checker=lambda: cancelled), pytest.raises(IkaRequestCancelled):
                await async_api_request_retry(f"http://127.0.0.1:{port}", {}, {}, client=client, timeout=1)
            elapsed = time.monotonic() - started
            await asyncio.wait_for(disconnected.wait(), 1)
            assert not client.is_closed
            assert asyncio.current_task().cancelling() == 0
        await asyncio.gather(*handlers)
        assert elapsed < 0.15, elapsed
    asyncio.run(run())


@pytest.mark.parametrize("both", [False, True])
def test_external_task_cancellation_keeps_cancelled_error_and_count(both):
    async def run():
        cancelled = False
        drained = asyncio.Event()

        async def operation():
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()

        async def worker():
            with request_controls(cancel_checker=lambda: cancelled):
                return await await_with_controls(operation)

        task = asyncio.create_task(worker())
        await asyncio.sleep(0.01)
        cancelled = both
        task.cancel("external")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelling() == 1
        assert drained.is_set()
    asyncio.run(run())
