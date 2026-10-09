import asyncio
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import httpx
import pytest

from IkaModel import request_interface as requests
from IkaModel.chat_interface.tool_execution_async import async_execute_tool
from IkaModel.chat_interface.tool_execution_sync import execute_tool
from IkaModel.codex import chat_helpers_codex as codex
from IkaModel.request_control import (
    check_request_controls,
    effective_max_retries,
    register_request_abort_callback,
    request_cancelled,
    request_controls,
)
from IkaModel.request_lifecycle import await_with_controls, request_scope, sleep_with_controls
from IkaModel.runtime_errors import IkaRequestCancelled, IkaRequestDeadlineExceeded
from IkaModel.worker_context import submit_with_controls

URL = "https://example.invalid/v1/messages"


def test_control_errors_keep_api_error_family_and_legacy_errors_pickle():
    assert isinstance(IkaRequestCancelled("cancelled"), requests.IkaAPIError)
    error = requests.IkaHTTPError("failed", status_code=503)
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is requests.IkaHTTPError
    assert restored.status_code == 503
    assert requests.IkaAPIError.__module__ == "IkaModel.request_interface"


def test_default_request_controls_preserve_retry_count():
    assert effective_max_retries(3) == 3
    assert effective_max_retries(0) == 0
    assert not request_cancelled()


def test_nested_controls_restore_after_exception():
    with request_controls(max_retries=2):
        with pytest.raises(ValueError), request_controls(max_retries=1):
            assert effective_max_retries(3) == 1
            raise ValueError("test")
        assert effective_max_retries(3) == 2
    assert effective_max_retries(3) == 3


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_invalid_deadline_does_not_modify_controls(timeout):
    with pytest.raises(ValueError), request_controls(timeout=timeout):
        pass
    check_request_controls()


def test_cancelled_request_never_contacts_provider(monkeypatch):
    post = Mock()
    monkeypatch.setattr(requests.httpx, "post", post)
    with request_controls(cancel_checker=lambda: True), pytest.raises(IkaRequestCancelled):
        requests.api_request_retry(URL, {}, {})
    post.assert_not_called()


def test_zero_deadline_never_contacts_provider(monkeypatch):
    post = Mock()
    monkeypatch.setattr(requests.httpx, "post", post)
    with request_controls(timeout=0), pytest.raises(IkaRequestDeadlineExceeded):
        requests.api_request_retry(URL, {}, {})
    post.assert_not_called()


def test_bounded_retry_override_keeps_default_post_transport(monkeypatch):
    response = httpx.Response(503, json={"error": "busy"})
    post = Mock(return_value=response)
    monkeypatch.setattr(requests.httpx, "post", post)
    with request_controls(max_retries=1), pytest.raises(requests.IkaAPIError):
        requests.api_request_retry(URL, {}, {}, wait_seconds=0)
    assert post.call_count == 1


def test_retry_wait_observes_cancellation_without_sleeping_full_delay():
    cancelled = threading.Event()
    waits = []

    def sleeper(delay):
        waits.append(delay)
        cancelled.set()

    with request_controls(cancel_checker=cancelled.is_set), pytest.raises(IkaRequestCancelled):
        sleep_with_controls(30, sleeper)
    assert len(waits) == 1 and waits[0] <= 0.05


def test_request_scope_unregisters_and_ignores_stale_abort_callbacks():
    callbacks = []
    cleanup = []
    close = Mock()

    def register(callback):
        callbacks.append(callback)
        return lambda: cleanup.append(True)

    with request_controls(abort_registrar=register), request_scope(close):
        pass
    assert cleanup == [True]
    callbacks[0]()
    close.assert_not_called()


def test_sync_deadline_closes_request_and_unregisters_abort():
    released = threading.Event()
    registered = []

    class Client:
        calls = 0

        def close(self):
            released.set()

        def post(self, *args, **kwargs):
            self.calls += 1
            assert released.wait(1)
            return httpx.Response(200, json={"ok": True})

    def register(callback):
        registered.append(callback)
        return lambda: registered.remove(callback)

    client = Client()
    with request_controls(timeout=0.03, abort_registrar=register), pytest.raises(IkaRequestDeadlineExceeded):
        requests.api_request_retry(URL, {}, {}, client=client)
    assert client.calls == 1 and not registered


def test_sync_tool_and_worker_receive_controls():
    with request_controls(max_retries=1):
        assert execute_tool("inspect", {}, {"inspect": lambda args: effective_max_retries(3)}) == "1"
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert submit_with_controls(pool, effective_max_retries, 3).result() == 1
    assert effective_max_retries(3) == 3


def test_sync_tool_cancellation_propagates_without_error_json():
    cancelled = threading.Event()

    def tool(args):
        cancelled.set()
        return "ignored"

    with request_controls(cancel_checker=cancelled.is_set), pytest.raises(IkaRequestCancelled):
        execute_tool("inspect", {}, {"inspect": tool})


def test_async_contexts_are_isolated_and_sync_tools_inherit():
    async def job(attempts):
        with request_controls(max_retries=attempts):
            await asyncio.sleep(0)
            return await async_execute_tool("inspect", {}, {"inspect": lambda args: effective_max_retries(3)})

    async def run():
        return await asyncio.gather(job(1), job(2))

    assert asyncio.run(run()) == ["1", "2"]
    assert effective_max_retries(3) == 3


def test_async_deadline_cancels_transport_and_never_retries():
    class Client:
        calls = 0
        cancelled = False

        async def post(self, *args, **kwargs):
            self.calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

    client = Client()

    async def run():
        with request_controls(timeout=0.03), pytest.raises(IkaRequestDeadlineExceeded):
            await requests.async_api_request_retry(URL, {}, {}, client=client)

    asyncio.run(run())
    assert client.calls == 1 and client.cancelled


def test_async_abort_unregisters_and_does_not_leave_parent_cancelled():
    callbacks = []
    cancelled = threading.Event()

    def register(callback):
        callbacks.append(callback)
        return lambda: callbacks.remove(callback)

    async def run():
        entered = asyncio.Event()

        async def operation():
            entered.set()
            await asyncio.Event().wait()

        async def cancel():
            await entered.wait()
            cancelled.set()
            callbacks[0]()

        with request_controls(cancel_checker=cancelled.is_set, abort_registrar=register):
            trigger = asyncio.create_task(cancel())
            with pytest.raises(IkaRequestCancelled):
                await await_with_controls(operation)
            await trigger
            assert asyncio.current_task().cancelling() == 0
        assert not callbacks

    asyncio.run(run())


def test_async_codex_worker_keeps_cancel_checker_and_registrar(monkeypatch):
    registered = []

    def register(callback):
        registered.append(callback)
        return lambda: registered.remove(callback)

    def fake_request(*args, **kwargs):
        assert effective_max_retries(3) == 1
        assert not request_cancelled()
        cleanup = register_request_abort_callback(lambda: None)
        assert cleanup is not None
        cleanup()
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(codex, "request_codex", fake_request)

    async def run():
        with request_controls(max_retries=1, cancel_checker=lambda: False, abort_registrar=register):
            await requests.async_api_request_retry("https://example.invalid/backend-api/codex/responses", {}, {})

    asyncio.run(run())
    assert not registered
