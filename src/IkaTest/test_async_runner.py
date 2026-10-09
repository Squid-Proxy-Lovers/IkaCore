import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

import pytest

from IkaModel import runtime_options
from IkaModel.async_runner import close_thread_runner, run_async


@pytest.fixture(autouse=True)
def clean_runner():
    yield
    close_thread_runner()


async def loop_identity():
    return asyncio.get_running_loop()


def test_default_uses_distinct_closed_event_loops():
    first, second = run_async(loop_identity), run_async(loop_identity)
    assert first is not second
    assert first.is_closed() and second.is_closed()


def test_opt_in_reuses_thread_loop_and_closes_explicitly():
    with runtime_options(persistent_async_runner=True):
        first, second = run_async(loop_identity), run_async(loop_identity)
    assert first is second and not first.is_closed()
    close_thread_runner()
    assert first.is_closed()
    close_thread_runner()


def test_each_call_copies_current_context_and_cleans_background_tasks():
    value = ContextVar("runner_test", default="default")
    cancelled = []

    async def background():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    async def operation():
        old = value.get()
        value.set("child-only")
        asyncio.create_task(background())
        await asyncio.sleep(0)
        return old

    with runtime_options(persistent_async_runner=True):
        value.set("first")
        assert run_async(operation) == "first"
        value.set("second")
        assert run_async(operation) == "second"
        assert value.get() == "second"
    assert cancelled == [True, True]


def test_nested_run_refuses_before_creating_coroutine_or_cancelling_outer_tasks():
    calls = []

    def factory():
        calls.append(True)
        return loop_identity()

    async def outer():
        sibling = asyncio.create_task(asyncio.sleep(0.01, result="alive"))
        with runtime_options(persistent_async_runner=True), pytest.raises(RuntimeError, match="active event loop"):
            run_async(factory)
        assert await sibling == "alive"

    asyncio.run(outer())
    assert not calls


def test_thread_workers_do_not_share_event_loops():
    barrier = threading.Barrier(2)

    def worker():
        with runtime_options(persistent_async_runner=True):
            loop = run_async(loop_identity)
            barrier.wait(timeout=1)
            close_thread_runner()
            return loop

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: worker(), range(2)))
    assert first is not second
    assert first.is_closed() and second.is_closed()


def test_failure_cleans_remaining_tasks_and_next_call_still_works():
    async def fail():
        asyncio.create_task(asyncio.sleep(10))
        raise ValueError("expected")

    with runtime_options(persistent_async_runner=True):
        with pytest.raises(ValueError):
            run_async(fail)
        assert not run_async(loop_identity).is_closed()
