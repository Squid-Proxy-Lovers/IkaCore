import asyncio

import pytest

from IkaModel.async_runner import run_async


async def loop_identity():
    return asyncio.get_running_loop()


def test_default_uses_distinct_closed_event_loops():
    first, second = run_async(loop_identity), run_async(loop_identity)
    assert first is not second
    assert first.is_closed() and second.is_closed()


def test_nested_run_refuses_before_creating_coroutine_or_cancelling_outer_tasks():
    calls = []

    def factory():
        calls.append(True)
        return loop_identity()

    async def outer():
        sibling = asyncio.create_task(asyncio.sleep(0.01, result="alive"))
        with pytest.raises(RuntimeError, match="active event loop"):
            run_async(factory)
        assert await sibling == "alive"

    asyncio.run(outer())
    assert not calls
