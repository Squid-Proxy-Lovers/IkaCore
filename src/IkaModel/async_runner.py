"""Optional thread-owned event loops for repeated synchronous async calls."""

# pyright: strict

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Coroutine
from contextvars import copy_context
from typing import Any, TypeVar

from .runtime_policy import current_runtime_options

T = TypeVar("T")
_local = threading.local()


async def _cancel_remaining_tasks() -> None:
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


class _ThreadRunner:
    def __init__(self) -> None:
        self.runner = asyncio.Runner()

    def close(self) -> None:
        self.runner.close()

    def __del__(self) -> None:
        self.close()


def close_thread_runner() -> None:
    owner = getattr(_local, "owner", None)
    if owner is not None:
        owner.close()
        del _local.owner


def run_async(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    if not current_runtime_options().persistent_async_runner:
        return asyncio.run(factory())
    # Check before creating a coroutine or touching tasks belonging to another loop.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("synchronous async execution cannot run inside an active event loop")
    owner = getattr(_local, "owner", None)
    if owner is None:
        owner = _ThreadRunner()
        _local.owner = owner
    runner: asyncio.Runner = owner.runner
    try:
        return runner.run(factory(), context=copy_context())
    finally:
        runner.run(_cancel_remaining_tasks(), context=copy_context())
