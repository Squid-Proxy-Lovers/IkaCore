"""Run coroutines from sync code on a persistent per-thread event loop.

``asyncio.run`` creates and destroys an event loop on every call, and async
HTTP connections cannot outlive their loop, so an async agent calling it once
per chat round reconnected to its provider every round. ``run_coroutine``
keeps one loop per thread (via ``asyncio.Runner``) so pooled connections
survive between rounds, while preserving what callers can observe from
``asyncio.run``: each call runs in a fresh copy of the caller's context, tasks
left pending when the coroutine finishes are cancelled, and calling it from a
thread whose loop is already running raises ``RuntimeError``.
"""

# pyright: strict

from __future__ import annotations

import asyncio
import contextvars
import threading
import weakref
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

_local = threading.local()


class _RunnerHolder:
    """Lives in the thread-local slot; when the thread ends (or at exit) its runner is closed."""

    __slots__ = ("runner", "__weakref__")

    def __init__(self, runner: asyncio.Runner) -> None:
        self.runner = runner


def _close_runner(runner: asyncio.Runner) -> None:
    try:
        runner.close()
    except RuntimeError:  # still running (interpreter teardown mid-call); nothing safe to do
        pass


def _runner() -> asyncio.Runner:
    holder: _RunnerHolder | None = getattr(_local, "holder", None)
    if holder is None:
        holder = _RunnerHolder(asyncio.Runner())
        weakref.finalize(holder, _close_runner, holder.runner)
        _local.holder = holder
    return holder.runner


def _cancel_leftovers(runner: asyncio.Runner) -> None:
    loop = runner.get_loop()
    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
    if not pending:
        return
    for task in pending:
        task.cancel()

    async def _drain() -> None:
        await asyncio.gather(*pending, return_exceptions=True)

    runner.run(_drain())


def run_coroutine(coro: Coroutine[Any, Any, T]) -> T:
    runner = _runner()
    try:
        return runner.run(coro, context=contextvars.copy_context())
    finally:
        _cancel_leftovers(runner)


__all__ = ["run_coroutine"]
