"""Propagate opt-in runtime controls into synchronous workers."""

# pyright: strict

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from typing import ParamSpec, TypeVar

from .execution_hooks import hooks_enabled
from .request_control import controls_enabled
from .runtime_policy import runtime_options_enabled

P = ParamSpec("P")
T = TypeVar("T")


def submit_with_controls(pool: ThreadPoolExecutor, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Future[T]:
    if not (controls_enabled() or runtime_options_enabled() or hooks_enabled()):
        return pool.submit(function, *args, **kwargs)
    return pool.submit(copy_context().run, partial(function, *args, **kwargs))


def run_in_executor_with_controls(
    loop: asyncio.AbstractEventLoop, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> asyncio.Future[T]:
    call = partial(function, *args, **kwargs)
    if not (controls_enabled() or runtime_options_enabled() or hooks_enabled()):
        return loop.run_in_executor(None, call)
    return loop.run_in_executor(None, copy_context().run, call)
