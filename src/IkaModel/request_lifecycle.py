"""Cancellation registration, cooperative waits, and deadline cleanup."""

# pyright: strict

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Generator
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from typing import Any, Optional, TypeVar

from .request_control import (
    check_request_controls,
    effective_timeout,
    register_request_abort_callback,
    remaining_request_time,
    request_cancelled,
    request_lifecycle_active,
)

T = TypeVar("T")


def _quiet_close(callback: Callable[[], None]) -> None:
    try:
        callback()
    except (OSError, RuntimeError):
        pass


@contextmanager
def request_scope(close: Callable[[], None]) -> Generator[None, None, None]:
    check_request_controls()
    active = True

    def abort() -> None:
        if active:
            _quiet_close(close)

    unregister = register_request_abort_callback(abort)
    remaining = remaining_request_time()
    timer = threading.Timer(remaining, abort) if remaining is not None else None
    if timer is not None:
        timer.daemon = True
        timer.start()
    try:
        yield
        check_request_controls()
    finally:
        active = False
        if timer is not None:
            timer.cancel()
        if unregister is not None:
            _quiet_close(unregister)


def sleep_with_controls(delay: float, sleeper: Callable[[float], None]) -> None:
    if not request_lifecycle_active():
        sleeper(delay)
        return
    end = time.monotonic() + delay
    while True:
        check_request_controls()
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        sleeper(min(0.05, remaining))


async def async_sleep_with_controls(delay: float, sleeper: Callable[[float], Awaitable[Any]]) -> None:
    if not request_lifecycle_active():
        await sleeper(delay)
        return
    end = time.monotonic() + delay
    while True:
        check_request_controls()
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        await sleeper(min(0.05, remaining))


def wait_for_tool(future: Future[T], timeout: Optional[float]) -> T:
    if not request_lifecycle_active():
        return future.result(timeout=timeout)
    end = None if timeout is None else time.monotonic() + timeout
    while True:
        check_request_controls()
        remaining = None if end is None else max(0, end - time.monotonic())
        interval = effective_timeout(0.05 if remaining is None else min(0.05, remaining))
        try:
            result = future.result(timeout=interval)
        except FutureTimeoutError:
            check_request_controls()
            if future.done() or (end is not None and time.monotonic() >= end):
                raise
            continue
        check_request_controls()
        return result


async def await_with_controls(factory: Callable[[], Awaitable[T]]) -> T:
    check_request_controls()
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    active = True
    requested_cancel = False

    def perform_abort() -> None:
        nonlocal requested_cancel
        if active and task is not None and not task.done():
            requested_cancel = True
            task.cancel()

    def abort() -> None:
        if active:
            loop.call_soon_threadsafe(perform_abort)

    unregister = register_request_abort_callback(abort)
    try:
        async with asyncio.timeout(remaining_request_time()):
            result = await factory()
        check_request_controls()
        return result
    except asyncio.CancelledError:
        if requested_cancel and task is not None and request_cancelled():
            task.uncancel()
        check_request_controls()
        raise
    except TimeoutError:
        check_request_controls()
        raise
    finally:
        active = False
        if unregister is not None:
            _quiet_close(unregister)
