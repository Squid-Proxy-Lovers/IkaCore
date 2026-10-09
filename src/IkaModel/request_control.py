"""Opt-in execution controls shared by async tasks and copied worker contexts."""

# pyright: strict

from __future__ import annotations

import math
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Optional

from .runtime_errors import IkaRequestCancelled, IkaRequestDeadlineExceeded

AbortRegistrar = Callable[[Callable[[], None]], Optional[Callable[[], None]]]
CancelChecker = Callable[[], bool]


@dataclass(frozen=True)
class _Controls:
    cancel_checker: Optional[CancelChecker] = None
    abort_registrar: Optional[AbortRegistrar] = None
    deadline: Optional[float] = None
    max_retries: Optional[int] = None
    tool_result_max_chars: Optional[int] = None
    codex_tool_output_max_chars: Optional[int] = None


_controls: ContextVar[_Controls] = ContextVar("ika_request_controls", default=_Controls())


def set_request_cancel_checker(checker: Optional[CancelChecker]) -> Optional[CancelChecker]:
    previous = _controls.get()
    _controls.set(replace(previous, cancel_checker=checker))
    return previous.cancel_checker


def set_request_abort_registrar(registrar: Optional[AbortRegistrar]) -> Optional[AbortRegistrar]:
    previous = _controls.get()
    _controls.set(replace(previous, abort_registrar=registrar))
    return previous.abort_registrar


def set_request_max_retries(max_retries: Optional[int]) -> Optional[int]:
    previous = _controls.get()
    _controls.set(replace(previous, max_retries=None if max_retries is None else max(1, int(max_retries))))
    return previous.max_retries


def effective_max_retries(default: int) -> int:
    override = _controls.get().max_retries
    return default if override is None else override


def request_cancelled() -> bool:
    checker = _controls.get().cancel_checker
    return bool(checker and checker())


def request_lifecycle_active() -> bool:
    controls = _controls.get()
    return controls.cancel_checker is not None or controls.abort_registrar is not None or controls.deadline is not None


def controls_enabled() -> bool:
    return _controls.get() != _Controls()


def register_request_abort_callback(callback: Callable[[], None]) -> Optional[Callable[[], None]]:
    registrar = _controls.get().abort_registrar
    return registrar(callback) if registrar is not None else None


def remaining_request_time() -> Optional[float]:
    deadline = _controls.get().deadline
    return None if deadline is None else deadline - time.monotonic()


def check_request_controls() -> None:
    if request_cancelled():
        raise IkaRequestCancelled("execution cancelled")
    remaining = remaining_request_time()
    if remaining is not None and remaining <= 0:
        raise IkaRequestDeadlineExceeded("execution deadline expired")


def effective_timeout(default: Optional[float]) -> Optional[float]:
    check_request_controls()
    remaining = remaining_request_time()
    if remaining is None:
        return default
    return remaining if default is None else min(default, remaining)


def tool_result_limit() -> Optional[int]:
    return _controls.get().tool_result_max_chars


def codex_tool_output_limit() -> Optional[int]:
    return _controls.get().codex_tool_output_max_chars


def _positive_limit(value: object, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@contextmanager
def request_controls(
    *,
    cancel_checker: Optional[CancelChecker] = None,
    abort_registrar: Optional[AbortRegistrar] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    tool_result_max_chars: Optional[int] = None,
    codex_tool_output_max_chars: Optional[int] = None,
) -> Generator[None, None, None]:
    """Scope controls to an execution; nested scopes inherit unspecified settings."""
    previous = _controls.get()
    deadline = previous.deadline
    if timeout is not None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        proposed = time.monotonic() + timeout
        deadline = proposed if deadline is None else min(deadline, proposed)
    _positive_limit(tool_result_max_chars, "tool_result_max_chars")
    _positive_limit(codex_tool_output_max_chars, "codex_tool_output_max_chars")
    token = _controls.set(_Controls(
        cancel_checker=cancel_checker if cancel_checker is not None else previous.cancel_checker,
        abort_registrar=abort_registrar if abort_registrar is not None else previous.abort_registrar,
        deadline=deadline,
        max_retries=previous.max_retries if max_retries is None else max(1, int(max_retries)),
        tool_result_max_chars=previous.tool_result_max_chars if tool_result_max_chars is None else tool_result_max_chars,
        codex_tool_output_max_chars=(previous.codex_tool_output_max_chars
                                     if codex_tool_output_max_chars is None else codex_tool_output_max_chars),
    ))
    try:
        yield
    finally:
        _controls.reset(token)
