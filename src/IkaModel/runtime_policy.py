"""Execution-scoped resource and context settings."""

# pyright: strict

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Optional


@dataclass(frozen=True)
class RuntimeOptions:
    shared_ssl_context: bool = False
    reuse_connections: bool = False
    parallel_context_preparation: bool = False
    recover_truncated_control_calls: bool = False
    compact_at_fraction: Optional[float] = None
    summary_context_threshold: Optional[int] = None


_DEFAULT = RuntimeOptions()
_options: ContextVar[RuntimeOptions] = ContextVar("ika_runtime_options", default=_DEFAULT)


def current_runtime_options() -> RuntimeOptions:
    return _options.get()


def runtime_options_enabled() -> bool:
    return _options.get() != _DEFAULT


def _validate_threshold(value: object) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError("summary_context_threshold must be a non-negative integer")


def _validate_options(options: RuntimeOptions) -> None:
    flags = ("shared_ssl_context", "reuse_connections", "parallel_context_preparation", "recover_truncated_control_calls")
    for name in flags:
        if not isinstance(getattr(options, name), bool):
            raise ValueError(f"{name} must be a boolean")
    fraction = options.compact_at_fraction
    if fraction is not None and (not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("compact_at_fraction must be greater than zero and at most one")
    _validate_threshold(options.summary_context_threshold)


@contextmanager
def runtime_options(options: Optional[RuntimeOptions] = None, **changes: Any) -> Generator[RuntimeOptions, None, None]:
    configured = replace(options or _options.get(), **changes)
    _validate_options(configured)
    token = _options.set(configured)
    try:
        if configured.reuse_connections:
            from .connection_scope import connection_scope
            with connection_scope():
                yield configured
        else:
            yield configured
    finally:
        _options.reset(token)
