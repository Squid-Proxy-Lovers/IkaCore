"""Context-local request substitution, shared by model and summary requests."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import httpx

RequestAdapter = Callable[[str, dict[str, Any]], httpx.Response]
_adapter: ContextVar[RequestAdapter | None] = ContextVar("ika_request_adapter", default=None)


def current_request_adapter() -> RequestAdapter | None:
    return _adapter.get()


@contextmanager
def request_adapter(adapter: RequestAdapter) -> Iterator[None]:
    token = _adapter.set(adapter)
    try:
        yield
    finally:
        _adapter.reset(token)
