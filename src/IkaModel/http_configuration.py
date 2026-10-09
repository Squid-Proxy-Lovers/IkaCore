"""Opt-in SSL-context reuse that retains httpx environment verification."""

# pyright: strict

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import ssl

_ENV_INPUTS = ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE")
_lock = threading.Lock()
_cached_key: object = None
_cached_context: Optional["ssl.SSLContext"] = None


def _env_key() -> tuple[tuple[str, Optional[str], Optional[tuple[int, int]]], ...]:
    values: list[tuple[str, Optional[str], Optional[tuple[int, int]]]] = []
    for name in _ENV_INPUTS:
        value = os.environ.get(name)
        stamp = None
        if value and name != "SSLKEYLOGFILE":
            try:
                stat = os.stat(value)
                stamp = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                pass
        values.append((name, value, stamp))
    return tuple(values)


def shared_ssl_context() -> "ssl.SSLContext":
    from httpx import create_ssl_context

    global _cached_key, _cached_context
    key = _env_key()
    with _lock:
        if _cached_context is None or _cached_key != key:
            _cached_context = create_ssl_context(verify=True, trust_env=True)
            _cached_key = key
        return _cached_context


def clear_ssl_context_cache() -> None:
    global _cached_key, _cached_context
    with _lock:
        _cached_key = None
        _cached_context = None
