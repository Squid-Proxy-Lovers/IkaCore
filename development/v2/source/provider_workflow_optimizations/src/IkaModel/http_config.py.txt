"""Process-wide httpx configuration shared by every IkaCore HTTP call site.

Building an ``ssl.SSLContext`` loads the whole CA bundle from disk, which costs
tens to hundreds of milliseconds. httpx builds one for every ``Client``,
``AsyncClient`` and module-level ``httpx.post``/``httpx.stream`` call, and
IkaCore creates those per agent, per async chat call and per Codex request.
Passing one shared context as ``verify=`` removes that cost while leaving
client lifetimes untouched (agents still own and close their clients, and
async clients stay bound to the event loop that created them).

The cache key is every environment input httpx and the stdlib consult when
building a default context, so the resulting context is exactly what httpx
would have built for the same environment.
"""

# pyright: strict

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import ssl

_ENV_INPUTS = ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE")

_lock = threading.Lock()
_cached_key: Optional[tuple[Optional[str], ...]] = None
_cached_context: Optional["ssl.SSLContext"] = None


def _env_key() -> tuple[Optional[str], ...]:
    return tuple(os.environ.get(name) for name in _ENV_INPUTS)


def _build_context() -> "ssl.SSLContext":
    # Deferred import keeps ``import IkaCore`` from paying for ssl/certifi.
    from httpx import create_ssl_context

    return create_ssl_context(verify=True, trust_env=True)


def shared_ssl_context() -> "ssl.SSLContext":
    """Return the default-verification SSL context for the current environment."""
    global _cached_key, _cached_context
    key = _env_key()
    context = _cached_context
    if context is not None and _cached_key == key:
        return context
    with _lock:
        if _cached_context is None or _cached_key != key:
            _cached_context = _build_context()
            _cached_key = key
        return _cached_context


def clear_ssl_context_cache() -> None:
    """Drop the cached context (e.g. after CA files change on disk)."""
    global _cached_key, _cached_context
    with _lock:
        _cached_key = None
        _cached_context = None


__all__ = ["clear_ssl_context_cache", "shared_ssl_context"]
