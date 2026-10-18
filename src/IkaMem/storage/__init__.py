"""storage backends."""

# pyright: strict

from __future__ import annotations

from IkaMem.storage.interface import Storage

try:
    from IkaMem.storage.mem0_storage import Mem0Store as _Mem0Store
    Mem0Store: type[_Mem0Store] | None = _Mem0Store
except ImportError:
    Mem0Store = None

__all__ = [
    "Storage",
    "Mem0Store",
]
