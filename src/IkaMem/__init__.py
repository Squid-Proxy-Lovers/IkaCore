"""memory system using mem0 memory layer."""

# pyright: strict

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from IkaMem.long_term_memory import LTMemory
from IkaMem.memory import Memory
from IkaMem.memory_items import LTMemItem, STMemItem
from IkaMem.short_term_memory import STMemory
from IkaMem.storage.interface import Storage
from IkaMem.storage.mem0_storage import Mem0Store

try:
    __version__ = version("ikacore")
except PackageNotFoundError:
    __version__ = "2.0.0"

__all__ = [
    "__version__",
    "Memory",
    "STMemory",
    "LTMemory",
    "STMemItem",
    "LTMemItem",
    "Storage",
    "Mem0Store",
]
