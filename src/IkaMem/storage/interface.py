# pyright: strict

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Storage(ABC):
    @abstractmethod
    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        """Save a value to storage with associated metadata.
        
        Args:
            value: The value to store
            metadata: Metadata associated with the value
        """

    @abstractmethod
    def search(
        self, query: str, limit: int, score_threshold: float
    ) -> list[Any]:
        """
        Search storage for relevant entries.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            list of matching entries
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored values."""
