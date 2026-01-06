from typing import Any


class Storage:
    """abstract storage interface."""
    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        """Save a value to storage with associated metadata.
        
        Args:
            value: The value to store
            metadata: Metadata associated with the value
        """
        pass

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
        return []

    def reset(self) -> None:
        """Reset/clear the storage."""
        pass

