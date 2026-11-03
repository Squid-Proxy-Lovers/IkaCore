# pyright: strict

from __future__ import annotations

from typing import Any, Optional, Protocol


class StorageBackend(Protocol):
    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        ...

    def search(self, query: str, limit: int, score_threshold: float) -> list[Any]:
        ...

    def reset(self) -> None:
        ...


class Memory:
    def __init__(self, storage: StorageBackend) -> None:
        """
        init memory.
        
        Args:
            storage: storage backend 
        """
        self.storage: StorageBackend = storage
        self._agent: Optional[str] = None
        self._task: Optional[str] = None

    @property
    def task(self) -> Optional[str]:
        return self._task

    @task.setter
    def task(self, task: Optional[str]) -> None:
        self._task = task

    @property
    def agent(self) -> Optional[str]:
        return self._agent

    @agent.setter
    def agent(self, agent: Optional[str]) -> None:
        self._agent = agent


    def save(self, value: Any, metadata: Optional[dict[str, Any]] = None) -> None:
        """
        save value to memory.
        
        Args:
            value: the value to save
            metadata: optional metadata to associate with the value
        """
        metadata = metadata or {}
        self.storage.save(value, metadata)


    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        """
        search memory for relevant entries.
        
        Args:
            query: the search query
            limit: maximum number of results to return
            score_threshold: minimum similarity score for results

        Returns:
            list of matching memory entries
        """
        return self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold
        )

    def reset(self) -> None:
        self.storage.reset()
