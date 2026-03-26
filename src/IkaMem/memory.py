from typing import Any, Optional


class Memory:
    def __init__(self, storage: Any):
        """
        init memory.
        
        Args:
            storage: storage backend 
        """
        self.storage = storage
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
