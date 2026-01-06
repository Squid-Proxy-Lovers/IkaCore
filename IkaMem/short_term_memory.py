import time
from typing import Any, Optional

from IkaMem.memory import Memory
from IkaMem.memory_items import STMemItem
from IkaMem.storage.mem0_storage import Mem0Store


class STMemory(Memory):
    """short-term memory for managing data related to immediate tasks and interactions."""
    def __init__(
        self,
        embedder_config: Optional[dict[str, Any]] = None,
        storage: Optional[Any] = None,
    ) -> None:
        """
        init ShortTermMemory.
        
        Args:
            embedder_config: config for memory provider
                - provider: 'mem0' to use Mem0Store
                - config: Configuration dict for Mem0 (api_key, user_id, etc.)
            storage: optional custom storage backend. If not provided and
                embedder_config specifies 'mem0', Mem0Store will be used.
        """
        memory_provider = None
        if embedder_config and isinstance(embedder_config, dict):
            memory_provider = embedder_config.get("provider")

        if memory_provider == "mem0" and not storage:
            config = embedder_config.get("config") if embedder_config else None
            storage = Mem0Store(memory_type="short_term", config=config)
        elif not storage:
            raise ValueError("no storage backend provided")

        super().__init__(storage=storage)
        self._memory_provider = memory_provider


    def save(
        self,
        value: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        save data to short-term memory.
        
        Args:
            value: the value/content to save
            metadata: optional metadata to associate with the value
        """
        start_time = time.time()
        
        try:
            # create short-term memory item
            item = STMemItem(data=value, metadata=metadata, agent=self.agent)
            
            if self._memory_provider == "mem0":
                item.data = f"Remember the following insights from Agent run: {item.data}"

            super().save(value=item.data, metadata=item.metadata)
            
            elapsed = (time.time() - start_time) * 1000
            print(f"[STMemory] saved in {elapsed:.2f}ms")
            
        except Exception as e:
            print(f"[STMemory] save failed: {str(e)}")
            raise


    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        """
        search short-term memory for relevant entries.

        Args:
            query: The search query
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score for results

        Returns:
            list of matching memory entries
        """
        start_time = time.time()
        
        try:
            results = self.storage.search(
                query=query, limit=limit, score_threshold=score_threshold
            )
            
            elapsed = (time.time() - start_time) * 1000
            print(f"[STMemory] search completed in {elapsed:.2f}ms, found {len(results)} results")
            
            return list(results)
            
        except Exception as e:
            print(f"[STMemory] search failed: {str(e)}")
            raise

