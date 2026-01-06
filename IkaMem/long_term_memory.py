import time
from typing import Any, Optional

from IkaMem.memory import Memory
from IkaMem.memory_items import LTMemItem
from IkaMem.storage.mem0_storage import Mem0Store


class LTMemory(Memory):
    """long-term memory for cross-run execution and performance data."""

    def __init__(
        self,
        embedder_config: Optional[dict[str, Any]] = None,
        storage: Optional[Any] = None,
    ) -> None:
        """
        init LongTermMemory.
        
        Args:
            embedder_config: config for memory provider
                - provider: 'mem0' to use Mem0Store
                - config: Configuration dict for Mem0 (api_key, user_id, etc.)
            storage: Optional custom storage backend. If not provided and
                embedder_config specifies 'mem0', Mem0Storage will be used.
        """
        memory_provider = None
        if embedder_config and isinstance(embedder_config, dict):
            memory_provider = embedder_config.get("provider")

        if memory_provider == "mem0" and not storage:
            config = embedder_config.get("config") if embedder_config else None
            storage = Mem0Store(memory_type="long_term", config=config)
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
        save data to long-term memory.
        
        Args:
            value: The value/content to save (usually a LTMemItem)
            metadata: optional metadata to associate with the value
        """
        start_time = time.time()
        
        try:
            if isinstance(value, LTMemItem):
                item_metadata = value.metadata.copy()
                item_metadata.update({
                    "agent": value.agent,
                    "expected_output": value.expected_output,
                    "quality": value.quality,
                    "datetime": value.datetime,
                })
                if metadata:
                    item_metadata.update(metadata)
                
                # save task description with enriched metadata
                super().save(value=value.task, metadata=item_metadata)
            else:
                # save as-is with provided metadata
                super().save(value=value, metadata=metadata or {})
            
            elapsed = (time.time() - start_time) * 1000
            print(f"[LTMemory] saved in {elapsed:.2f}ms")
            
        except Exception as e:
            print(f"[LTMemory] save failed: {str(e)}")
            raise


    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        """
        search long-term memory for relevant entries.
        
        note:  default limit is 3 to keep context manageable as long-term
        memory grows over time.

        Args:
            query: the search query
            limit: maximum number of results to return
            score_threshold: minimum similarity score for results

        Returns:
            list of matching memory entries
        """
        start_time = time.time()
        
        try:
            results = self.storage.search(
                query=query, limit=limit, score_threshold=score_threshold
            )
            
            elapsed = (time.time() - start_time) * 1000
            print(f"[LTMemory] search completed in {elapsed:.2f}ms, found {len(results)} results")
            
            return results or []
            
        except Exception as e:
            print(f"[LTMemory] search failed: {str(e)}")
            raise

