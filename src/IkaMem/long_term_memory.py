import logging
import time
from typing import Any, Optional

from IkaMem.memory import Memory
from IkaMem.memory_items import LTMemItem
from IkaMem.storage.mem0_storage import Mem0Store

LOG = logging.getLogger(__name__)


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
        self._search_limit = 10  # default, can be overridden
        self._filter_func = lambda results: results[:5]  # default: top 5 by relevance - since mem0 already orders results by relevance


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
                    "expected_output": value.expected_output, # puts the expected output in the metadata
                    "quality": value.quality, # i'm not sure how we can dynamically define quality if we don't want to force the user to handle this everytime.
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
            LOG.debug("long-term memory saved in %.2fms", elapsed)
            
        except Exception:
            LOG.exception("long-term memory save failed")
            raise


    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        """
        search long-term memory and apply custom filter.
        
        Args:
            query: the search query
            limit: maximum number of results (uses _search_limit if None)
            score_threshold: minimum similarity score for results

        Returns:
            list of filtered memory entries 
        """
        start_time = time.time()
        
        try:
            search_limit = limit if limit is not None else self._search_limit
            raw_results = self.storage.search(
                query=query, limit=search_limit, score_threshold=score_threshold
            )
            
            elapsed = (time.time() - start_time) * 1000
            LOG.debug("long-term memory search completed in %.2fms, found %d results", elapsed, len(raw_results))
            
            # apply custom filter (must return list)
            filtered = self._filter_func(raw_results)
            
            # enforce list return type
            if not isinstance(filtered, list):
                raise TypeError(f"filter function must return list, got {type(filtered).__name__}")
            
            return filtered
            
        except Exception:
            LOG.exception("long-term memory search failed")
            raise
