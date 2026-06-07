# pyright: strict

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Optional, cast

from IkaMem.memory import Memory, StorageBackend
from IkaMem.memory_items import LTMemItem
from IkaMem.storage.mem0_storage import Mem0Store

LOG = logging.getLogger(__name__)


class LongTermMemorySaveMixin:
    storage: StorageBackend

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
            self.storage.save(value=value.task, metadata=item_metadata)
        else:
            # save as-is with provided metadata
            self.storage.save(value=value, metadata=metadata or {})

        elapsed = (time.time() - start_time) * 1000
        LOG.debug("long-term memory saved in %.2fms", elapsed)


class LongTermMemorySearchMixin:
    storage: StorageBackend
    _search_limit: int
    _filter_func: Callable[[list[Any]], list[Any]]

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
        
        search_limit = limit if limit is not None else self._search_limit
        raw_results = self.storage.search(
            query=query, limit=search_limit, score_threshold=score_threshold
        )

        elapsed = (time.time() - start_time) * 1000
        LOG.debug("long-term memory search completed in %.2fms, found %d results", elapsed, len(raw_results))

        # apply custom filter (must return list)
        filtered = self._filter_func(raw_results)

        # enforce list return type
        if not isinstance(cast(object, filtered), list):
            raise TypeError(f"filter function must return list, got {type(filtered).__name__}")

        return filtered


class LTMemory(LongTermMemorySaveMixin, LongTermMemorySearchMixin, Memory):
    """long-term memory for cross-run execution and performance data."""

    def __init__(
        self,
        embedder_config: Optional[dict[str, Any]] = None,
        storage: Optional[StorageBackend] = None,
    ) -> None:
        provider = embedder_config.get("provider") if embedder_config else None
        memory_provider = provider if isinstance(provider, str) else None

        if memory_provider == "mem0" and not storage:
            config = cast(Optional[dict[str, Any]], embedder_config.get("config") if embedder_config else None)
            storage = Mem0Store(memory_type="long_term", config=config)
        elif not storage:
            raise ValueError("no storage backend provided")

        super().__init__(storage=storage)
        self._memory_provider: Optional[str] = memory_provider
        self._search_limit = 10
        self._filter_func: Callable[[list[Any]], list[Any]] = self._default_filter

    @staticmethod
    def _default_filter(results: list[Any]) -> list[Any]:
        return results[:5]
