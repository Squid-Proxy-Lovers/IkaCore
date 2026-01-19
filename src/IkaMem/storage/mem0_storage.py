import os
from collections import defaultdict
from typing import Any, Optional

try:
    from mem0 import Memory, MemoryClient
    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False
    Memory = None
    MemoryClient = None

from IkaMem.storage.interface import Storage


class Mem0Store(Storage):

    def __init__(self, memory_type: str, config: Optional[dict[str, Any]] = None):
        super().__init__()
        """
        init Mem0Store
        Args:
            memory_type: Type of memory ('short_term' or 'long_term')
            config: Optional configuration dictionary containing:
                - api_key: Mem0 API key (or set MEM0_API_KEY env var)
                - org_id: Organization ID
                - project_id: Project ID
                - user_id: User identifier
                - agent_id: Agent identifier
                - run_id: Run identifier (for short_term memory)
                - local_mem0_config: Local Mem0 configuration dict
                - infer: Whether to infer categories (default True)
                - includes: Categories to include
                - excludes: Categories to exclude
                - custom_categories: Custom categories list
        """
        super().__init__()
        
        self._validate_type(memory_type)
        self.memory_type = memory_type
        self.config = config or {}
        
        self.mem0_run_id = self.config.get("run_id")
        self.includes = self.config.get("includes")
        self.excludes = self.config.get("excludes")
        self.custom_categories = self.config.get("custom_categories")
        self.infer = self.config.get("infer", True)
        
        self._initialize_memory()

    def _initialize_memory(self) -> None:
        api_key = self.config.get("api_key") or os.getenv("MEM0_API_KEY")
        org_id = self.config.get("org_id")
        project_id = self.config.get("project_id")
        local_config = self.config.get("local_mem0_config")

        if api_key:
            # set MemoryClient for cloud-based Mem0
            if org_id and project_id:
                self.memory = MemoryClient(
                    api_key=api_key, org_id=org_id, project_id=project_id
                )
            else:
                self.memory = MemoryClient(api_key=api_key)
            
            # supdate project with custom categories if provided
            if self.custom_categories:
                self.memory.update_project(custom_categories=self.custom_categories)
        else:
            if local_config and len(local_config) > 0:
                self.memory = Memory.from_config(local_config)
            else:
                self.memory = Memory()

    def _create_filter_for_search(self) -> dict[str, list]:
        """
        create filter dictionary for searching memory.
        
        note: filter dictionary with AND/OR conditions
        """
        filter_dict = defaultdict(list)

        if self.memory_type == "short_term" and self.mem0_run_id:
            # For short-term memory with run_id, filter by run
            filter_dict["AND"].append({"run_id": self.mem0_run_id})
        else:
            # For long-term or short-term without run_id
            user_id = self.config.get("user_id", "")
            agent_id = self.config.get("agent_id", "")

            if user_id and agent_id:
                # Both user and agent - use OR condition
                filter_dict["OR"].append({"user_id": user_id})
                filter_dict["OR"].append({"agent_id": agent_id})
            elif user_id:
                # Only user
                filter_dict["AND"].append({"user_id": user_id})
            elif agent_id:
                # Only agent
                filter_dict["AND"].append({"agent_id": agent_id})

        return dict(filter_dict)

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        """
        save a value to Mem0 storage.
        
        Args:
            value: the value/content to save
            metadata: associated metadata
        """
        # Prepare conversation format for Mem0
        conversations = [{"role": "assistant", "content": value}]

        user_id = self.config.get("user_id", "")

        # Map memory types to metadata tags
        base_metadata = {
            "short_term": "short_term",
            "long_term": "long_term",
        }

        # Build parameters for memory.add()
        params: dict[str, Any] = {
            "metadata": {"type": base_metadata[self.memory_type], **metadata},
            "infer": self.infer,
        }

        # Add MemoryClient-specific parameters
        if isinstance(self.memory, MemoryClient):
            params["includes"] = self.includes
            params["excludes"] = self.excludes
            params["output_format"] = "v1.1"
            params["version"] = "v2"

        # Add run_id for short-term memory
        if self.memory_type == "short_term" and self.mem0_run_id:
            params["run_id"] = self.mem0_run_id

        # Add user_id if provided
        if user_id:
            params["user_id"] = user_id

        # Add agent_id if provided
        if agent_id := self.config.get("agent_id", ""):
            params["agent_id"] = agent_id

        # Save to Mem0
        self.memory.add(conversations, **params)

    def search(
        self, query: str, limit: int = 5, score_threshold: float = 0.6
    ) -> list[Any]:
        """
        search mem0 storage for relevant entries.
        
        Args:
            query: search query string
            limit: maximum number of results
            score_threshold: minimum similarity score
            
        Returns:
            list of matching memory entries
        """
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "version": "v2",
            "output_format": "v1.1",
        }

        if user_id := self.config.get("user_id", ""):
            params["user_id"] = user_id

        # Add metadata filter based on memory type
        memory_type_map = {
            "short_term": {"type": "short_term"},
            "long_term": {"type": "long_term"},
        }

        if self.memory_type in memory_type_map:
            params["metadata"] = memory_type_map[self.memory_type]
            if self.memory_type == "short_term":
                params["run_id"] = self.mem0_run_id

        # Add filters and threshold
        params["filters"] = self._create_filter_for_search()
        params["threshold"] = score_threshold

        # Remove parameters not supported by local Memory (use pop to avoid KeyError)
        if isinstance(self.memory, Memory):
            params.pop("metadata", None)
            params.pop("version", None)
            params.pop("output_format", None)
            params.pop("run_id", None)

        # Execute search
        results = self.memory.search(**params)

        # Normalize results format
        for result in results["results"]:
            result["content"] = result["memory"]

        return [r for r in results["results"]]

    def reset(self) -> None:
        if self.memory:
            self.memory.reset()

