# pyright: strict

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, Protocol, cast

from IkaCore.agent_runtime_payloads import JsonDict
from IkaMem import LTMemItem, LTMemory, STMemory


class _AgentMemoryState(Protocol):
    name: str
    short_term_memory: Optional[STMemory]
    long_term_memory: Optional[LTMemory]


class ShortTermAgentMemoryMixin:
    short_term_memory: Optional[STMemory]

    def init_short_term_memory(self: _AgentMemoryState, embedder_config: JsonDict) -> None:
        self.short_term_memory = STMemory(embedder_config=embedder_config)
        self.short_term_memory.agent = self.name

    def _save_to_short_term(self: _AgentMemoryState, data: str, metadata: Optional[JsonDict] = None) -> str:
        if not self.short_term_memory:
            return "error: short-term memory not initialized"

        try:
            self.short_term_memory.save(data, metadata or {})
            return f"saved to short-term memory: {data[:50]}..."
        except Exception as e:
            return f"error saving to short-term memory: {str(e)}"

    def _search_short_term(
        self: _AgentMemoryState,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> JsonDict:
        if not self.short_term_memory:
            return {"error": "short-term memory not initialized"}
        query = (query or "").strip()
        if not query:
            return {"error": "query is required"}
        limit = max(1, min(int(limit or 5), 50))
        score_threshold = max(0.0, min(float(score_threshold or 0.6), 1.0))
        try:
            results = self.short_term_memory.search(query=query, limit=limit, score_threshold=score_threshold)
            structured: list[JsonDict] = []
            for result in results or []:
                if isinstance(result, dict):
                    mapped = cast(JsonDict, result)
                    data = mapped.get("data") or mapped.get("content") or str(mapped)
                    metadata = mapped.get("metadata", {})
                    structured.append({
                        "data": data if isinstance(data, str) else str(data),
                        "metadata": metadata if isinstance(metadata, dict) else {},
                    })
                else:
                    structured.append({"data": str(result), "metadata": {}})
            return {
                "query": query,
                "limit": limit,
                "score_threshold": score_threshold,
                "count": len(structured),
                "results": structured,
            }
        except Exception as e:
            return {"error": f"error searching short-term memory: {str(e)}"}


class LongTermAgentMemorySaveMixin:
    long_term_memory: Optional[LTMemory]

    def _save_to_long_term(self: _AgentMemoryState, payload: JsonDict) -> str:
        if not self.long_term_memory:
            return "error: long-term memory not initialized"

        try:
            from datetime import datetime
            task = str(payload.get("task") or "").strip()
            output = str(payload.get("output") or "").strip()
            raw_metadata: object = payload.get("metadata") or {}
            metadata = cast(JsonDict, raw_metadata) if isinstance(raw_metadata, dict) else {"metadata": raw_metadata}
            if not task or not output:
                return "error: long_term_save requires 'task' and 'output'"
            item = LTMemItem(
                agent=self.name,
                task=task,
                expected_output=output,
                datetime=datetime.now().isoformat(),
                quality=1.0,  # default quality
                metadata=metadata,
            )
            self.long_term_memory.save(item)
            return f"saved to long-term memory - task: {task[:30]}..."
        except Exception as e:
            return f"error saving to long-term memory: {str(e)}"


class LongTermAgentMemorySearchMixin(LongTermAgentMemorySaveMixin):
    def _search_long_term(
        self: _AgentMemoryState,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
        filter_func: Optional[Callable[..., Any]] = None,
    ) -> JsonDict:
        if not self.long_term_memory:
            return {"error": "long-term memory not initialized"}
        query = (query or "").strip()
        if not query:
            return {"error": "query is required"}
        limit = max(1, min(int(limit or 5), 50))
        score_threshold = max(0.0, min(float(score_threshold or 0.6), 1.0))
        try:
            results = cast(
                list[object],
                self.long_term_memory.search(query=query, limit=limit, score_threshold=score_threshold),
            )
            if filter_func:
                try:
                    filtered: object = filter_func(results)
                except Exception as e:
                    return {"error": f"long-term memory filter failed: {str(e)}"}
                if not isinstance(filtered, list):
                    return {"error": f"long-term memory filter failed: expected list, got {type(filtered).__name__}"}
                results = cast(list[object], filtered)
            structured: list[JsonDict] = []
            for result in results:
                if hasattr(result, "task") and hasattr(result, "expected_output"):
                    raw_metadata = getattr(result, "metadata", {}) if hasattr(result, "metadata") else {}
                    structured.append({
                        "task": getattr(result, "task", "") or "",
                        "output": getattr(result, "expected_output", "") or "",
                        "metadata": raw_metadata if isinstance(raw_metadata, dict) else {},
                    })
                elif isinstance(result, dict):
                    mapped = cast(JsonDict, result)
                    task = mapped.get("task", mapped.get("content", ""))
                    output = mapped.get("expected_output", mapped.get("output", ""))
                    metadata = mapped.get("metadata", {})
                    structured.append({
                        "task": task if isinstance(task, str) else str(task),
                        "output": output if isinstance(output, str) else str(output),
                        "metadata": metadata if isinstance(metadata, dict) else {},
                    })
                else:
                    structured.append({"task": str(result), "output": "", "metadata": {}})
            return {
                "query": query,
                "limit": limit,
                "score_threshold": score_threshold,
                "count": len(structured),
                "results": structured,
            }
        except Exception as e:
            return {"error": f"error searching long-term memory: {str(e)}"}


class AgentMemoryMixin(ShortTermAgentMemoryMixin, LongTermAgentMemorySearchMixin):
    pass
