from __future__ import annotations

from typing import Optional, Dict, Callable, Any

from pathlib import Path
import sys

# Ensure IkaMem is importable (same pattern as in agents.py)
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_dir / "IkaMem"))

from IkaMem import STMemory, LTMemory, LTMemItem  # type: ignore


class AgentMemoryMixin:

    short_term_memory: Optional[STMemory]
    long_term_memory: Optional[LTMemory]

    def init_short_term_memory(self, embedder_config: dict) -> None:
        self.short_term_memory = STMemory(embedder_config=embedder_config)
        self.short_term_memory.agent = self.name
    
    def _save_to_short_term(self, data: str, metadata: Optional[Dict] = None) -> str:
        if not self.short_term_memory:
            return "error: short-term memory not initialized"
        
        try:
            self.short_term_memory.save(data, metadata or {})
            return f"saved to short-term memory: {data[:50]}..."
        except Exception as e:
            return f"error saving to short-term memory: {str(e)}"
    
    def _save_to_long_term(self, task: str, output: str) -> str:
        if not self.long_term_memory:
            return "error: long-term memory not initialized"
        
        try:
            from datetime import datetime
            item = LTMemItem(
                agent=self.name,
                task=task,
                expected_output=output,
                datetime=datetime.now().isoformat(),
                quality=1.0,  # default quality
                metadata={}
            )
            self.long_term_memory.save(item)
            return f"saved to long-term memory - task: {task[:30]}..."
        except Exception as e:
            return f"error saving to long-term memory: {str(e)}"

    def _search_short_term(self, query: str, limit: int = 5, score_threshold: float = 0.6) -> dict:
        if not self.short_term_memory:
            return {"error": "short-term memory not initialized"}
        query = (query or "").strip()
        if not query:
            return {"error": "query is required"}
        limit = max(1, min(int(limit or 5), 50))
        score_threshold = max(0.0, min(float(score_threshold or 0.6), 1.0))
        try:
            results = self.short_term_memory.search(query=query, limit=limit, score_threshold=score_threshold)
            structured = []
            for result in results or []:
                if isinstance(result, dict):
                    structured.append({
                        "data": result.get("data") or result.get("content") or str(result),
                        "metadata": result.get("metadata", {}),
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

    def _search_long_term(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
        filter_func: Optional[Callable[..., Any]] = None,
    ) -> dict:
        if not self.long_term_memory:
            return {"error": "long-term memory not initialized"}
        query = (query or "").strip()
        if not query:
            return {"error": "query is required"}
        limit = max(1, min(int(limit or 5), 50))
        score_threshold = max(0.0, min(float(score_threshold or 0.6), 1.0))
        try:
            results = self.long_term_memory.search(query=query, limit=limit, score_threshold=score_threshold)
            if filter_func:
                try:
                    results = filter_func(results)
                except Exception:
                    pass
            structured = []
            for result in results or []:
                if hasattr(result, "task") and hasattr(result, "expected_output"):
                    structured.append({
                        "task": getattr(result, "task", "") or "",
                        "output": getattr(result, "expected_output", "") or "",
                        "metadata": getattr(result, "metadata", {}) if hasattr(result, "metadata") else {},
                    })
                elif isinstance(result, dict):
                    structured.append({
                        "task": result.get("task", result.get("content", "")),
                        "output": result.get("expected_output", result.get("output", "")),
                        "metadata": result.get("metadata", {}),
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

