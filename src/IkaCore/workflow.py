"""Public workflow API."""

# pyright: strict

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .workflow_async import WorkflowAsyncExecutionMixin
from .workflow_async_executor import AsyncWorkflowExecutor
from .workflow_types import WorkflowCompressionHook, WorkflowEdge, WorkflowNode, WorkflowResult


class IkaWorkflow(WorkflowAsyncExecutionMixin):
    def __init__(
        self,
        name: str,
        description: str,
        nodes: List[WorkflowNode],
        edges: List[WorkflowEdge],
        compress_hook: Optional[WorkflowCompressionHook] = None,
        start_node: Optional[str] = None,
        async_executor: Optional[AsyncWorkflowExecutor] = None,
        max_parallel_workers: int = 30,
    ):
        if not nodes:
            raise ValueError("Workflow requires at least one node.")
        self.name = name
        self.description = description
        self.nodes = nodes
        self.edges = edges
        self.compress_hook = compress_hook or self._default_compress_hook
        self.start_node = start_node or nodes[0].name
        self.async_executor = async_executor or AsyncWorkflowExecutor(max_workers=max_parallel_workers)

        self._node_index: Dict[str, WorkflowNode] = {node.name: node for node in nodes}
        self._results: Dict[str, WorkflowResult] = {}
        self._visiting: Set[str] = set()
        self._node_dependencies: Dict[str, Set[str]] = {}
        self._node_dependents: Dict[str, Set[str]] = {}
        self._edges_by_source: Dict[str, List[WorkflowEdge]] = {}
        self._next_edges_by_source: Dict[str, List[WorkflowEdge]] = {}

        self._validate_nodes()
        self._validate_edges()
        self._bind_stage_wiring()
        self._build_dependency_graph()
        self._next_reachable_nodes: Set[str] = self._compute_next_reachable_nodes()

    def run(self, initial_context: Optional[str] = None, use_async: bool = False) -> Dict[str, WorkflowResult]:
        if use_async:
            return self.run_async(initial_context)

        upstream_contexts: Dict[str, List[str]] = {}
        if initial_context:
            upstream_contexts[self.start_node] = [initial_context]
        self._results = {}
        self._visiting = set()
        self._run_node(self.start_node, upstream_contexts)
        return self._results


__all__ = [
    "AsyncWorkflowExecutor",
    "IkaWorkflow",
    "WorkflowCompressionHook",
    "WorkflowEdge",
    "WorkflowNode",
    "WorkflowResult",
]
