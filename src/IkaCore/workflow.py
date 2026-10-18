"""Public workflow API."""

# pyright: strict
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .config import OutputConfig
from .events import RuntimeOutput, can_inherit_silent_output, output_scope
from .output_adapters import make_output
from .workflow_async import WorkflowExecution
from .workflow_async_executor import AsyncWorkflowExecutor
from .workflow_context_cache import reset_context_cache
from .workflow_snapshots import WorkflowSnapshotMixin
from .workflow_types import (
    WorkflowCompressionHook,
    WorkflowEdge,
    WorkflowError,
    WorkflowNode,
    WorkflowNodeError,
    WorkflowResult,
)


class IkaWorkflow(WorkflowSnapshotMixin, WorkflowExecution):
    def __init__(
        self,
        name: str,
        description: str,
        nodes: List[WorkflowNode],
        edges: List[WorkflowEdge],
        compress_hook: Optional[WorkflowCompressionHook] = None,
        start_node: Optional[str] = None,
        async_executor: Optional[AsyncWorkflowExecutor] = None,
        max_parallel_workers: int = 4,
        output: Optional[OutputConfig] = None,
    ):
        if not nodes:
            raise ValueError("Workflow requires at least one node.")
        self.events: RuntimeOutput = make_output(output or OutputConfig())
        self.name = name
        self.description = description
        self.nodes = nodes
        self.edges = edges
        self.compress_hook: WorkflowCompressionHook = compress_hook or self._default_compress_hook
        self.start_node: str = start_node or nodes[0].name
        self.async_executor: AsyncWorkflowExecutor = async_executor or AsyncWorkflowExecutor(max_workers=max_parallel_workers)
        self._owns_executor = async_executor is None

        self._node_index: Dict[str, WorkflowNode] = {node.name: node for node in nodes}
        self._results: Dict[str, WorkflowResult] = {}
        self._node_dependencies: Dict[str, Set[str]] = {}
        self._node_dependents: Dict[str, Set[str]] = {}
        self._edges_by_source: Dict[str, List[WorkflowEdge]] = {}
        self._next_edges_by_source: Dict[str, List[WorkflowEdge]] = {}

        self._validate_nodes()
        self._validate_edges()
        self._bind_stage_wiring()
        self._build_dependency_graph()
        self._next_reachable_nodes: Set[str] = self._compute_next_reachable_nodes()

    def run(self, initial_context: Optional[str] = None, parallel: bool = False, *,
            concurrency: int | None = None) -> Dict[str, WorkflowResult]:
        """Block until completion; parallel uses worker threads, bounded by concurrency."""
        if concurrency is not None:
            if concurrency < 1:
                raise ValueError("concurrency must be positive")
            if concurrency != self.async_executor.max_workers:
                if not self._owns_executor:
                    raise ValueError("concurrency must match the caller-owned executor")
                self.async_executor.shutdown()
                self.async_executor: AsyncWorkflowExecutor = AsyncWorkflowExecutor(max_workers=concurrency)
            parallel = True
        use_async = parallel
        if can_inherit_silent_output(self.events):
            return self._run(initial_context, use_async)
        with output_scope(self.events, share_cli=True):
            return self._run(initial_context, use_async)

    def _run(self, initial_context: Optional[str], use_async: bool) -> Dict[str, WorkflowResult]:
        self._validate_execution_graph()
        if self._workflow_snapshot is not None:
            return self._snapshot_run(lambda: run_workflow(self, initial_context, use_async), initial_context, use_async)
        return run_workflow(self, initial_context, use_async)



def run_workflow(self: IkaWorkflow, initial_context: Optional[str], use_async: bool) -> Dict[str, WorkflowResult]:
    reset_context_cache(self, initial_context)
    if use_async:
        return self._run_parallel(initial_context)
    from .workflow_sync_dataflow import run_sync_dataflow
    return run_sync_dataflow(self, initial_context)


__all__ = [
    "AsyncWorkflowExecutor",
    "IkaWorkflow",
    "WorkflowCompressionHook",
    "WorkflowEdge",
    "WorkflowNode",
    "WorkflowResult",
    "WorkflowError",
    "WorkflowNodeError",
]
