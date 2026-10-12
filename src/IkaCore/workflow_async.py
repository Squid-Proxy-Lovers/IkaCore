"""Asynchronous workflow scheduling and result-processing mixins."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from IkaCore.cli_output import get_cli_output

from .workflow_core import WorkflowCore
from .workflow_types import WorkflowNode, WorkflowResult, WorkflowStateProtocol


class WorkflowExecution(WorkflowCore):
    def _initial_async_state(
        self: WorkflowStateProtocol,
        initial_context: Optional[str],
    ) -> tuple[
        Dict[str, List[str]],
        Set[str],
        Set[str],
        Set[str],
    ]:
        upstream_contexts: Dict[str, List[str]] = {}
        if initial_context:
            upstream_contexts[self.start_node] = [initial_context]

        self._results = {}
        ready_nodes: Set[str] = {self.start_node}
        completed_nodes: Set[str] = set()
        pending_nodes: Set[str] = set(self._next_reachable_nodes)
        pending_nodes.discard(self.start_node)
        return upstream_contexts, ready_nodes, completed_nodes, pending_nodes

    def _start_async_cli(self: WorkflowStateProtocol, cli: Any) -> None:
        cli.workflow_status(
            self.name,
            f"Starting async execution. Max parallel workers: {self.async_executor.max_workers}",
            step=0,
        )
        cli.start_parallel()

    def _finish_async_cli(self: WorkflowStateProtocol, cli: Any) -> None:
        cli.end_parallel()
        cli.workflow_status(
            self.name,
            f"Completed async execution. Results for {len(self._results)} nodes.",
            step=len(self._results),
        )

    @staticmethod
    def _instance_inputs_for_node(node: WorkflowNode) -> List[Optional[str]]:
        inputs = list(getattr(node, "instance_inputs", None) or [])
        return inputs + [None] * (node.instances - len(inputs))

    def _async_context_for_node(
        self: WorkflowStateProtocol,
        node: WorkflowNode,
        node_name: str,
        upstream_contexts: Dict[str, List[str]],
        initial_context: Optional[str],
    ) -> str:
        if upstream_contexts.get(node_name):
            return self._prepare_context(node, upstream_contexts[node_name])
        if initial_context and node_name == self.start_node:
            return initial_context
        return ""

    def _record_async_success(self: WorkflowStateProtocol, result: Dict[str, Any]) -> None:
        node_name = result["node_name"]
        instance_id = result["instance_id"]
        execution_output = result["result"]
        final_message = execution_output.get("final_message", "")
        summary = execution_output.get("summary", "") or final_message

        if node_name not in self._results:
            self._results[node_name] = WorkflowResult(
                name=node_name,
                final=final_message,
                summary=f"[Instance {instance_id}]: {summary}",
                history=execution_output,
                child_summaries={},
            )
            return

        existing = self._results[node_name].summary
        self._results[node_name].summary = f"{existing}\n\n[Instance {instance_id}]: {summary}"

    def _activate_async_dependents(
        self: WorkflowStateProtocol,
        result: Dict[str, Any],
        upstream_contexts: Dict[str, List[str]],
        completed_nodes: Set[str],
        ready_nodes: Set[str],
        pending_nodes: Set[str],
    ) -> None:
        if not result.get("success"):
            return
        node_name = result["node_name"]
        completed_nodes.add(node_name)
        summary = result["result"].get("summary", result["result"].get("final_message", ""))
        for edge in self._next_edges_by_source.get(node_name, []):
            upstream_contexts.setdefault(edge.target, []).append(summary)
            target_deps = self._node_dependencies[edge.target]
            if (
                target_deps.issubset(completed_nodes)
                and edge.target in self._next_reachable_nodes
                and edge.target not in completed_nodes
                and edge.target not in ready_nodes
            ):
                ready_nodes.add(edge.target)
                pending_nodes.discard(edge.target)

    def _activate_unblocked_pending_nodes(
        self: WorkflowStateProtocol,
        pending_nodes: Set[str],
        completed_nodes: Set[str],
        ready_nodes: Set[str],
    ) -> None:
        for node_name in list(pending_nodes):
            node_deps = self._node_dependencies[node_name]
            if node_name in self._next_reachable_nodes and node_deps.issubset(completed_nodes):
                ready_nodes.add(node_name)
                pending_nodes.discard(node_name)

    def run_async(self: WorkflowStateProtocol, initial_context: Optional[str] = None) -> Dict[str, WorkflowResult]:
        self._validate_execution_graph()
        from .workflow_context_cache import reset_context_cache
        reset_context_cache(self, initial_context)
        upstream_contexts, ready_nodes, completed_nodes, pending_nodes = self._initial_async_state(
            initial_context
        )
        cli = get_cli_output()
        self._start_async_cli(cli)

        from .workflow_dataflow import DataflowRun
        try:
            DataflowRun(self, upstream_contexts, ready_nodes, completed_nodes, pending_nodes, initial_context, cli).run()
            return self._results
        finally:
            self._finish_async_cli(cli)
