"""Asynchronous workflow scheduling and result-processing mixins."""

from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Optional, Set

from IkaCore.cli_output import get_cli_output

from .workflow_core import WorkflowNodeExecutionMixin
from .workflow_types import WorkflowNode, WorkflowResult, WorkflowStateProtocol


class WorkflowAsyncStateMixin(WorkflowNodeExecutionMixin):
    def _initial_async_state(
        self: WorkflowStateProtocol,
        initial_context: Optional[str],
    ) -> tuple[
        Dict[str, List[str]],
        Set[str],
        Set[str],
        Set[str],
        Dict[str, List[concurrent.futures.Future]],
    ]:
        upstream_contexts: Dict[str, List[str]] = {}
        if initial_context:
            upstream_contexts[self.start_node] = [initial_context]

        self._results = {}
        self._visiting = set()
        ready_nodes: Set[str] = {self.start_node}
        completed_nodes: Set[str] = set()
        pending_nodes: Set[str] = set(self._next_reachable_nodes)
        pending_nodes.discard(self.start_node)
        node_futures: Dict[str, List[concurrent.futures.Future]] = {}
        return upstream_contexts, ready_nodes, completed_nodes, pending_nodes, node_futures

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


class WorkflowAsyncNodeSchedulingMixin(WorkflowAsyncStateMixin):
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

    def _schedule_async_node(
        self: WorkflowStateProtocol,
        node_name: str,
        upstream_contexts: Dict[str, List[str]],
        initial_context: Optional[str],
        cli: Any,
        step: int,
    ) -> List[concurrent.futures.Future]:
        node = self._node_index[node_name]
        instance_inputs = self._instance_inputs_for_node(node)
        cli.workflow_status(
            self.name,
            f"Executing node '{node_name}' with {node.instances} instance(s)",
            step=step,
        )

        futures: List[concurrent.futures.Future] = []
        for instance_id in range(node.instances):
            instance_input = instance_inputs[instance_id] if instance_id < len(instance_inputs) else None
            agent_instance = self._create_agent_instance(node.agent, instance_id, instance_input)
            if node.stage_wiring:
                agent_instance.apply_workflow_stage_wiring(node.stage_wiring)
            future = self.async_executor.schedule_node(
                node_name=node_name,
                agent=agent_instance,
                context=self._async_context_for_node(node, node_name, upstream_contexts, initial_context),
                instance_id=instance_id,
            )
            futures.append(future)
        return futures


class WorkflowAsyncSchedulingMixin(WorkflowAsyncNodeSchedulingMixin):
    def _schedule_ready_async_nodes(
        self: WorkflowStateProtocol,
        ready_nodes: Set[str],
        upstream_contexts: Dict[str, List[str]],
        initial_context: Optional[str],
        node_futures: Dict[str, List[concurrent.futures.Future]],
        cli: Any,
        completed_count: int,
    ) -> List[concurrent.futures.Future]:
        current_futures: List[concurrent.futures.Future] = []
        for node_name in list(ready_nodes):
            futures = self._schedule_async_node(
                node_name,
                upstream_contexts,
                initial_context,
                cli,
                step=completed_count + 1,
            )
            node_futures[node_name] = futures
            current_futures.extend(futures)
            ready_nodes.remove(node_name)
        return current_futures


class WorkflowAsyncRecordMixin(WorkflowAsyncSchedulingMixin):
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


class WorkflowAsyncActivationMixin(WorkflowAsyncRecordMixin):
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


class WorkflowAsyncResultMixin(WorkflowAsyncActivationMixin):
    def _process_async_results(
        self: WorkflowStateProtocol,
        current_futures: List[concurrent.futures.Future],
        upstream_contexts: Dict[str, List[str]],
        completed_nodes: Set[str],
        ready_nodes: Set[str],
        pending_nodes: Set[str],
        node_futures: Dict[str, List[concurrent.futures.Future]],
    ) -> None:
        if not current_futures:
            return
        results = self.async_executor.wait_for_completion(current_futures)
        blocked_nodes = {
            result["node_name"]
            for result in results
            if "node_name" in result and not result.get("success")
        }
        for result in results:
            if result.get("success") and result.get("node_name") not in blocked_nodes:
                self._record_async_success(result)
        for result in results:
            if result.get("node_name") in blocked_nodes:
                continue
            self._activate_async_dependents(result, upstream_contexts, completed_nodes, ready_nodes, pending_nodes)
        for node_name in {result["node_name"] for result in results if "node_name" in result}:
            node_futures.pop(node_name, None)


class WorkflowAsyncExecutionMixin(WorkflowAsyncResultMixin):
    def run_async(self: WorkflowStateProtocol, initial_context: Optional[str] = None) -> Dict[str, WorkflowResult]:
        upstream_contexts, ready_nodes, completed_nodes, pending_nodes, node_futures = self._initial_async_state(
            initial_context
        )
        cli = get_cli_output()
        self._start_async_cli(cli)

        while ready_nodes or any(node_futures.values()):
            current_futures = self._schedule_ready_async_nodes(
                ready_nodes,
                upstream_contexts,
                initial_context,
                node_futures,
                cli,
                completed_count=len(completed_nodes),
            )
            self._process_async_results(
                current_futures,
                upstream_contexts,
                completed_nodes,
                ready_nodes,
                pending_nodes,
                node_futures,
            )
            self._activate_unblocked_pending_nodes(pending_nodes, completed_nodes, ready_nodes)

        self.async_executor.drain()
        self._finish_async_cli(cli)
        return self._results

