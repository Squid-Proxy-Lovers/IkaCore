"""Event-driven (dataflow) scheduling for asynchronous workflow runs.

A node is started the moment every one of its dependencies has completed,
rather than when a whole wave of concurrently running nodes has finished, so
a fast branch is never held back by an unrelated slow one: with the executor
not saturated, the run finishes on the graph's critical path.

Starting a node has two phases. Its context (upstream results passed through
``compress_hook``, usually an LLM summarisation) is prepared once per node on
a small private pool, so the scheduler thread never blocks on it and the
node's instances share one prepared context instead of each triggering its
own identical summarisation. Then its instances are handed to the workflow's
executor through the ordinary ``schedule_node`` contract.

Completion semantics are unchanged: a node completes when all its instances
have finished; if any instance failed, none of its results are recorded and
its dependents do not run; results and downstream contexts are recorded
through the same helpers as before, in completion order.
"""

from __future__ import annotations

import concurrent.futures
import itertools
from typing import Any, Dict, List, Optional, Set

from .workflow_types import WorkflowStateProtocol

_CONTEXT = "context"
_INSTANCE = "instance"


class _DataflowNodePhases:
    """Per-node phases: hand prepared instances to the executor, then complete the node."""

    workflow: WorkflowStateProtocol
    upstream_contexts: Dict[str, List[str]]
    ready_nodes: Set[str]
    completed_nodes: Set[str]
    pending_nodes: Set[str]
    cli: Any
    instance_results: Dict[str, List[Dict[str, Any]]]
    remaining: Dict[str, int]

    def _track(self, future: concurrent.futures.Future[Any], kind: str, node_name: str) -> None:
        raise NotImplementedError

    def _schedule_instances(self, node_name: str, context: str) -> None:
        workflow = self.workflow
        node = workflow._node_index[node_name]
        instance_inputs = workflow._instance_inputs_for_node(node)
        self.cli.workflow_status(
            workflow.name,
            f"Executing node '{node_name}' with {node.instances} instance(s)",
            step=len(self.completed_nodes) + 1,
        )
        self.instance_results[node_name] = []
        self.remaining[node_name] = node.instances
        for instance_id in range(node.instances):
            instance_input = instance_inputs[instance_id] if instance_id < len(instance_inputs) else None
            agent = workflow._create_agent_instance(node.agent, instance_id, instance_input)
            if node.stage_wiring:
                agent.apply_workflow_stage_wiring(node.stage_wiring)
            future = workflow.async_executor.schedule_node(
                node_name=node_name, agent=agent, context=context, instance_id=instance_id
            )
            self._track(future, _INSTANCE, node_name)

    def _complete_node(self, node_name: str) -> None:
        results = self.instance_results.pop(node_name)
        blocked = any(not result.get("success") for result in results)
        if not blocked:
            for result in results:
                self.workflow._record_async_success(result)
            for result in results:
                self.workflow._activate_async_dependents(
                    result, self.upstream_contexts, self.completed_nodes, self.ready_nodes, self.pending_nodes
                )
        self.workflow._activate_unblocked_pending_nodes(self.pending_nodes, self.completed_nodes, self.ready_nodes)


class DataflowRun(_DataflowNodePhases):
    def __init__(
        self,
        workflow: WorkflowStateProtocol,
        upstream_contexts: Dict[str, List[str]],
        ready_nodes: Set[str],
        completed_nodes: Set[str],
        pending_nodes: Set[str],
        initial_context: Optional[str],
        cli: Any,
    ) -> None:
        self.workflow = workflow
        self.upstream_contexts = upstream_contexts
        self.ready_nodes = ready_nodes
        self.completed_nodes = completed_nodes
        self.pending_nodes = pending_nodes
        self.initial_context = initial_context
        self.cli = cli
        self.inflight: Dict[concurrent.futures.Future[Any], tuple[int, str, str]] = {}
        self.instance_results = {}
        self.remaining = {}
        self._sequence = itertools.count()
        workers = max(1, int(getattr(workflow.async_executor, "max_workers", 1) or 1))
        self.context_pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wf-context")

    def run(self) -> None:
        try:
            self._start_ready_nodes()
            while self.inflight:
                done, _ = concurrent.futures.wait(list(self.inflight), return_when=concurrent.futures.FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: self.inflight[f][0]):
                    self._on_done(future)
        finally:
            self.context_pool.shutdown(wait=True, cancel_futures=True)

    def _track(self, future: concurrent.futures.Future[Any], kind: str, node_name: str) -> None:
        self.inflight[future] = (next(self._sequence), kind, node_name)

    def _start_ready_nodes(self) -> None:
        for node_name in sorted(self.ready_nodes):
            self.ready_nodes.discard(node_name)
            node = self.workflow._node_index[node_name]
            contexts = list(self.upstream_contexts.get(node_name, []))  # complete: every dependency is done
            snapshot = {node_name: contexts} if contexts else {}
            future = self.context_pool.submit(
                self.workflow._async_context_for_node, node, node_name, snapshot, self.initial_context
            )
            self._track(future, _CONTEXT, node_name)

    def _on_done(self, future: concurrent.futures.Future[Any]) -> None:
        _, kind, node_name = self.inflight.pop(future)
        if kind == _CONTEXT:
            self._schedule_instances(node_name, future.result())
            return
        result = self.workflow.async_executor.wait_for_completion([future])[0]
        self.instance_results[node_name].append(result)
        self.remaining[node_name] -= 1
        if self.remaining[node_name] == 0:
            self._complete_node(node_name)
            self._start_ready_nodes()


__all__ = ["DataflowRun"]
