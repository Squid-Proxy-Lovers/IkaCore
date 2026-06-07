"""Workflow data types and type-only service contracts."""

# pyright: strict

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from IkaCore.agents import IkaBaseAgent

WorkflowCompressionHook = Callable[[list[str], IkaBaseAgent], str]
WorkflowFuture = concurrent.futures.Future[dict[str, Any]]


@dataclass
class WorkflowEdge:
    """Execution edge between workflow nodes."""

    source: str
    target: str
    edge_type: str = "next"
    stage_index: Optional[int] = None

    def __post_init__(self) -> None:
        if self.edge_type not in {"next", "child"}:
            raise ValueError("edge_type must be next or child")


@dataclass
class WorkflowNode:
    """Workflow node containing an agent and optional execution fan-out."""

    name: str
    agent: IkaBaseAgent
    stage_wiring: Optional[dict[int, dict[str, list[IkaBaseAgent]]]] = None
    instances: int = 1
    instance_inputs: Optional[list[str]] = None

    def __post_init__(self) -> None:
        if self.instances < 1:
            raise ValueError("WorkflowNode.instances must be at least 1")
        if self.instance_inputs is not None and len(self.instance_inputs) > self.instances:
            raise ValueError("WorkflowNode.instance_inputs cannot exceed instances")


@dataclass
class WorkflowResult:
    """Stored result for a completed workflow node."""

    name: str
    final: str
    summary: str
    history: dict[str, Any]
    child_summaries: dict[str, str] = field(default_factory=lambda: {})


class _WorkflowValueState(Protocol):
    name: str
    description: str
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge]
    compress_hook: WorkflowCompressionHook
    start_node: str
    async_executor: Any
    _node_index: dict[str, WorkflowNode]
    _results: dict[str, WorkflowResult]
    _visiting: set[str]
    _node_dependencies: dict[str, set[str]]
    _node_dependents: dict[str, set[str]]
    _edges_by_source: dict[str, list[WorkflowEdge]]
    _next_edges_by_source: dict[str, list[WorkflowEdge]]
    _next_reachable_nodes: set[str]


class _WorkflowCoreServices(Protocol):
    def _prepare_context(self, node: WorkflowNode, upstream: list[str]) -> str:
        ...

    def _apply_stage_wiring(self, node: WorkflowNode) -> None:
        ...

    def _run_node(self, node_name: str, upstream_contexts: dict[str, list[str]]) -> WorkflowResult:
        ...

    def _create_agent_instance(
        self,
        agent: IkaBaseAgent,
        instance_id: int,
        instance_input: Optional[str] = None,
    ) -> IkaBaseAgent:
        ...

    def _instance_inputs_for_node(self, node: WorkflowNode) -> list[Optional[str]]:
        ...


class _WorkflowAsyncSchedulingServices(Protocol):
    def _initial_async_state(
        self,
        initial_context: Optional[str],
    ) -> tuple[
        dict[str, list[str]],
        set[str],
        set[str],
        set[str],
        dict[str, list[WorkflowFuture]],
    ]:
        ...

    def _start_async_cli(self, cli: Any) -> None:
        ...

    def _finish_async_cli(self, cli: Any) -> None:
        ...

    def _async_context_for_node(
        self,
        node: WorkflowNode,
        node_name: str,
        upstream_contexts: dict[str, list[str]],
        initial_context: Optional[str],
    ) -> str:
        ...

    def _schedule_async_node(
        self,
        node_name: str,
        upstream_contexts: dict[str, list[str]],
        initial_context: Optional[str],
        cli: Any,
        step: int,
    ) -> list[WorkflowFuture]:
        ...

    def _schedule_ready_async_nodes(
        self,
        ready_nodes: set[str],
        upstream_contexts: dict[str, list[str]],
        initial_context: Optional[str],
        node_futures: dict[str, list[WorkflowFuture]],
        cli: Any,
        completed_count: int,
    ) -> list[WorkflowFuture]:
        ...


class _WorkflowAsyncResultServices(Protocol):
    def _record_async_success(self, result: dict[str, Any]) -> None:
        ...

    def _activate_async_dependents(
        self,
        result: dict[str, Any],
        upstream_contexts: dict[str, list[str]],
        completed_nodes: set[str],
        ready_nodes: set[str],
        pending_nodes: set[str],
    ) -> None:
        ...

    def _process_async_results(
        self,
        current_futures: list[WorkflowFuture],
        upstream_contexts: dict[str, list[str]],
        completed_nodes: set[str],
        ready_nodes: set[str],
        pending_nodes: set[str],
        node_futures: dict[str, list[WorkflowFuture]],
    ) -> None:
        ...

    def _activate_unblocked_pending_nodes(
        self,
        pending_nodes: set[str],
        completed_nodes: set[str],
        ready_nodes: set[str],
    ) -> None:
        ...


class WorkflowStateProtocol(
    _WorkflowValueState,
    _WorkflowCoreServices,
    _WorkflowAsyncSchedulingServices,
    _WorkflowAsyncResultServices,
    Protocol,
):
    pass
