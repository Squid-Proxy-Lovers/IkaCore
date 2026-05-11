from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Any
from copy import deepcopy

from IkaCore.agents import IkaBaseAgent, summarise_message_history
from IkaCore.cli_output import get_cli_output


WorkflowCompressionHook = Callable[[List[str], IkaBaseAgent], str]


@dataclass
class WorkflowEdge:
    """
    Base class used to define exectuion flow between nodes
    and exectuion type ie next or child. The stage index 
    is used to allow for staged agents to be executed.
    - we can only have stage index if the edge_type is child.
    --> the idea here is that we allow the existance of staged agents, 
    and creating specific subagents for each stage.

    Example:

    edges = [WorkflowEdge(source="node_a", target="node_b", edge_type="next"),]
    This will execute node_a and then node_b in a linear fashion.

    here is an example of a staged agent workflow:
    edges = [
        WorkflowEdge(source="node_a", target="node_b", edge_type="child", stage_index=1),
        WorkflowEdge(source="node_a", target="node_c", edge_type="child", stage_index=2),
        WorkflowEdge(source="node_c", target="node_d", edge_type="next"),
    ]
    This will execute node_a and then node_b and node_c in a staged fashion.
    node_b will be executed with the stage_index 1 and node_c will be executed with the stage_index 2.
    node_d will be executed after node_c.
    """

    source: str
    target: str
    edge_type: str = "next"
    stage_index: Optional[int] = None

    def __post_init__(self) -> None:
        if self.edge_type not in {"next", "child"}:
            raise ValueError("edge_type must be next or child")


@dataclass
class WorkflowNode:
    """
    This is the base class used to define a node in the workflow

    It contains the agent to be executed and the stage_wiring
    The stage_wiring is used to define the subagents to be executed for each stage

    The instances is used to define the number of instances of the agent to be executed
    -> this is used to allow for parallel execution of the same agent

    The instance_inputs is used to define the input for each instance
    -> this is used to allow for parallel execution of the same agent with different inputs (useful for compelex task and trying different approaches)
    """
    name: str
    agent: IkaBaseAgent
    stage_wiring: Optional[Dict[int, Dict[str, List[IkaBaseAgent]]]] = None
    instances: int = 1
    instance_inputs: Optional[List[str]] = None


@dataclass
class WorkflowResult:
    """
    This is how we managed agent messages and results
    a) final: the final message from the agent
    b) summary: when enabled we have summarzation based on the agent message-history 
     ==> this is not enabled by default, but can be enabled by setting the summarize_final attribute to True in the IkaAgent class
    c) history: this the raw result of `agent.execution()`
    d) child_summaries: the summaries of the child agents, we can use this 
    for compressing the context for the next agent in the workflow
    """
    name: str
    final: str
    summary: str
    history: Dict
    child_summaries: Dict[str, str] = field(default_factory=dict)


class AsyncWorkflowExecutor:
    def __init__(self, max_workers: int = 30):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending_tasks: Dict[str, concurrent.futures.Future] = {}
        self.completed_results: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def schedule_node(
        self,
        node_name: str,
        agent: IkaBaseAgent,
        context: str = "",
        instance_id: int = 0
    ) -> concurrent.futures.Future:
        def run_agent():
            try:
                cli = get_cli_output()
                cli.set_instance_id(instance_id)

                if context:
                    agent.inject_workflow_context(context)

                if getattr(agent, "use_async", False) and hasattr(agent, "async_execution"):
                    result = asyncio.run(agent.async_execution())
                else:
                    result = agent.execution()
                return {
                    "node_name": node_name,
                    "instance_id": instance_id,
                    "result": result,
                    "success": True
                }
            except Exception as e:
                return {
                    "node_name": node_name,
                    "instance_id": instance_id,
                    "result": {"error": str(e)},
                    "success": False
                }
        
        task_id = f"{node_name}_{instance_id}"
        future = self.executor.submit(run_agent)
        
        with self.lock:
            self.pending_tasks[task_id] = future
        
        return future

    def wait_for_completion(self, futures: List[concurrent.futures.Future], timeout: Optional[float] = None) -> List[Any]:
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
                with self.lock:
                    task_id = f"{result['node_name']}_{result['instance_id']}"
                    self.completed_results[task_id] = result
                    if task_id in self.pending_tasks:
                        del self.pending_tasks[task_id]
            except Exception as e:
                results.append({
                    "result": {"error": str(e)},
                    "success": False
                })
        return results

    def drain(self) -> None:
        with self.lock:
            pending = list(self.pending_tasks.values())
        
        if pending:
            self.wait_for_completion(pending)

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)


class IkaWorkflow:
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

        self._validate_nodes()
        self._validate_edges()
        self._bind_stage_wiring()
        self._build_dependency_graph()
        self._next_reachable_nodes: Set[str] = self._compute_next_reachable_nodes()

    def __repr__(self) -> str:
        lines = []
        lines.append(f"IkaWorkflow: {self.name}")
        lines.append(f"Description: {self.description}")
        lines.append(f"Start Node: {self.start_node}")
        lines.append("")
        lines.append("Nodes:")
        for node in self.nodes:
            agent_info = f"{node.agent.name} ({node.agent.__class__.__name__})"
            if node.agent.Stages:
                agent_info += f" [{len(node.agent.Stages)} stages]"
            lines.append(f"  - {node.name}: {agent_info}")
            if node.stage_wiring:
                lines.append(f"    Stage Wiring: {list(node.stage_wiring.keys())}")
        lines.append("")
        lines.append("Edges:")
        for edge in self.edges:
            edge_symbol = "-->" if edge.edge_type == "next" else "<->"
            stage_info = f" [stage {edge.stage_index}]" if edge.stage_index is not None else ""
            lines.append(f"  {edge.source} {edge_symbol} {edge.target} ({edge.edge_type}){stage_info}")
        
        lines.append("")
        lines.append("Graph Structure:")
        node_connections = {}
        for edge in self.edges:
            node_connections.setdefault(edge.source, []).append((edge.target, edge.edge_type, edge.stage_index))
        
        def print_node(node_name: str, indent: int = 0, visited: Optional[Set[str]] = None) -> List[str]:
            if visited is None:
                visited = set()
            result = []
            prefix = "  " * indent
            marker = "*" if node_name == self.start_node else "-"
            cycle_marker = " (cycle)" if node_name in visited else ""
            result.append(f"{prefix}{marker} {node_name}{cycle_marker}")
            
            if node_name in visited:
                return result
            visited.add(node_name)
            
            if node_name in node_connections:
                for target, edge_type, stage_idx in node_connections[node_name]:
                    edge_label = "child" if edge_type == "child" else "next"
                    stage_label = f" [stage {stage_idx}]" if stage_idx is not None else ""
                    result.append(f"{prefix}  |--({edge_label}{stage_label})-->")
                    result.extend(print_node(target, indent + 2, visited.copy()))
            
            return result
        
        lines.extend(print_node(self.start_node))
        return "\n".join(lines)

    def _validate_nodes(self) -> None:
        if self.start_node not in self._node_index:
            raise ValueError(f"start_node '{self.start_node}' is not defined in nodes.")

        for node in self.nodes:
            agent = node.agent
            if getattr(agent, "Stages", None):
                if agent.subagents or agent.next_agent:
                    raise ValueError(
                        f"Agent '{agent.name}' has stages and cannot declare subagents or next_agent directly."
                    )

    def _validate_edges(self) -> None:
        for edge in self.edges:
            if edge.source not in self._node_index:
                raise ValueError(f"Edge source '{edge.source}' is not in workflow nodes.")
            if edge.target not in self._node_index:
                raise ValueError(f"Edge target '{edge.target}' is not in workflow nodes.")
            if edge.edge_type == "child" and edge.stage_index is None:
                raise ValueError(f"Child edge '{edge.source}' -> '{edge.target}' must define stage_index.")

    def _bind_stage_wiring(self) -> None:
        for edge in self.edges:
            if edge.edge_type != "child":
                continue
            if edge.stage_index is None:
                continue
            parent = self._node_index[edge.source]
            target = self._node_index[edge.target]
            parent.stage_wiring = parent.stage_wiring or {}
            wiring = parent.stage_wiring.setdefault(edge.stage_index, {})
            wiring.setdefault("subagents", [])
            wiring["subagents"].append(target.agent)

    def _build_dependency_graph(self) -> None:
        for node in self.nodes:
            self._node_dependencies[node.name] = set()
            self._node_dependents[node.name] = set()
        
        for edge in self.edges:
            if edge.edge_type == "next":
                self._node_dependencies[edge.target].add(edge.source)
                self._node_dependents[edge.source].add(edge.target)

    def _compute_next_reachable_nodes(self) -> Set[str]:
        reachable: Set[str] = set()
        stack: List[str] = [self.start_node]
        while stack:
            node_name = stack.pop()
            if node_name in reachable:
                continue
            reachable.add(node_name)
            for edge in self.edges:
                if edge.source == node_name and edge.edge_type == "next":
                    stack.append(edge.target)
        return reachable

    def _default_compress_hook(self, contexts: List[str], agent: IkaBaseAgent) -> str:
        merged = "\n\n".join([c for c in contexts if c]) if contexts else ""
        if not merged:
            return ""
        try:
            barebone = agent.get_barebone(agent.system_prompt or agent.description or agent.prompt, [])
            history = {
                "system": {"message": merged, "tokens": 0},
                "first_input": {"message": merged, "tokens": 0},
                "summary": {"message": "", "tokens": 0},
                "messages": {},
            }
            summary = summarise_message_history(barebone, history)
            return summary or merged
        except Exception:
            return merged

    def _apply_stage_wiring(self, node: WorkflowNode) -> None:
        if node.stage_wiring:
            node.agent.apply_workflow_stage_wiring(node.stage_wiring)

    def _prepare_context(self, node: WorkflowNode, upstream: List[str]) -> str:
        return self.compress_hook(upstream, node.agent) if upstream else ""

    def _run_node(self, node_name: str, upstream_contexts: Dict[str, List[str]]) -> WorkflowResult:
        if node_name in self._results:
            return self._results[node_name]
        if node_name in self._visiting:
            raise ValueError(f"Cycle detected in workflow at node '{node_name}'.")

        self._visiting.add(node_name)
        node = self._node_index[node_name]
        context_payload = upstream_contexts.get(node_name, [])
        context_text = self._prepare_context(node, context_payload)

        self._apply_stage_wiring(node)
        if context_text:
            node.agent.inject_workflow_context(context_text)

        execution_output = node.agent.execution()
        final_message = execution_output.get("final_message") or ""
        summary = execution_output.get("summary") or final_message
        child_summaries: Dict[str, str] = {}
        downstream_context = self.compress_hook([summary], node.agent)

        result = WorkflowResult(
            name=node_name,
            final=final_message,
            summary=summary,
            history=execution_output,
            child_summaries=child_summaries,
        )
        self._results[node_name] = result
        self._visiting.remove(node_name)

        for edge in self.edges:
            if edge.source == node_name and edge.edge_type == "next":
                upstream_contexts.setdefault(edge.target, []).append(downstream_context)
                self._run_node(edge.target, upstream_contexts)

        return result

    def _run_node_async(
        self,
        node_name: str,
        upstream_contexts: Dict[str, List[str]],
        instance_id: int = 0,
        instance_input: Optional[str] = None
    ) -> WorkflowResult:
        if node_name in self._results and instance_id == 0:
            return self._results[node_name]

        node = self._node_index[node_name]
        agent_copy = deepcopy(node.agent)
        if instance_id > 0:
            agent_copy.name = f"{node.agent.name}_instance_{instance_id}"
        
        context_payload = upstream_contexts.get(node_name, [])
        context_text = self._prepare_context(node, context_payload)

        if instance_input:
            agent_copy.prompt = instance_input
            context_text = instance_input

        if node.stage_wiring:
            agent_copy.apply_workflow_stage_wiring(node.stage_wiring)
        if context_text:
            agent_copy.inject_workflow_context(context_text)

        execution_output = agent_copy.execution()
        final_message = execution_output.get("final_message") or ""
        summary = execution_output.get("summary") or final_message

        result = WorkflowResult(
            name=node_name,
            final=final_message,
            summary=summary,
            history=execution_output,
            child_summaries={},
        )
        
        if instance_id == 0:
            self._results[node_name] = result
        
        return result

    def _create_agent_instance(self, agent: IkaBaseAgent, instance_id: int, instance_input: Optional[str] = None) -> IkaBaseAgent:
        agent_copy = deepcopy(agent)
        if instance_id > 0:
            agent_copy.name = f"{agent.name}_instance_{instance_id}"
        
        if instance_input:
            agent_copy.prompt = instance_input
        
        return agent_copy

    def run_async(self, initial_context: Optional[str] = None) -> Dict[str, WorkflowResult]:
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
        cli = get_cli_output()
        cli.workflow_status(
            self.name,
            f"Starting async execution. Max parallel workers: {self.async_executor.max_workers}",
            step=0
        )
        cli.start_parallel()
        
        while ready_nodes or any(node_futures.values()):
            current_futures = []

            for node_name in list(ready_nodes):
                node = self._node_index[node_name]
                num_instances = node.instances
                instance_inputs = getattr(node, 'instance_inputs', None) or [None] * num_instances

                cli.workflow_status(
                    self.name,
                    f"Executing node '{node_name}' with {num_instances} instance(s)",
                    step=len(completed_nodes) + 1
                )

                node_futures[node_name] = []

                for instance_id in range(num_instances):
                    instance_input = instance_inputs[instance_id] if instance_id < len(instance_inputs) else None
                    agent_instance = self._create_agent_instance(node.agent, instance_id, instance_input)
                    if node.stage_wiring:
                        agent_instance.apply_workflow_stage_wiring(node.stage_wiring)
                    context_text = ""
                    if upstream_contexts.get(node_name):
                        context_text = self._prepare_context(node, upstream_contexts[node_name])
                    elif initial_context and node_name == self.start_node:
                        context_text = initial_context
                    
                    future = self.async_executor.schedule_node(
                        node_name=node_name,
                        agent=agent_instance,
                        context=context_text,
                        instance_id=instance_id
                    )
                    node_futures[node_name].append(future)
                    current_futures.append(future)
                ready_nodes.remove(node_name)
                completed_nodes.add(node_name)

            if current_futures:
                results = self.async_executor.wait_for_completion(current_futures)
                for result in results:
                    if not result.get("success"):
                        continue
                    node_name = result['node_name']
                    instance_id = result['instance_id']
                    execution_output = result['result']
                    final_message = execution_output.get("final_message", "")
                    summary = execution_output.get("summary", "")

                    # Robust fallback: use final_message if summary is empty
                    if not summary or summary.strip() == "":
                        summary = final_message

                    if node_name not in self._results:
                        self._results[node_name] = WorkflowResult(
                            name=node_name,
                            final=final_message,
                            summary=f"[Instance {instance_id}]: {summary}",  # Always prefix with instance ID
                            history=execution_output,
                            child_summaries={},
                        )
                    else:
                        existing = self._results[node_name].summary
                        self._results[node_name].summary = f"{existing}\n\n[Instance {instance_id}]: {summary}"
                for result in results:
                    node_name = result['node_name']
                    if result['success']:
                        summary = result['result'].get("summary", result['result'].get("final_message", ""))
                        for edge in self.edges:
                            if edge.source == node_name and edge.edge_type == "next":
                                upstream_contexts.setdefault(edge.target, []).append(summary)
                                target_deps = self._node_dependencies[edge.target]
                                if target_deps.issubset(completed_nodes):
                                    if edge.target in self._next_reachable_nodes and edge.target not in completed_nodes and edge.target not in ready_nodes:
                                        ready_nodes.add(edge.target)
                                        pending_nodes.discard(edge.target)
                for n in {r["node_name"] for r in results}:
                    node_futures.pop(n, None)
            for node_name in list(pending_nodes):
                node_deps = self._node_dependencies[node_name]
                if node_name in self._next_reachable_nodes and node_deps.issubset(completed_nodes):
                    ready_nodes.add(node_name)
                    pending_nodes.discard(node_name)

        self.async_executor.drain()
        cli.end_parallel()

        cli.workflow_status(
            self.name,
            f"Completed async execution. Results for {len(self._results)} nodes.",
            step=len(self._results)
        )
        return self._results

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
