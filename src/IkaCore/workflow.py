from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Any
from copy import deepcopy

from IkaCore.agents import IkaBaseAgent, summarise_message_history
from IkaCore.cli_output import get_cli_output, OutputType


WorkflowCompressionHook = Callable[[List[str], IkaBaseAgent], str]


@dataclass
class WorkflowEdge:
    source: str
    target: str
    edge_type: str = "next"  # "next" continues execution, "child" returns to parent context
    stage_index: Optional[int] = None  # optional stage binding for staged agents

    def __post_init__(self) -> None:
        if self.edge_type not in {"next", "child"}:
            raise ValueError("edge_type must be 'next' or 'child'")


@dataclass
class WorkflowNode:
    name: str
    agent: IkaBaseAgent
    stage_wiring: Optional[Dict[int, Dict[str, List[IkaBaseAgent]]]] = None
    metadata: Optional[Dict] = None
    instances: int = 1  # Number of parallel instances to run
    instance_inputs: Optional[List[str]] = None  # Optional list of inputs for each instance


@dataclass
class WorkflowResult:
    name: str
    final: str
    summary: str
    history: Dict
    child_summaries: Dict[str, str] = field(default_factory=dict)


class AsyncWorkflowExecutor:
    """
    Executes workflow nodes asynchronously with support for parallel execution
    of multiple agent instances.
    """

    def __init__(self, max_workers: int = 30):
        """
        Args:
            max_workers: Maximum number of concurrent agent executions
        """
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
        """
        Schedule a node for async execution.
        
        Args:
            node_name: Name of the node
            agent: Agent instance to execute
            context: Optional context to inject
            instance_id: Instance ID for parallel runs of the same agent
        
        Returns:
            Future object representing the execution
        """
        def run_agent():
            try:
                # Set instance ID for CLI output buffering
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
        """
        Wait for multiple futures to complete.
        
        Args:
            futures: List of futures to wait for
            timeout: Optional timeout in seconds
        
        Returns:
            List of results
        """
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
        self._node_dependencies: Dict[str, Set[str]] = {}  # Maps node -> set of nodes it depends on
        self._node_dependents: Dict[str, Set[str]] = {}  # Maps node -> set of nodes that depend on it

        self._validate_nodes()
        self._validate_edges()
        self._bind_stage_wiring()
        self._build_dependency_graph()

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

    # ------------------------------------------------------------------
    # Validation and setup
    # ------------------------------------------------------------------
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

    def _bind_stage_wiring(self) -> None:
        """
        Convert child edges that target specific stages into stage wiring
        so staged agents gain access to child agents for that stage.
        """
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
        """
        Build dependency graph for parallel execution.
        Nodes depend on their upstream nodes (sources of incoming edges).
        """
        for node in self.nodes:
            self._node_dependencies[node.name] = set()
            self._node_dependents[node.name] = set()
        
        for edge in self.edges:
            if edge.edge_type == "next":
                # Target depends on source
                self._node_dependencies[edge.target].add(edge.source)
                self._node_dependents[edge.source].add(edge.target)

    # ------------------------------------------------------------------
    # Compression / summarisation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
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

        # Run child edges immediately so their summaries return to the parent.
        for edge in self.edges:
            if edge.source != node_name or edge.edge_type != "child":
                continue
            upstream_contexts.setdefault(edge.target, []).append(summary)
            child_result = self._run_node(edge.target, upstream_contexts)
            child_summaries[edge.target] = child_result.summary

        # Combine parent summary with child summaries for downstream next edges.
        downstream_context = self.compress_hook(
            [summary] + list(child_summaries.values()),
            node.agent,
        )

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
        """
        Async version of _run_node that supports parallel execution.
        """
        if node_name in self._results and instance_id == 0:
            # Only cache results for the first instance
            return self._results[node_name]
        
        node = self._node_index[node_name]
        
        # Create a copy of the agent for parallel execution
        agent_copy = deepcopy(node.agent)
        if instance_id > 0:
            agent_copy.name = f"{node.agent.name}_instance_{instance_id}"
        
        context_payload = upstream_contexts.get(node_name, [])
        context_text = self._prepare_context(node, context_payload)
        
        # Use instance-specific input if provided
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
        """
        Create a deep copy of an agent for parallel execution.
        
        Args:
            agent: Original agent to copy
            instance_id: Instance identifier
            instance_input: Optional input to override agent prompt
        
        Returns:
            New agent instance
        """
        agent_copy = deepcopy(agent)
        if instance_id > 0:
            agent_copy.name = f"{agent.name}_instance_{instance_id}"
        
        if instance_input:
            agent_copy.prompt = instance_input
        
        return agent_copy

    def run_async(self, initial_context: Optional[str] = None) -> Dict[str, WorkflowResult]:
        """
        Run workflow asynchronously with parallel execution support.
        
        This method:
        1. Identifies nodes that can run in parallel (no dependencies)
        2. Executes multiple instances of the same agent if specified
        3. Waits for dependencies before executing dependent nodes
        4. Aggregates results from parallel instances
        """
        upstream_contexts: Dict[str, List[str]] = {}
        if initial_context:
            upstream_contexts[self.start_node] = [initial_context]
        
        self._results = {}
        self._visiting = set()
        
        # Track which nodes are ready to execute (all dependencies satisfied)
        ready_nodes: Set[str] = {self.start_node}
        completed_nodes: Set[str] = set()
        pending_nodes: Set[str] = set(node.name for node in self.nodes)
        pending_nodes.discard(self.start_node)
        
        # Track parallel execution futures
        node_futures: Dict[str, List[concurrent.futures.Future]] = {}

        # Initialize CLI output and start buffering for parallel execution
        cli = get_cli_output()
        cli.workflow_status(
            self.name,
            f"Starting async execution. Max parallel workers: {self.async_executor.max_workers}",
            step=0
        )
        cli.start_parallel()
        
        while ready_nodes or any(node_futures.values()):
            # Execute all ready nodes in parallel
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
                    
                    # Create agent instance for parallel execution
                    agent_instance = self._create_agent_instance(node.agent, instance_id, instance_input)
                    if node.stage_wiring:
                        agent_instance.apply_workflow_stage_wiring(node.stage_wiring)

                    # Prepare context
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
            
            # Wait for at least one batch to complete
            if current_futures:
                results = self.async_executor.wait_for_completion(current_futures)
                
                # Aggregate results from parallel instances
                for result in results:
                    node_name = result['node_name']
                    instance_id = result['instance_id']
                    
                    if node_name not in self._results:
                        # First instance result becomes the primary result
                        execution_output = result['result']
                        final_message = execution_output.get("final_message", "")
                        summary = execution_output.get("summary", final_message)
                        
                        self._results[node_name] = WorkflowResult(
                            name=node_name,
                            final=final_message,
                            summary=summary,
                            history=execution_output,
                            child_summaries={},
                        )
                    else:
                        # Aggregate additional instance results
                        if instance_id > 0:
                            execution_output = result['result']
                            summary = execution_output.get("summary", execution_output.get("final_message", ""))
                            # Append to existing summary
                            existing_summary = self._results[node_name].summary
                            self._results[node_name].summary = f"{existing_summary}\n\n[Instance {instance_id}]: {summary}"
                
                # Update upstream contexts for dependent nodes
                for result in results:
                    node_name = result['node_name']
                    if result['success']:
                        summary = result['result'].get("summary", result['result'].get("final_message", ""))
                        
                        # Update contexts for next edges
                        for edge in self.edges:
                            if edge.source == node_name and edge.edge_type == "next":
                                upstream_contexts.setdefault(edge.target, []).append(summary)
                                
                                # Check if target node is now ready (all dependencies satisfied)
                                target_deps = self._node_dependencies[edge.target]
                                if target_deps.issubset(completed_nodes):
                                    if edge.target not in completed_nodes and edge.target not in ready_nodes:
                                        ready_nodes.add(edge.target)
                                        pending_nodes.discard(edge.target)
            
            # Check for newly ready nodes (dependencies satisfied)
            for node_name in list(pending_nodes):
                node_deps = self._node_dependencies[node_name]
                if node_deps.issubset(completed_nodes):
                    ready_nodes.add(node_name)
                    pending_nodes.discard(node_name)
        
        # Wait for any remaining futures
        self.async_executor.drain()

        # End parallel buffering and flush output in logical order
        cli.end_parallel()

        cli.workflow_status(
            self.name,
            f"Completed async execution. Results for {len(self._results)} nodes.",
            step=len(self._results)
        )
        return self._results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, initial_context: Optional[str] = None, use_async: bool = False) -> Dict[str, WorkflowResult]:
        """
        Run the workflow.
        
        Args:
            initial_context: Optional initial context for the start node
            use_async: If True, use async parallel execution
        
        Returns:
            Dictionary of node names to WorkflowResult
        """
        if use_async:
            return self.run_async(initial_context)
        
        upstream_contexts: Dict[str, List[str]] = {}
        if initial_context:
            upstream_contexts[self.start_node] = [initial_context]
        self._results = {}
        self._visiting = set()
        self._run_node(self.start_node, upstream_contexts)
        return self._results
