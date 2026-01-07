from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from agents import IkaBaseAgent, summarise_message_history


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


@dataclass
class WorkflowResult:
    name: str
    final: str
    summary: str
    history: Dict
    child_summaries: Dict[str, str] = field(default_factory=dict)


class AsyncWorkflowExecutor:
    """
    Skeleton for future async/parallel execution support.
    Implementors can extend schedule_node and drain to add real concurrency.
    """

    def schedule_node(self, node_name: str, runner: "IkaWorkflow") -> None:
        raise NotImplementedError("Async execution skeleton not implemented yet.")

    def drain(self) -> None:
        raise NotImplementedError("Async execution skeleton not implemented yet.")


class IkaWorkflow:
    def __init__(
        self,
        name: str,
        description: str,
        nodes: List[WorkflowNode],
        edges: List[WorkflowEdge],
        compress_hook: Optional[WorkflowCompressionHook] = None,
        start_node: Optional[str] = None,
    ):
        if not nodes:
            raise ValueError("Workflow requires at least one node.")
        self.name = name
        self.description = description
        self.nodes = nodes
        self.edges = edges
        self.compress_hook = compress_hook or self._default_compress_hook
        self.start_node = start_node or nodes[0].name

        self._node_index: Dict[str, WorkflowNode] = {node.name: node for node in nodes}
        self._results: Dict[str, WorkflowResult] = {}
        self._visiting: Set[str] = set()

        self._validate_nodes()
        self._validate_edges()
        self._bind_stage_wiring()

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

    # ------------------------------------------------------------------
    # Compression / summarisation
    # ------------------------------------------------------------------
    def _default_compress_hook(self, contexts: List[str], agent: IkaBaseAgent) -> str:
        """Default compression that reuses summarise_message_history when possible."""
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, initial_context: Optional[str] = None) -> Dict[str, WorkflowResult]:
        upstream_contexts: Dict[str, List[str]] = {}
        if initial_context:
            upstream_contexts[self.start_node] = [initial_context]
        self._results = {}
        self._visiting = set()
        self._run_node(self.start_node, upstream_contexts)
        return self._results