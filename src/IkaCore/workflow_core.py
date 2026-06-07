"""Core workflow graph and node execution mixins."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, cast

from IkaCore.agents import IkaBaseAgent
from IkaModel.chat_interface.chat_interface import summarise_message_history

from .workflow_types import WorkflowNode, WorkflowResult, WorkflowStateProtocol


class WorkflowDisplayMixin:
    def __repr__(self: WorkflowStateProtocol) -> str:
        lines = [
            f"IkaWorkflow: {self.name}",
            f"Description: {self.description}",
            f"Start Node: {self.start_node}",
            "",
            "Nodes:",
        ]
        for node in self.nodes:
            agent_info = f"{node.agent.name} ({node.agent.__class__.__name__})"
            if node.agent.Stages:
                agent_info += f" [{len(node.agent.Stages)} stages]"
            lines.append(f"  - {node.name}: {agent_info}")
            if node.stage_wiring:
                lines.append(f"    Stage Wiring: {list(node.stage_wiring.keys())}")
        lines.extend(["", "Edges:"])
        for edge in self.edges:
            edge_symbol = "-->" if edge.edge_type == "next" else "<->"
            stage_info = f" [stage {edge.stage_index}]" if edge.stage_index is not None else ""
            lines.append(f"  {edge.source} {edge_symbol} {edge.target} ({edge.edge_type}){stage_info}")

        lines.extend(["", "Graph Structure:"])

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

            for edge in self._edges_by_source.get(node_name, []):
                edge_label = "child" if edge.edge_type == "child" else "next"
                stage_label = f" [stage {edge.stage_index}]" if edge.stage_index is not None else ""
                result.append(f"{prefix}  |--({edge_label}{stage_label})-->")
                result.extend(print_node(edge.target, indent + 2, visited.copy()))

            return result

        lines.extend(print_node(self.start_node))
        return "\n".join(lines)


class WorkflowGraphMixin(WorkflowDisplayMixin):
    def _validate_nodes(self: WorkflowStateProtocol) -> None:
        if self.start_node not in self._node_index:
            raise ValueError(f"start_node '{self.start_node}' is not defined in nodes.")

        for node in self.nodes:
            agent = node.agent
            if getattr(agent, "Stages", None):
                if agent.subagents or agent.next_agent:
                    raise ValueError(
                        f"Agent '{agent.name}' has stages and cannot declare subagents or next_agent directly."
                    )

    def _validate_edges(self: WorkflowStateProtocol) -> None:
        for edge in self.edges:
            if edge.source not in self._node_index:
                raise ValueError(f"Edge source '{edge.source}' is not in workflow nodes.")
            if edge.target not in self._node_index:
                raise ValueError(f"Edge target '{edge.target}' is not in workflow nodes.")
            if edge.edge_type == "child" and edge.stage_index is None:
                raise ValueError(f"Child edge '{edge.source}' -> '{edge.target}' must define stage_index.")

    def _bind_stage_wiring(self: WorkflowStateProtocol) -> None:
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

    def _build_dependency_graph(self: WorkflowStateProtocol) -> None:
        for node in self.nodes:
            self._node_dependencies[node.name] = set()
            self._node_dependents[node.name] = set()
            self._edges_by_source[node.name] = []
            self._next_edges_by_source[node.name] = []

        for edge in self.edges:
            self._edges_by_source.setdefault(edge.source, []).append(edge)
            if edge.edge_type == "next":
                self._next_edges_by_source.setdefault(edge.source, []).append(edge)
                self._node_dependencies[edge.target].add(edge.source)
                self._node_dependents[edge.source].add(edge.target)

    def _compute_next_reachable_nodes(self: WorkflowStateProtocol) -> Set[str]:
        reachable: Set[str] = set()
        stack: List[str] = [self.start_node]
        while stack:
            node_name = stack.pop()
            if node_name in reachable:
                continue
            reachable.add(node_name)
            for edge in self._next_edges_by_source.get(node_name, []):
                stack.append(edge.target)
        return reachable


class WorkflowContextMixin(WorkflowGraphMixin):
    def _default_compress_hook(self, contexts: List[str], agent: IkaBaseAgent) -> str:
        merged = "\n\n".join([c for c in contexts if c]) if contexts else ""
        if not merged:
            return ""
        try:
            barebone = cast(Any, agent).get_barebone(agent.system_prompt or agent.description or agent.prompt, [])
            history = {
                "system": {"message": merged, "tokens": 0},
                "first_input": {"message": merged, "tokens": 0},
                "summary": {"message": "", "tokens": 0},
                "messages": {},
            }
            summary = summarise_message_history(barebone, history)
            return summary or merged
        except (RuntimeError, ValueError, TypeError, KeyError):
            return merged

    def _apply_stage_wiring(self, node: WorkflowNode) -> None:
        if node.stage_wiring:
            node.agent.apply_workflow_stage_wiring(node.stage_wiring)

    def _prepare_context(self: WorkflowStateProtocol, node: WorkflowNode, upstream: List[str]) -> str:
        return self.compress_hook(upstream, node.agent) if upstream else ""


class WorkflowSyncNodeExecutionMixin(WorkflowContextMixin):
    def _run_node(self: WorkflowStateProtocol, node_name: str, upstream_contexts: Dict[str, List[str]]) -> WorkflowResult:
        if node_name in self._results:
            return self._results[node_name]
        if node_name in self._visiting:
            raise ValueError(f"Cycle detected in workflow at node '{node_name}'.")

        self._visiting.add(node_name)
        node = self._node_index[node_name]
        context_text = self._prepare_context(node, upstream_contexts.get(node_name, []))

        self._apply_stage_wiring(node)
        if context_text:
            node.agent.inject_workflow_context(context_text)

        execution_output = node.agent.execution()
        final_message = execution_output.get("final_message") or ""
        summary = execution_output.get("summary") or final_message
        downstream_context = self.compress_hook([summary], node.agent)

        result = WorkflowResult(
            name=node_name,
            final=final_message,
            summary=summary,
            history=execution_output,
            child_summaries={},
        )
        self._results[node_name] = result
        self._visiting.remove(node_name)

        for edge in self._next_edges_by_source.get(node_name, []):
            upstream_contexts.setdefault(edge.target, []).append(downstream_context)
            self._run_node(edge.target, upstream_contexts)

        return result


class WorkflowNodeExecutionMixin(WorkflowSyncNodeExecutionMixin):
    def _run_node_async(
        self: WorkflowStateProtocol,
        node_name: str,
        upstream_contexts: Dict[str, List[str]],
        instance_id: int = 0,
        instance_input: Optional[str] = None,
    ) -> WorkflowResult:
        if node_name in self._results and instance_id == 0:
            return self._results[node_name]

        node = self._node_index[node_name]
        agent_copy = node.agent.clone_for_run()
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

    def _create_agent_instance(
        self,
        agent: IkaBaseAgent,
        instance_id: int,
        instance_input: Optional[str] = None,
    ) -> IkaBaseAgent:
        agent_copy = agent.clone_for_run()
        if instance_id > 0:
            agent_copy.name = f"{agent.name}_instance_{instance_id}"

        if instance_input:
            agent_copy.prompt = instance_input

        return agent_copy

