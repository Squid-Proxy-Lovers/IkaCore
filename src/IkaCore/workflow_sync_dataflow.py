"""Dependency-aware synchronous graph execution."""

# pyright: strict, reportPrivateUsage=false

from typing import Any, Optional

from IkaModel.request_control import check_request_controls
from IkaModel.runtime_errors import IkaRequestControlError

from .workflow_async_executor import _run_async_workflow_agent
from .workflow_snapshot_execution import wrap_snapshot_instance
from .workflow_types import WorkflowError, WorkflowNodeError, WorkflowResult


def run_sync_dataflow(workflow: Any, initial_context: Optional[str]) -> dict[str, WorkflowResult]:
    upstream: dict[str, list[str]] = {workflow.start_node: [initial_context]} if initial_context else {}
    pending = set(workflow._next_reachable_nodes)
    completed: set[str] = set()
    stored_results: dict[str, WorkflowResult] = {}
    workflow._results = stored_results
    while pending:
        ready = sorted(name for name in pending if workflow._node_dependencies[name] <= completed)
        if not ready:
            raise ValueError("workflow dependencies are cyclic or require an unreachable node")
        for name in ready:
            check_request_controls()
            node = workflow._node_index[name]
            try:
                context = workflow._prepare_context(node, upstream.get(name, []))
            except IkaRequestControlError:
                raise
            except Exception as error:
                raise WorkflowError([WorkflowNodeError(name, None, type(error).__name__, str(error))],
                                    stored_results) from error
            inputs = workflow._instance_inputs_for_node(node)
            results: list[dict[str, Any]] = []
            for instance in range(node.instances):
                agent = workflow._create_agent_instance(node.agent, instance, inputs[instance])
                agent = wrap_snapshot_instance(workflow, name, instance, agent)
                if node.stage_wiring:
                    agent.apply_workflow_stage_wiring(node.stage_wiring)
                if context:
                    agent.inject_workflow_context(context)
                result = _run_async_workflow_agent(name, agent, "", instance)
                if result.get("interrupted"):
                    workflow._interruptions[f"{name}:{instance}"] = result["result"]
                    return stored_results
                if not result["success"]:
                    error = result["result"]
                    raise WorkflowError([WorkflowNodeError(name, instance, error["error_type"], error["error"])],
                                        stored_results)
                results.append(result)
            for result in results:
                workflow._record_async_success(result)
                summary = result["result"].get("summary") or result["result"].get("final_message") or ""
                for target in workflow._node_dependents[name]:
                    upstream.setdefault(target, []).append(summary)
            pending.remove(name)
            completed.add(name)
    return stored_results
