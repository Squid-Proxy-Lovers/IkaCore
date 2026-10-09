"""Workflow-owned frames and completed-instance reuse, without changing default runs."""

# pyright: strict

from __future__ import annotations

from collections.abc import Callable, Coroutine
from copy import deepcopy
from typing import Any, cast

from IkaModel.async_runner import run_async
from IkaModel.execution_hooks import ExecutionHooks, execution_hooks
from IkaModel.runtime_policy import runtime_options

from .runtime_control import RuntimePauseRequested
from .workflow_snapshot_state import WorkflowSnapshotSession, save_workflow_boundary


def wrap_snapshot_instance(workflow: Any, node_name: str, instance: int, agent: Any) -> Any:
    session: WorkflowSnapshotSession | None = getattr(workflow, "_workflow_snapshot", None)
    if session is None or session.run_id is None:
        return agent
    asynchronous = getattr(agent, "async_execution", None)
    original = (lambda: run_async(cast(Callable[[], Coroutine[Any, Any, Any]], asynchronous))) if agent.use_async and callable(asynchronous) else agent.execution
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return execute_instance(session, node_name, instance, original, args, kwargs)
    agent.execution = wrapped
    agent._workflow_snapshot_instance = True
    return agent


def execute_instance(session: WorkflowSnapshotSession, node: str, instance: int, execute: Callable[..., Any],
                     args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    key = f"{node}:{instance}"
    frame = session.store.create_frame(str(session.run_id), "workflow_node", node,
        parent_frame_id=session.root_frame_id, metadata={"workflow_node_name": node, "workflow_instance_id": instance})
    def context() -> dict[str, Any]:
        return {"run_id": session.run_id, "frame_id": frame, "store_path": session.store.db_path, "control": session.control}
    hooks = ExecutionHooks(lambda label, payload: None, context, lambda name, args, meta: frame, lambda frame, result, error: None)
    status = "failed"
    try:
        if key in session.replay_cache:
            result = deepcopy(session.replay_cache[key])
        else:
            with execution_hooks(hooks):
                result = execute(*args, **kwargs)
        if result.get("status") == "paused":
            status, session.paused = "paused", True
            raise RuntimePauseRequested("workflow_instance", result.get("checkpoint_uid"), frame)
        status = "completed"
        with session.lock:
            session.results[key] = deepcopy(result)
            session.store.update_frame(frame, outputs=result)
            save_workflow_boundary(session, "workflow_instance_completed")
        return result
    finally:
        session.store.update_frame(frame, status=status)


def execute_workflow(session: WorkflowSnapshotSession, execute: Callable[[], Any], initial: str | None,
                     use_async: bool) -> Any:
    if not session.replaying:
        session.run_id = session.store.create_run("workflow", session.workflow.name)
        session.root_frame_id = session.store.create_frame(session.run_id, "workflow", session.workflow.name)
        session.store.set_run_root_frame(session.run_id, session.root_frame_id)
        session.results, session.replay_cache, session.last_snapshot_id = {}, {}, None
    session.initial_context, session.use_async, session.paused = initial, use_async, False
    replaying, session.replaying = session.replaying, False
    status = "failed"
    session.store.update_run_status(str(session.run_id), "running")
    try:
        with runtime_options(dataflow_workflows=True):
            if not replaying:
                save_workflow_boundary(session, "workflow_entry")
            result = execute()
            status = "paused" if session.paused else "completed" if len(result) == len(session.workflow._next_reachable_nodes) else "failed"
            save_workflow_boundary(session, "workflow_" + status)
            return result
    except RuntimePauseRequested:
        status = "paused"
        return session.workflow._results
    finally:
        session.store.update_frame(str(session.root_frame_id), status=status)
        session.store.update_run_status(str(session.run_id), status, ended=status in {"failed", "completed"})
