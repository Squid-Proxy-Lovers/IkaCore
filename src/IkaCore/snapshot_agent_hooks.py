"""Agent, stage, and tool frames for opt-in runtime snapshots."""

# pyright: strict

from __future__ import annotations

from typing import Any, Optional

from IkaModel.base import AgentEndException, HumanInputRequired
from IkaModel.execution_hooks import ExecutionHooks, current_execution_context, current_tool_frame

from .runtime_control import RuntimePauseRequested
from .snapshot_session import SnapshotSession, record_boundary
from .snapshot_store import SnapshotStore


def ensure_session(agent: Any) -> Optional[SnapshotSession]:
    session: Optional[SnapshotSession] = getattr(agent, "_snapshot_runtime", None)
    parent = current_execution_context()
    if session is None and parent:
        session = SnapshotSession(agent, SnapshotStore(parent["store_path"]), parent["control"])
        agent._snapshot_runtime = session
    if session is None:
        return None
    if session.run_id is not None and not session.replaying:
        frame = session.store.get_frame(str(session.root_frame_id))
        if frame is not None and frame["status"] in {"completed", "cancelled", "failed"}:
            session.run_id, session.root_frame_id, session.stage_frame_id = None, None, None
            session.stage_index, session.last_snapshot_id, session.pending_response = None, None, None
            session.live_messages, session.skip_boundary, session.remaining_steps = None, None, None
    if session.run_id is None:
        if parent and parent.get("store_path") == session.store.db_path:
            run_id = str(parent["run_id"])
            parent_frame = parent["frame_id"]
        else:
            run_id = session.store.create_run("agent", agent.name)
            parent_frame = None
        session.run_id = run_id
        session.root_frame_id = session.store.create_frame(run_id, "agent", agent.name,
            parent_frame_id=parent_frame, caller_frame_id=parent_frame, return_to_frame_id=parent_frame,
            resolved_config={"model_id": agent.model_id, "maxsteps": agent.maxsteps, "use_async": agent.use_async})
        run = session.store.get_run(run_id)
        if run is not None and not run.get("root_frame_id"):
            session.store.set_run_root_frame(run_id, session.root_frame_id)
    return session


def start_stage(session: SnapshotSession, index: int, payload: dict[str, Any]) -> None:
    if session.stage_index == index and session.stage_frame_id is not None:
        return
    session.stage_index = index
    session.stage_frame_id = session.store.create_frame(str(session.run_id), "stage", str(payload.get("name", index)),
        parent_frame_id=session.root_frame_id, caller_frame_id=session.root_frame_id,
        return_to_frame_id=session.root_frame_id, position={"stage_index": index})
    record_boundary(session, "frame_entry", payload)


def tool_started(session: SnapshotSession, name: str, args: dict[str, Any], metadata: dict[str, Any]) -> str:
    if name in {"agent_end", "stage_end", "change_stage", "ask_user"}:
        metadata = {**metadata, "replay_policy": "allow", "side_effect_type": "control"}
    parent = current_tool_frame() or session.stage_frame_id or session.root_frame_id
    frame = session.store.create_frame(str(session.run_id), metadata.get("frame_type", "tool"), name,
        parent_frame_id=parent, caller_frame_id=parent, return_to_frame_id=parent, inputs=args, metadata=metadata)
    session.store.create_snapshot(str(session.run_id), frame, "frame_entry", {"tool_name": name, "arguments": args},
                                  resume_strategy="manual")
    return frame


def tool_finished(session: SnapshotSession, frame: str, result: object, error: Optional[BaseException]) -> None:
    status = "paused" if isinstance(error, (RuntimePauseRequested, HumanInputRequired)) else "completed" if error is None or isinstance(error, AgentEndException) else "failed"
    session.store.update_frame(frame, status=status, outputs={"result": result} if error is None else {})
    if error is None or isinstance(error, AgentEndException):
        info = session.store.get_frame(frame)
        metadata: dict[str, Any] = info["metadata_json"] if info is not None else {}
        payload = {"tool_name": info["frame_name"] if info is not None else "tool", **metadata}
        session.store.record_event(str(session.run_id), frame, "tool_completed", payload)
        if info is not None and info["parent_frame_id"]:
            session.store.record_event(str(session.run_id), info["parent_frame_id"], "tool_completed", payload)


def hooks_for_session(session: SnapshotSession) -> ExecutionHooks:
    def boundary(label: str, payload: dict[str, Any]) -> Optional[str]:
        if label == "stage_entry":
            start_stage(session, int(payload["stage_index"]), payload)
        return record_boundary(session, label, payload)

    def context() -> dict[str, Any]:
        return {"run_id": session.run_id, "frame_id": current_tool_frame() or session.stage_frame_id or session.root_frame_id,
                "store_path": session.store.db_path, "control": session.control}

    return ExecutionHooks(boundary, context, lambda name, args, meta: tool_started(session, name, args, meta),
                          lambda frame, result, error: tool_finished(session, frame, result, error))
