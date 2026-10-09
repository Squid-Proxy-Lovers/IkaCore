"""Per-agent snapshot state, separate from legacy checkpoint configuration."""

# pyright: strict

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from IkaModel.execution_hooks import current_tool_frame

from .runtime_control import RuntimeControl, RuntimePauseRequested
from .snapshot_store import SnapshotStore


@dataclass
class SnapshotSession:
    agent: Any
    store: SnapshotStore
    control: RuntimeControl
    run_id: Optional[str] = None
    root_frame_id: Optional[str] = None
    stage_frame_id: Optional[str] = None
    stage_index: Optional[int] = None
    last_snapshot_id: Optional[str] = None
    remaining_steps: Optional[int] = None
    last_content: str = ""
    live_messages: Optional[list[dict[str, Any]]] = None
    pending_response: Optional[dict[str, Any]] = None
    skip_boundary: Optional[str] = None
    replaying: bool = False
    lock: Any = field(default_factory=threading.RLock)


def capture_state(session: SnapshotSession, label: str, payload: dict[str, Any]) -> dict[str, Any]:
    remaining = session.remaining_steps if session.remaining_steps is not None else session.agent.maxsteps
    if session.stage_index is None:
        state = session.agent._agent_checkpoint_payload(remaining, session.last_content)
    else:
        state = session.agent._stage_checkpoint_payload(session.stage_index, remaining, session.last_content, "stage", None)
    state = dict(state)
    if session.live_messages is not None:
        state["chat_messages"] = deepcopy(session.live_messages)
    state.update({"runtime_boundary": label, "run_id": session.run_id, "root_frame_id": session.root_frame_id,
                  "frame_id": current_tool_frame() or session.stage_frame_id or session.root_frame_id,
                  "total_usage": deepcopy(session.agent._total_usage), "total_cost": deepcopy(session.agent._total_cost)})
    if label == "agent_completed":
        state["completed_output"] = deepcopy(payload["output"])
    if label == "human_input":
        state.update({"scope": "hitl", "interrupt_status": payload["status"],
                      "interrupt_kind": "hitl", "interrupt_data": deepcopy(payload.get("interrupt_data", {}))})
    if label == "post_model":
        state["pending_response"] = deepcopy(payload)
    if current_tool_frame() is not None:
        state["tool_checkpoint"] = True
        state["tool_payload"] = deepcopy(payload)
    return state


def record_boundary(session: SnapshotSession, label: str, payload: dict[str, Any]) -> Optional[str]:
    with session.lock:
        messages = payload.get("messages")
        if isinstance(messages, list):
            session.live_messages = deepcopy(cast(list[dict[str, Any]], messages))
        if "remaining_steps" in payload:
            session.remaining_steps = int(payload["remaining_steps"])
        if "last_content" in payload:
            session.last_content = str(payload["last_content"])
        frame = current_tool_frame() or session.stage_frame_id or session.root_frame_id
        if session.run_id is None or frame is None:
            return None
        state = capture_state(session, label, payload)
        snapshot = session.store.create_snapshot(session.run_id, frame, label, state,
                    resume_strategy="manual" if current_tool_frame() else "exact", prev_snapshot_id=session.last_snapshot_id)
        session.last_snapshot_id = snapshot
        session.store.record_event(session.run_id, frame, "snapshot_created", {"snapshot_id": snapshot, "boundary": label})
        if session.skip_boundary == label:
            session.skip_boundary = None
        elif session.control.should_pause(label):
            session.store.update_frame(frame, status="paused")
            session.store.update_run_status(session.run_id, "paused")
            raise RuntimePauseRequested(label, snapshot, frame)
        return snapshot
