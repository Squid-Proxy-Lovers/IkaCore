"""Opt-in snapshot APIs without replacing existing checkpoint entry points."""

# pyright: strict

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from .runtime_control import RuntimeControl
from .snapshot_session import SnapshotSession
from .snapshot_store import SnapshotStore


class AgentSnapshotControlMixin:
    _snapshot_runtime: Optional[SnapshotSession] = None

    def enable_snapshots(self: Any, db_path: str = "runtime-data/snapshots.db",
                         control: Optional[RuntimeControl] = None) -> SnapshotStore:
        store = SnapshotStore(db_path)
        self._snapshot_runtime = SnapshotSession(self, store, control or RuntimeControl())
        return store

    def configure_runtime_control(self: Any, *, pause_points: Optional[set[str]] = None,
                                  parallel_replay: Optional[dict[str, list[int]]] = None,
                                  reuse_historical_parallel_results: bool = True) -> RuntimeControl:
        if self._snapshot_runtime is None:
            self.enable_snapshots()
        session = self._snapshot_runtime
        if session is None:
            raise ValueError("snapshot runtime could not be enabled")
        control = session.control
        if pause_points is not None:
            control.pause_points = set(pause_points)
        if parallel_replay is not None:
            control.parallel_replay = deepcopy(parallel_replay)
        control.reuse_historical_parallel_results = reuse_historical_parallel_results
        return control

    def request_pause_at(self: Any, *checkpoint_kinds: str) -> None:
        self.configure_runtime_control().request_pause(*checkpoint_kinds)

    def clear_pause_requests(self: Any, *checkpoint_kinds: str) -> None:
        if self._snapshot_runtime is not None:
            self._snapshot_runtime.control.clear_pause(*checkpoint_kinds)


class AgentSnapshotReplayMixin(AgentSnapshotControlMixin):
    def resume_from_snapshot(self: Any, snapshot_id: str, *, allow_unsafe_replay: bool = False,
                             resume_input: Optional[str] = None) -> dict[str, Any]:
        session: Optional[SnapshotSession] = self._snapshot_runtime
        if session is None:
            raise ValueError("enable_snapshots must be called before snapshot replay")
        info = session.store.resume_from_snapshot(snapshot_id)
        state = info["state"]
        if "completed_output" in state:
            return {**state["completed_output"], "runtime": {"run_id": info["run_id"], "frame_id": info["frame_id"], "last_snapshot_id": snapshot_id}}
        if state.get("tool_checkpoint") and not allow_unsafe_replay:
            raise ValueError("mid-tool snapshots require an explicit unsafe-replay decision")
        self._resume_checkpoint = deepcopy(state)
        self.message_history = deepcopy(state.get("message_history", self.message_history))
        self._total_usage = deepcopy(state.get("total_usage", self._total_usage))
        self._total_cost = deepcopy(state.get("total_cost", self._total_cost))
        session.run_id, session.root_frame_id = info["run_id"], state.get("root_frame_id", info["frame_id"])
        session.stage_index = state.get("stage_index")
        stages = [frame for frame in session.store.get_frame_lineage(info["frame_id"]) if frame["frame_type"] == "stage"]
        session.stage_frame_id = stages[-1]["frame_id"] if stages else None
        session.last_snapshot_id, session.skip_boundary = snapshot_id, state.get("runtime_boundary")
        if session.skip_boundary == "frame_entry":
            session.skip_boundary = None
        session.pending_response = state.get("pending_response")
        session.live_messages = state.get("chat_messages")
        self._snapshot_resume_messages = deepcopy(session.live_messages)
        session.replaying = True
        return self.execution(resume_input=resume_input)

    def fork_from_snapshot(self: Any, snapshot_id: str, *, allow_unsafe_replay: bool = False) -> dict[str, Any]:
        if self._snapshot_runtime is None:
            raise ValueError("enable_snapshots must be called before snapshot replay")
        fork = self._snapshot_runtime.store.fork_run_from_snapshot(snapshot_id, allow_unsafe_replay=allow_unsafe_replay)
        return self.resume_from_snapshot(fork["snapshot_id"], allow_unsafe_replay=allow_unsafe_replay)

    def restart_from_frame(self: Any, frame_id: str, *, allow_unsafe_replay: bool = False) -> dict[str, Any]:
        if self._snapshot_runtime is None:
            raise ValueError("enable_snapshots must be called before snapshot replay")
        replay = self._snapshot_runtime.store.restart_frame(frame_id, allow_unsafe_replay=allow_unsafe_replay)
        return self.resume_from_snapshot(replay["snapshot_id"], allow_unsafe_replay=allow_unsafe_replay)
