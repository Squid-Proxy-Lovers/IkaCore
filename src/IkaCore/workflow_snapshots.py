"""Explicit workflow snapshot and instance-replay entry points."""

# pyright: strict

from typing import Any, Optional

from .runtime_control import RuntimeControl
from .snapshot_store import SnapshotStore
from .workflow_snapshot_execution import execute_workflow
from .workflow_snapshot_state import WorkflowSnapshotSession, graph_fingerprint, replay_cache


class WorkflowSnapshotMixin:
    _workflow_snapshot: Optional[WorkflowSnapshotSession] = None

    def enable_snapshots(self: Any, db_path: str = "runtime-data/snapshots.db",
                         control: Optional[RuntimeControl] = None) -> SnapshotStore:
        store = SnapshotStore(db_path)
        self._workflow_snapshot = WorkflowSnapshotSession(self, store, control or RuntimeControl())
        return store

    def snapshot_runtime_info(self: Any) -> dict[str, Any]:
        session: Optional[WorkflowSnapshotSession] = self._workflow_snapshot
        if session is None:
            return {}
        return {"run_id": session.run_id, "frame_id": session.root_frame_id,
                "last_snapshot_id": session.last_snapshot_id, "paused": session.paused}

    def resume_from_snapshot(self: Any, snapshot_id: str, *, allow_unsafe_replay: bool = False) -> Any:
        session: Optional[WorkflowSnapshotSession] = self._workflow_snapshot
        if session is None:
            raise ValueError("enable_snapshots must be called before workflow replay")
        info = session.store.resume_from_snapshot(snapshot_id)
        state = info["state"]
        if state.get("workflow_fingerprint") != graph_fingerprint(self):
            raise ValueError("snapshot belongs to a different workflow configuration")
        session.run_id, session.root_frame_id = info["run_id"], info["frame_id"]
        session.results = state["instance_results"]
        session.replay_cache = replay_cache(session, session.results, allow_unsafe_replay)
        session.last_snapshot_id, session.replaying = snapshot_id, True
        return self.run(state["initial_context"], use_async=state["use_async"])

    def _snapshot_run(self: Any, execute: Any, initial: Optional[str], asynchronous: bool) -> Any:
        return execute_workflow(self._workflow_snapshot, execute, initial, asynchronous)
