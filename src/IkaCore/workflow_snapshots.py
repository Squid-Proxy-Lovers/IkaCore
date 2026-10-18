"""Explicit workflow snapshot and instance-replay entry points."""

# pyright: strict

from typing import Any, Optional

from .runtime_control import RuntimeControl
from .snapshot_store import SnapshotStore
from .workflow_snapshot_execution import execute_workflow
from .workflow_snapshot_state import (
    INTERRUPTED_STATUSES,
    WorkflowSnapshotSession,
    graph_fingerprint,
    interrupted_snapshot,
    replay_cache,
)


def _resume_inputs(value: str | dict[str, str] | None, interrupted: dict[str, str]) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        if len(interrupted) != 1:
            raise ValueError("resume_input requires one interrupted instance; use a mapping of node:instance to answer")
        return {next(iter(interrupted)): value}
    if not set(value) <= interrupted.keys():
        raise ValueError("resume_input keys must identify interrupted instances as node:instance")
    return dict(value)


def _reuse_resumed_children(session: WorkflowSnapshotSession) -> None:
    for key, snapshot_id in list(session.interrupted_instances.items()):
        snapshot = interrupted_snapshot(session, snapshot_id)
        root = snapshot["state_json"]["root_frame_id"]
        frame = session.store.get_frame(root)
        if frame is not None and frame["status"] == "completed" and frame["outputs_json"].get("status") not in INTERRUPTED_STATUSES:
            session.replay_cache[key] = frame["outputs_json"]
            del session.interrupted_instances[key]
        elif frame is not None and frame["status"] == "paused":
            snapshots = session.store.list_snapshots(str(session.run_id))
            latest = [s for s in snapshots if s["state_json"].get("root_frame_id") == root]
            if latest:
                session.interrupted_instances[key] = latest[-1]["snapshot_id"]


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
                "last_snapshot_id": session.last_snapshot_id, "paused": session.paused,
                "interrupted_instances": dict(session.interrupted_instances)}

    def resume_from_snapshot(self: Any, snapshot_id: str, *, allow_unsafe_replay: bool = False,
                             resume_input: str | dict[str, str] | None = None) -> Any:
        session: Optional[WorkflowSnapshotSession] = self._workflow_snapshot
        if session is None:
            raise ValueError("enable_snapshots must be called before workflow replay")
        info = session.store.resume_from_snapshot(snapshot_id)
        state = info["state"]
        if state.get("workflow_fingerprint") != graph_fingerprint(self):
            raise ValueError("snapshot belongs to a different workflow configuration")
        session.run_id, session.root_frame_id = info["run_id"], info["frame_id"]
        session.interrupted_instances = dict(state.get("interrupted_instances", {}))
        session.resume_inputs = _resume_inputs(resume_input, session.interrupted_instances)
        session.allow_unsafe_replay = allow_unsafe_replay
        session.replay_cache = replay_cache(session, state["instance_results"], allow_unsafe_replay)
        _reuse_resumed_children(session)
        session.results = dict(session.replay_cache)
        session.last_snapshot_id, session.replaying = snapshot_id, True
        return self.run(state["initial_context"], parallel=state["use_async"])

    def fork_from_snapshot(self: Any, snapshot_id: str, *, allow_unsafe_replay: bool = False,
                           resume_input: str | dict[str, str] | None = None) -> Any:
        session: Optional[WorkflowSnapshotSession] = self._workflow_snapshot
        if session is None:
            raise ValueError("enable_snapshots must be called before workflow replay")
        fork = session.store.fork_run_from_snapshot(snapshot_id, allow_unsafe_replay=allow_unsafe_replay)
        return self.resume_from_snapshot(fork["snapshot_id"], allow_unsafe_replay=allow_unsafe_replay,
                                         resume_input=resume_input)

    def _snapshot_run(self: Any, execute: Any, initial: Optional[str], asynchronous: bool) -> Any:
        return execute_workflow(self._workflow_snapshot, execute, initial, asynchronous)
