"""State and replay checks for explicitly enabled workflow snapshots."""

# pyright: strict

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from .runtime_control import RuntimeControl, RuntimePauseRequested
from .snapshot_store import SnapshotStore


def graph_fingerprint(workflow: Any) -> str:
    nodes = [(n.name, n.instances, n.instance_inputs, n.agent.model_id, n.agent.prompt,
              n.agent.system_prompt, n.agent.description, [(s.name, s.prompt) for s in cast(list[Any], n.agent.Stages or [])])
             for n in workflow.nodes]
    edges = [(e.source, e.target, e.edge_type, e.stage_index) for e in workflow.edges]
    return hashlib.sha256(json.dumps([nodes, edges, workflow.start_node], sort_keys=True).encode()).hexdigest()


@dataclass
class WorkflowSnapshotSession:
    workflow: Any
    store: SnapshotStore
    control: RuntimeControl
    run_id: Optional[str] = None
    root_frame_id: Optional[str] = None
    last_snapshot_id: Optional[str] = None
    initial_context: Optional[str] = None
    use_async: bool = False
    results: dict[str, dict[str, Any]] = field(default_factory=lambda: dict[str, dict[str, Any]]())
    replay_cache: dict[str, dict[str, Any]] = field(default_factory=lambda: dict[str, dict[str, Any]]())
    replaying: bool = False
    paused: bool = False
    lock: Any = field(default_factory=threading.RLock)


def save_workflow_boundary(session: WorkflowSnapshotSession, label: str) -> str:
    with session.lock:
        state = {"workflow_fingerprint": graph_fingerprint(session.workflow), "initial_context": session.initial_context,
                 "use_async": session.use_async, "instance_results": session.results}
        snapshot = session.store.create_snapshot(str(session.run_id), str(session.root_frame_id), label, state,
                    resume_strategy="workflow", prev_snapshot_id=session.last_snapshot_id)
        session.last_snapshot_id = snapshot
        if session.control.should_pause(label):
            session.paused = True
            raise RuntimePauseRequested(label, snapshot, session.root_frame_id)
        return snapshot


def replay_cache(session: WorkflowSnapshotSession, saved: dict[str, Any], allow_unsafe: bool) -> dict[str, dict[str, Any]]:
    workflow = session.workflow
    selected = session.control.parallel_replay
    invalidated: set[str] = set()
    for node_name, instances in selected.items():
        node = workflow._node_index.get(node_name)
        if node is None or any(type(i) is not int or not 0 <= i < node.instances for i in instances):
            raise ValueError("parallel replay must select existing node instance IDs")
        frontier = list(workflow._node_dependents[node_name])
        while frontier:
            name = frontier.pop()
            if name not in invalidated:
                invalidated.add(name)
                frontier.extend(workflow._node_dependents[name])
    cache: dict[str, dict[str, Any]] = {}
    for key, result in saved.items():
        node, _, instance = key.rpartition(":")
        rerun = (not session.control.reuse_historical_parallel_results or node in invalidated
                 or int(instance) in selected.get(node, []))
        if not rerun:
            cache[key] = result
    for frame in session.store.list_frames(str(session.run_id)):
        meta = frame["metadata_json"]
        node = meta.get("workflow_node_name")
        if node is not None and f"{node}:{meta['workflow_instance_id']}" not in cache:
            if not allow_unsafe and not session.store.assess_frame_replay(frame["frame_id"])["safe"]:
                raise ValueError("selected workflow instance contains non-replayable tools")
    return cache
