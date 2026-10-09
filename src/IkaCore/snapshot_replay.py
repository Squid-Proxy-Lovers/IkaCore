# pyright: strict

from __future__ import annotations

import json
from typing import Any, Optional


def resume_from_snapshot(self: Any, snapshot_id: str, *, allow_redacted_state: bool = False) -> dict[str, Any]:
    snapshot = self.load_snapshot(snapshot_id)
    if not snapshot:
        raise ValueError(f"Snapshot '{snapshot_id}' not found")
    state = dict(snapshot["state_json"])
    if state.get("_redacted_fields") and not allow_redacted_state:
        raise ValueError("snapshot contains redacted credential fields; restore them explicitly before replay")
    state["_snapshot_meta"] = {
        "snapshot_id": snapshot["snapshot_id"],
        "run_id": snapshot["run_id"],
        "frame_id": snapshot["frame_id"],
        "snapshot_kind": snapshot["snapshot_kind"],
        "resume_strategy": snapshot["resume_strategy"],
        "created_at": snapshot["created_at"],
    }
    self.record_event(
        snapshot["run_id"],
        snapshot["frame_id"],
        "snapshot_resumed",
        {"snapshot_id": snapshot_id},
    )
    return {
        "run_id": snapshot["run_id"],
        "frame_id": snapshot["frame_id"],
        "snapshot_id": snapshot_id,
        "state": state,
    }


def _clone_lineage(self: Any, lineage: list[dict[str, Any]], new_run_id: str) -> dict[str, str]:
    frame_id_map: dict[str, str] = {}
    for index, frame in enumerate(lineage):
        new_status = "completed" if index < len(lineage) - 1 else "pending"
        new_frame_id = self.create_frame(
            run_id=new_run_id,
            frame_type=frame["frame_type"],
            frame_name=frame["frame_name"],
            status=new_status,
            parent_frame_id=frame_id_map.get(frame.get("parent_frame_id") or ""),
            caller_frame_id=frame_id_map.get(frame.get("caller_frame_id") or ""),
            return_to_frame_id=frame_id_map.get(frame.get("return_to_frame_id") or ""),
            return_slot=frame.get("return_slot"),
            return_mode=frame.get("return_mode"),
            on_complete=frame.get("on_complete"),
            position=frame["position_json"],
            inputs=frame["inputs_json"],
            outputs=frame["outputs_json"] if index < len(lineage) - 1 else {},
            resolved_config=frame["resolved_config_json"],
            metadata={
                **frame["metadata_json"],
                "cloned_from_frame_id": frame["frame_id"],
            },
        )
        frame_id_map[frame["frame_id"]] = new_frame_id

    return frame_id_map


def _remap_state(state: dict[str, Any], mapping: dict[str, str], run_id: str) -> dict[str, Any]:
    cloned = json.loads(json.dumps(state))
    cloned["run_id"] = run_id
    for key in ("frame_id", "root_frame_id", "parent_frame_id", "return_to_frame_id"):
        if cloned.get(key) in mapping:
            cloned[key] = mapping[cloned[key]]
    return cloned


def _clone_entry_snapshots(store: Any, lineage: list[dict[str, Any]], mapping: dict[str, str], run_id: str) -> None:
    for frame in lineage:
        entry = store.get_frame_entry_snapshot(frame["frame_id"])
        if entry is not None:
            store.create_snapshot(run_id, mapping[frame["frame_id"]], "frame_entry",
                _remap_state(entry["state_json"], mapping, run_id), resume_strategy=entry["resume_strategy"],
                branch_origin_snapshot_id=entry["snapshot_id"])


def _finish_fork(self: Any, snapshot_id: str, snapshot: dict[str, Any], target_frame: dict[str, Any],
                 new_run_id: str, new_root_frame_id: str, frame_id_map: dict[str, str],
                 cloned_state: dict[str, Any], replay_assessment: dict[str, Any],
                 allow_unsafe_replay: bool) -> dict[str, Any]:
    new_snapshot_id = self.create_snapshot(
        new_run_id,
        frame_id_map[target_frame["frame_id"]],
        snapshot_kind="fork_replay",
        state=cloned_state,
        resume_strategy=snapshot["resume_strategy"],
        label=f"forked-from:{snapshot_id}",
        branch_origin_snapshot_id=snapshot_id,
    )
    self.record_event(
        new_run_id,
        frame_id_map[target_frame["frame_id"]],
        "run_forked",
        {
            "source_run_id": snapshot["run_id"],
            "source_snapshot_id": snapshot_id,
            "source_frame_id": target_frame["frame_id"],
            "allow_unsafe_replay": allow_unsafe_replay,
        },
    )
    return {
        "run_id": new_run_id,
        "root_frame_id": new_root_frame_id,
        "frame_id": frame_id_map[target_frame["frame_id"]],
        "snapshot_id": new_snapshot_id,
        "state": self.resume_from_snapshot(new_snapshot_id)["state"],
        "replay_assessment": replay_assessment,
    }


def fork_run_from_snapshot(
    self: Any,
    snapshot_id: str,
    *,
    allow_unsafe_replay: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    snapshot = self.load_snapshot(snapshot_id)
    if not snapshot:
        raise ValueError(f"Snapshot '{snapshot_id}' not found")

    target_frame = self.get_frame(snapshot["frame_id"])
    if not target_frame:
        raise ValueError(f"Frame '{snapshot['frame_id']}' not found")

    replay_assessment = self.assess_frame_replay(target_frame["frame_id"])
    if not replay_assessment["safe"] and not allow_unsafe_replay:
        raise ValueError(
            f"Frame '{target_frame['frame_id']}' contains non-replayable tool calls: "
            + ", ".join(blocker["tool_name"] or "<unknown>" for blocker in replay_assessment["blockers"])
        )

    source_run = self.get_run(snapshot["run_id"])
    lineage = self.get_frame_lineage(target_frame["frame_id"])
    fork_metadata = dict(metadata or {})
    fork_metadata.update({
        "forked_from_run_id": snapshot["run_id"],
        "forked_from_snapshot_id": snapshot_id,
        "forked_from_frame_id": target_frame["frame_id"],
    })
    new_run_id = self.create_run(
        entry_type=source_run["entry_type"] if source_run else "fork",
        entry_name=source_run["entry_name"] if source_run else target_frame["frame_name"],
        status="paused",
        metadata=fork_metadata,
        branch_from_snapshot_id=snapshot_id,
    )

    frame_id_map = _clone_lineage(self, lineage, new_run_id)
    root_source_id = lineage[0]["frame_id"]
    new_root_frame_id = frame_id_map[root_source_id]
    self.set_run_root_frame(new_run_id, new_root_frame_id)

    _clone_entry_snapshots(self, lineage, frame_id_map, new_run_id)
    cloned_state = _remap_state(snapshot["state_json"], frame_id_map, new_run_id)
    cloned_state["frame_id"] = frame_id_map[target_frame["frame_id"]]

    return _finish_fork(self, snapshot_id, snapshot, target_frame, new_run_id, new_root_frame_id,
                        frame_id_map, cloned_state, replay_assessment, allow_unsafe_replay)


def restart_frame(self: Any, frame_id: str, *, allow_unsafe_replay: bool = False) -> dict[str, Any]:
    entry_snapshot = self.get_frame_entry_snapshot(frame_id)
    if not entry_snapshot:
        raise ValueError(f"Frame '{frame_id}' does not have a frame_entry snapshot")
    return self.fork_run_from_snapshot(
        entry_snapshot["snapshot_id"],
        allow_unsafe_replay=allow_unsafe_replay,
        metadata={"restart_from_frame_id": frame_id},
    )
