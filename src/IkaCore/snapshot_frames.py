# pyright: strict

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from .snapshot_privacy import snapshot_json
from .snapshot_schema import connect
from .snapshot_schema import utc_now as _utc_now


def create_frame(
    self: Any,
    run_id: str,
    frame_type: str,
    frame_name: str,
    *,
    status: str = "running",
    parent_frame_id: Optional[str] = None,
    caller_frame_id: Optional[str] = None,
    return_to_frame_id: Optional[str] = None,
    return_slot: Optional[str] = None,
    return_mode: Optional[str] = None,
    on_complete: Optional[str] = None,
    position: Optional[dict[str, Any]] = None,
    inputs: Optional[dict[str, Any]] = None,
    outputs: Optional[dict[str, Any]] = None,
    resolved_config: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    frame_id = str(uuid4())
    now = _utc_now()
    with connect(self) as conn:
        conn.execute(
            """
            INSERT INTO frames (
                frame_id, run_id, parent_frame_id, caller_frame_id,
                return_to_frame_id, frame_type, frame_name, status,
                return_slot, return_mode, on_complete,
                position_json, inputs_json, outputs_json,
                resolved_config_json, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_id,
                run_id,
                parent_frame_id,
                caller_frame_id,
                return_to_frame_id,
                frame_type,
                frame_name,
                status,
                return_slot,
                return_mode,
                on_complete,
                snapshot_json(position or {}),
                snapshot_json(inputs or {}),
                snapshot_json(outputs or {}),
                snapshot_json(resolved_config or {}),
                snapshot_json(metadata or {}),
                now,
                now,
            ),
        )
    return frame_id


def update_frame(
    self: Any,
    frame_id: str,
    *,
    status: Optional[str] = None,
    position: Optional[dict[str, Any]] = None,
    inputs: Optional[dict[str, Any]] = None,
    outputs: Optional[dict[str, Any]] = None,
    resolved_config: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    fields: list[str] = []
    params: list[Any] = []
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if position is not None:
        fields.append("position_json = ?")
        params.append(snapshot_json(position))
    if inputs is not None:
        fields.append("inputs_json = ?")
        params.append(snapshot_json(inputs))
    if outputs is not None:
        fields.append("outputs_json = ?")
        params.append(snapshot_json(outputs))
    if resolved_config is not None:
        fields.append("resolved_config_json = ?")
        params.append(snapshot_json(resolved_config))
    if metadata is not None:
        fields.append("metadata_json = ?")
        params.append(snapshot_json(metadata))
    fields.append("updated_at = ?")
    params.append(_utc_now())
    params.append(frame_id)
    with connect(self) as conn:
        conn.execute(
            f"UPDATE frames SET {', '.join(fields)} WHERE frame_id = ?",
            params,
        )


def get_frame(self: Any, frame_id: str) -> Optional[dict[str, Any]]:
    with connect(self) as conn:
        row = conn.execute("SELECT * FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()
    return self._decode_frame_row(row)


def list_frames(self: Any, run_id: str) -> list[dict[str, Any]]:
    with connect(self) as conn:
        rows = conn.execute(
            "SELECT * FROM frames WHERE run_id = ? ORDER BY created_at, frame_id",
            (run_id,),
        ).fetchall()
    return [self._decode_frame_row(row) for row in rows]


def get_frame_lineage(self: Any, frame_id: str) -> list[dict[str, Any]]:
    lineage: list[dict[str, Any]] = []
    current = self.get_frame(frame_id)
    seen: set[str] = set()
    while current:
        if current["frame_id"] in seen:
            raise ValueError("cyclic runtime frame lineage")
        seen.add(current["frame_id"])
        lineage.append(current)
        parent_id = current.get("parent_frame_id")
        current = self.get_frame(parent_id) if parent_id else None
    lineage.reverse()
    return lineage


def get_call_tree(self: Any, run_id: str) -> list[dict[str, Any]]:
    frames = self.list_frames(run_id)
    by_parent: dict[Optional[str], list[dict[str, Any]]] = {}
    for frame in frames:
        by_parent.setdefault(frame.get("parent_frame_id"), []).append(frame)

    def build(parent_id: Optional[str]) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        for frame in by_parent.get(parent_id, []):
            node = {
                "frame_id": frame["frame_id"],
                "frame_type": frame["frame_type"],
                "frame_name": frame["frame_name"],
                "status": frame["status"],
                "return_to_frame_id": frame["return_to_frame_id"],
                "children": build(frame["frame_id"]),
            }
            nodes.append(node)
        return nodes

    return build(None)


def list_workflow_node_instances(self: Any, run_id: str, node_name: str) -> list[dict[str, Any]]:
    frames = self.list_frames(run_id)
    matches: list[dict[str, Any]] = []
    for frame in frames:
        metadata = frame.get("metadata_json", {})
        if metadata.get("workflow_node_name") == node_name:
            matches.append(frame)
    return sorted(matches, key=lambda frame: int(frame.get("metadata_json", {}).get("workflow_instance_id") or 0))
