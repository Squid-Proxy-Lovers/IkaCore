# pyright: strict

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from .snapshot_privacy import snapshot_json
from .snapshot_schema import connect, decode_row
from .snapshot_schema import utc_now as _utc_now


def record_event(
    self: Any,
    run_id: str,
    frame_id: Optional[str],
    event_type: str,
    payload: Optional[dict[str, Any]] = None,
) -> str:
    event_id = str(uuid4())
    payload = payload or {}
    with connect(self) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) + 1 AS next_seq FROM events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        next_seq = int(row["next_seq"]) if row else 1
        conn.execute(
            """
            INSERT INTO events (
                event_id, run_id, frame_id, event_type, event_seq, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                run_id,
                frame_id,
                event_type,
                next_seq,
                snapshot_json(payload),
                _utc_now(),
            ),
        )
    return event_id


def list_events(self: Any, run_id: str) -> list[dict[str, Any]]:
    with connect(self) as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY event_seq",
            (run_id,),
        ).fetchall()
    output: list[dict[str, Any]] = []
    for row in rows:
        item = decode_row(row)
        item["payload_json"] = json.loads(item["payload_json"])
        output.append(item)
    return output


def list_frame_events(self: Any, frame_id: str) -> list[dict[str, Any]]:
    with connect(self) as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE frame_id = ? ORDER BY event_seq",
            (frame_id,),
        ).fetchall()
    output: list[dict[str, Any]] = []
    for row in rows:
        item = decode_row(row)
        item["payload_json"] = json.loads(item["payload_json"])
        output.append(item)
    return output


def _descendant_frames(store: Any, frame: dict[str, Any]) -> list[dict[str, Any]]:
    frames = store.list_frames(frame["run_id"])
    selected = {frame["frame_id"]}
    while True:
        expanded = selected | {item["frame_id"] for item in frames if item["parent_frame_id"] in selected}
        if expanded == selected:
            return [item for item in frames if item["frame_id"] in selected]
        selected = expanded


def assess_frame_replay(self: Any, frame_id: str) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    frame = self.get_frame(frame_id)
    if frame is None:
        raise ValueError("runtime frame not found")
    descendants = _descendant_frames(self, frame)
    frame_ids = {item["frame_id"] for item in descendants}
    tool_events = [event for event in self.list_events(frame["run_id"])
                   if event["frame_id"] in frame_ids and event["event_type"] == "tool_completed"]
    for event in tool_events:
        payload = event["payload_json"]
        replay_policy = payload.get("replay_policy", "deny")
        side_effect_type = payload.get("side_effect_type", "unknown")
        if replay_policy != "allow":
            blockers.append(
                {
                    "tool_name": payload.get("tool_name"),
                    "side_effect_type": side_effect_type,
                    "replay_policy": replay_policy,
                    "reason": "Tool execution is marked non-replayable.",
                }
            )
    for child in descendants:
        if (child["frame_type"] == "tool" and child["status"] != "completed"
                and child["metadata_json"].get("replay_policy", "deny") != "allow"):
            blockers.append({"tool_name": child["frame_name"], "side_effect_type": "unknown",
                             "replay_policy": "deny", "reason": "Tool may have partially completed."})
    return {
        "frame_id": frame_id,
        "safe": len(blockers) == 0,
        "blockers": blockers,
        "tool_event_count": len(tool_events),
    }


def add_artifact(
    self: Any,
    run_id: str,
    artifact_type: str,
    payload: dict[str, Any],
    *,
    frame_id: Optional[str] = None,
) -> str:
    artifact_id = str(uuid4())
    with connect(self) as conn:
        conn.execute(
            """
            INSERT INTO artifacts (
                artifact_id, run_id, frame_id, artifact_type, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                run_id,
                frame_id,
                artifact_type,
                snapshot_json(payload),
                _utc_now(),
            ),
        )
    return artifact_id
