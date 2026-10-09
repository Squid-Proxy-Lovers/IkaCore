# pyright: strict

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from .snapshot_privacy import sanitized_state, snapshot_json
from .snapshot_schema import connect, decode_row
from .snapshot_schema import utc_now as _utc_now


def _validate_links(store: Any, run_id: str, snapshot_id: Optional[str], previous: Optional[str], origin: Optional[str]) -> None:
    if snapshot_id and store.load_snapshot(snapshot_id) is not None:
        raise ValueError("snapshot IDs are immutable and cannot be overwritten")
    if previous:
        snapshot = store.load_snapshot(previous)
        if snapshot is None or snapshot["run_id"] != run_id:
            raise ValueError("snapshot predecessor must exist in the same run")
    if origin and store.load_snapshot(origin) is None:
        raise ValueError("snapshot branch origin must exist")


def create_snapshot(
    self: Any,
    run_id: str,
    frame_id: str,
    snapshot_kind: str,
    state: dict[str, Any],
    *,
    resume_strategy: str = "exact",
    label: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    prev_snapshot_id: Optional[str] = None,
    branch_origin_snapshot_id: Optional[str] = None,
) -> str:
    frame = self.get_frame(frame_id)
    if not frame or frame["run_id"] != run_id:
        raise ValueError("snapshot frame must belong to its run")
    _validate_links(self, run_id, snapshot_id, prev_snapshot_id, branch_origin_snapshot_id)
    state = sanitized_state(state)
    snapshot_uid = snapshot_id or str(uuid4())
    with connect(self) as conn:
        conn.execute(
            """
            INSERT INTO snapshots (
                snapshot_id, run_id, frame_id, snapshot_kind, resume_strategy,
                label, state_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_uid,
                run_id,
                frame_id,
                snapshot_kind,
                resume_strategy,
                label,
                snapshot_json(state),
                _utc_now(),
            ),
        )
        if prev_snapshot_id is None:
            row = conn.execute(
                """
                SELECT snapshot_id FROM snapshots
                WHERE frame_id = ? AND snapshot_id != ?
                ORDER BY created_at DESC, snapshot_id DESC
                LIMIT 1
                """,
                (frame_id, snapshot_uid),
            ).fetchone()
            prev_snapshot_id = row["snapshot_id"] if row else None
        conn.execute(
            """
            INSERT OR REPLACE INTO snapshot_graph (
                snapshot_id, prev_snapshot_id, branch_origin_snapshot_id
            )
            VALUES (?, ?, ?)
            """,
            (
                snapshot_uid,
                prev_snapshot_id,
                branch_origin_snapshot_id,
            ),
        )
    return snapshot_uid


def load_snapshot(self: Any, snapshot_id: str) -> Optional[dict[str, Any]]:
    with connect(self) as conn:
        row = conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        graph_row = conn.execute(
            "SELECT prev_snapshot_id, branch_origin_snapshot_id FROM snapshot_graph WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
    if not row:
        return None
    out = decode_row(row)
    out["state_json"] = json.loads(out["state_json"])
    out["prev_snapshot_id"] = graph_row["prev_snapshot_id"] if graph_row else None
    out["branch_origin_snapshot_id"] = graph_row["branch_origin_snapshot_id"] if graph_row else None
    return out


def get_latest_snapshot_for_frame(self: Any, frame_id: str) -> Optional[dict[str, Any]]:
    with connect(self) as conn:
        row = conn.execute(
            """
            SELECT * FROM snapshots
            WHERE frame_id = ?
            ORDER BY created_at DESC, snapshot_id DESC
            LIMIT 1
            """,
            (frame_id,),
        ).fetchone()
        graph_row = conn.execute(
            "SELECT prev_snapshot_id, branch_origin_snapshot_id FROM snapshot_graph WHERE snapshot_id = ?",
            (row["snapshot_id"],),
        ).fetchone() if row else None
    if not row:
        return None
    out = decode_row(row)
    out["state_json"] = json.loads(out["state_json"])
    out["prev_snapshot_id"] = graph_row["prev_snapshot_id"] if graph_row else None
    out["branch_origin_snapshot_id"] = graph_row["branch_origin_snapshot_id"] if graph_row else None
    return out


def get_frame_entry_snapshot(self: Any, frame_id: str) -> Optional[dict[str, Any]]:
    with connect(self) as conn:
        row = conn.execute(
            """
            SELECT * FROM snapshots
            WHERE frame_id = ? AND snapshot_kind = 'frame_entry'
            ORDER BY created_at, snapshot_id
            LIMIT 1
            """,
            (frame_id,),
        ).fetchone()
        graph_row = conn.execute(
            "SELECT prev_snapshot_id, branch_origin_snapshot_id FROM snapshot_graph WHERE snapshot_id = ?",
            (row["snapshot_id"],),
        ).fetchone() if row else None
    if not row:
        return None
    out = decode_row(row)
    out["state_json"] = json.loads(out["state_json"])
    out["prev_snapshot_id"] = graph_row["prev_snapshot_id"] if graph_row else None
    out["branch_origin_snapshot_id"] = graph_row["branch_origin_snapshot_id"] if graph_row else None
    return out


def list_snapshots(self: Any, run_id: str, frame_id: Optional[str] = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM snapshots WHERE run_id = ?"
    params: list[Any] = [run_id]
    if frame_id is not None:
        query += " AND frame_id = ?"
        params.append(frame_id)
    query += " ORDER BY created_at, snapshot_id"
    with connect(self) as conn:
        rows = conn.execute(query, params).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = decode_row(row)
            item["state_json"] = json.loads(item["state_json"])
            graph_row = conn.execute(
                "SELECT prev_snapshot_id, branch_origin_snapshot_id FROM snapshot_graph WHERE snapshot_id = ?",
                (item["snapshot_id"],),
            ).fetchone()
            item["prev_snapshot_id"] = graph_row["prev_snapshot_id"] if graph_row else None
            item["branch_origin_snapshot_id"] = graph_row["branch_origin_snapshot_id"] if graph_row else None
            output.append(item)
    return output


def get_snapshot_chain(self: Any, snapshot_id: str) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    current_id = snapshot_id
    seen: set[str] = set()
    while current_id:
        if current_id in seen:
            raise ValueError("cyclic runtime lineage")
        seen.add(current_id)
        snapshot = self.load_snapshot(current_id)
        if not snapshot:
            break
        chain.append(snapshot)
        current_id = snapshot.get("prev_snapshot_id")
    chain.reverse()
    return chain
