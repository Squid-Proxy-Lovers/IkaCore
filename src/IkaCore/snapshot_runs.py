# pyright: strict

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from .snapshot_privacy import snapshot_json
from .snapshot_schema import connect, decode_row
from .snapshot_schema import utc_now as _utc_now


def create_run(
    self: Any,
    entry_type: str,
    entry_name: str,
    status: str = "running",
    metadata: Optional[dict[str, Any]] = None,
    branch_from_snapshot_id: Optional[str] = None,
) -> str:
    run_id = str(uuid4())
    with connect(self) as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, root_frame_id, status, entry_type, entry_name,
                branch_from_snapshot_id, metadata_json, started_at, ended_at
            )
            VALUES (?, NULL, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                run_id,
                status,
                entry_type,
                entry_name,
                branch_from_snapshot_id,
                snapshot_json(metadata or {}),
                _utc_now(),
            ),
        )
    return run_id


def set_run_root_frame(self: Any, run_id: str, root_frame_id: str) -> None:
    with connect(self) as conn:
        conn.execute(
            "UPDATE runs SET root_frame_id = ? WHERE run_id = ?",
            (root_frame_id, run_id),
        )


def update_run_status(self: Any, run_id: str, status: str, ended: bool = False) -> None:
    ended_at = _utc_now() if ended else None
    with connect(self) as conn:
        if ended:
            conn.execute(
                "UPDATE runs SET status = ?, ended_at = ? WHERE run_id = ?",
                (status, ended_at, run_id),
            )
        else:
            conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?",
                (status, run_id),
            )


def get_run(self: Any, run_id: str) -> Optional[dict[str, Any]]:
    with connect(self) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        return None
    out = decode_row(row)
    out["metadata_json"] = json.loads(out["metadata_json"])
    return out
