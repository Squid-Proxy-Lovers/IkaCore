"""Additive runtime-snapshot schema and scoped SQLite connections."""

# pyright: strict

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA = ('\n                CREATE TABLE IF NOT EXISTS runs (\n                    run_id TEXT PRIMARY KEY,\n                    root_frame_id TEXT,\n                    status TEXT NOT NULL,\n                    entry_type TEXT NOT NULL,\n                    entry_name TEXT NOT NULL,\n                    branch_from_snapshot_id TEXT,\n                    metadata_json TEXT NOT NULL,\n                    started_at DATETIME NOT NULL,\n                    ended_at DATETIME\n                )\n                ', '\n                CREATE TABLE IF NOT EXISTS frames (\n                    frame_id TEXT PRIMARY KEY,\n                    run_id TEXT NOT NULL,\n                    parent_frame_id TEXT,\n                    caller_frame_id TEXT,\n                    return_to_frame_id TEXT,\n                    frame_type TEXT NOT NULL,\n                    frame_name TEXT NOT NULL,\n                    status TEXT NOT NULL,\n                    return_slot TEXT,\n                    return_mode TEXT,\n                    on_complete TEXT,\n                    position_json TEXT NOT NULL,\n                    inputs_json TEXT NOT NULL,\n                    outputs_json TEXT NOT NULL,\n                    resolved_config_json TEXT NOT NULL,\n                    metadata_json TEXT NOT NULL,\n                    created_at DATETIME NOT NULL,\n                    updated_at DATETIME NOT NULL,\n                    FOREIGN KEY(run_id) REFERENCES runs(run_id)\n                )\n                ', '\n                CREATE TABLE IF NOT EXISTS snapshots (\n                    snapshot_id TEXT PRIMARY KEY,\n                    run_id TEXT NOT NULL,\n                    frame_id TEXT NOT NULL,\n                    snapshot_kind TEXT NOT NULL,\n                    resume_strategy TEXT NOT NULL,\n                    label TEXT,\n                    state_json TEXT NOT NULL,\n                    created_at DATETIME NOT NULL,\n                    FOREIGN KEY(run_id) REFERENCES runs(run_id),\n                    FOREIGN KEY(frame_id) REFERENCES frames(frame_id)\n                )\n                ', '\n                CREATE TABLE IF NOT EXISTS events (\n                    event_id TEXT PRIMARY KEY,\n                    run_id TEXT NOT NULL,\n                    frame_id TEXT,\n                    event_type TEXT NOT NULL,\n                    event_seq INTEGER NOT NULL,\n                    payload_json TEXT NOT NULL,\n                    created_at DATETIME NOT NULL,\n                    FOREIGN KEY(run_id) REFERENCES runs(run_id)\n                )\n                ', '\n                CREATE TABLE IF NOT EXISTS artifacts (\n                    artifact_id TEXT PRIMARY KEY,\n                    run_id TEXT NOT NULL,\n                    frame_id TEXT,\n                    artifact_type TEXT NOT NULL,\n                    payload_json TEXT NOT NULL,\n                    created_at DATETIME NOT NULL,\n                    FOREIGN KEY(run_id) REFERENCES runs(run_id)\n                )\n                ', '\n                CREATE TABLE IF NOT EXISTS snapshot_graph (\n                    snapshot_id TEXT PRIMARY KEY,\n                    prev_snapshot_id TEXT,\n                    branch_origin_snapshot_id TEXT,\n                    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)\n                )\n                ', 'CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)', 'CREATE INDEX IF NOT EXISTS idx_frames_run_id ON frames(run_id)', 'CREATE INDEX IF NOT EXISTS idx_frames_parent_frame_id ON frames(parent_frame_id)', 'CREATE INDEX IF NOT EXISTS idx_frames_return_to_frame_id ON frames(return_to_frame_id)', 'CREATE INDEX IF NOT EXISTS idx_snapshots_run_id ON snapshots(run_id)', 'CREATE INDEX IF NOT EXISTS idx_snapshots_frame_id ON snapshots(frame_id)', 'CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)', 'CREATE INDEX IF NOT EXISTS idx_snapshot_graph_prev ON snapshot_graph(prev_snapshot_id)')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(store: Any) -> Generator[sqlite3.Connection, None, None]:
    connection = sqlite3.connect(store.db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_schema(store: Any) -> None:
    with connect(store) as connection:
        for statement in SCHEMA:
            connection.execute(statement)


def decode_frame_row(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if not row:
            return None
        out = decode_row(row)
        for key in ("position_json", "inputs_json", "outputs_json", "resolved_config_json", "metadata_json"):
            out[key] = json.loads(out[key])
        return out


def decode_row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}
