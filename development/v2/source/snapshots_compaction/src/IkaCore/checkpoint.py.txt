import json
import sqlite3
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointStore:
    """
    Runtime-backed snapshot store.

    The legacy checkpoint API remains available, but checkpoints now map to
    snapshots attached to frames within a run. This gives the runtime enough
    structure to reconstruct call trees and resumption boundaries.
    """

    def __init__(self, db_path: str = "checkpoints.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    root_frame_id TEXT,
                    status TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    entry_name TEXT NOT NULL,
                    branch_from_snapshot_id TEXT,
                    metadata_json TEXT NOT NULL,
                    started_at DATETIME NOT NULL,
                    ended_at DATETIME
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS frames (
                    frame_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    parent_frame_id TEXT,
                    caller_frame_id TEXT,
                    return_to_frame_id TEXT,
                    frame_type TEXT NOT NULL,
                    frame_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    return_slot TEXT,
                    return_mode TEXT,
                    on_complete TEXT,
                    position_json TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    resolved_config_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    frame_id TEXT NOT NULL,
                    snapshot_kind TEXT NOT NULL,
                    resume_strategy TEXT NOT NULL,
                    label TEXT,
                    state_json TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id),
                    FOREIGN KEY(frame_id) REFERENCES frames(frame_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    frame_id TEXT,
                    event_type TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    frame_id TEXT,
                    artifact_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshot_graph (
                    snapshot_id TEXT PRIMARY KEY,
                    prev_snapshot_id TEXT,
                    branch_origin_snapshot_id TEXT,
                    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_run_id ON frames(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_parent_frame_id ON frames(parent_frame_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_return_to_frame_id ON frames(return_to_frame_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_run_id ON snapshots(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_frame_id ON snapshots(frame_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_graph_prev ON snapshot_graph(prev_snapshot_id)")

    def create_run(
        self,
        entry_type: str,
        entry_name: str,
        status: str = "running",
        metadata: Optional[dict[str, Any]] = None,
        branch_from_snapshot_id: Optional[str] = None,
    ) -> str:
        run_id = str(uuid4())
        with self._connect() as conn:
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
                    json.dumps(metadata or {}),
                    _utc_now(),
                ),
            )
        return run_id

    def set_run_root_frame(self, run_id: str, root_frame_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET root_frame_id = ? WHERE run_id = ?",
                (root_frame_id, run_id),
            )

    def update_run_status(self, run_id: str, status: str, ended: bool = False) -> None:
        ended_at = _utc_now() if ended else None
        with self._connect() as conn:
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

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["metadata_json"] = json.loads(out["metadata_json"])
        return out

    def create_frame(
        self,
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
        with self._connect() as conn:
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
                    json.dumps(position or {}),
                    json.dumps(inputs or {}),
                    json.dumps(outputs or {}),
                    json.dumps(resolved_config or {}),
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )
        return frame_id

    def update_frame(
        self,
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
            params.append(json.dumps(position))
        if inputs is not None:
            fields.append("inputs_json = ?")
            params.append(json.dumps(inputs))
        if outputs is not None:
            fields.append("outputs_json = ?")
            params.append(json.dumps(outputs))
        if resolved_config is not None:
            fields.append("resolved_config_json = ?")
            params.append(json.dumps(resolved_config))
        if metadata is not None:
            fields.append("metadata_json = ?")
            params.append(json.dumps(metadata))
        fields.append("updated_at = ?")
        params.append(_utc_now())
        params.append(frame_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE frames SET {', '.join(fields)} WHERE frame_id = ?",
                params,
            )

    def get_frame(self, frame_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()
        return self._decode_frame_row(row)

    def list_frames(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM frames WHERE run_id = ? ORDER BY created_at, frame_id",
                (run_id,),
            ).fetchall()
        return [self._decode_frame_row(row) for row in rows]

    def get_call_tree(self, run_id: str) -> list[dict[str, Any]]:
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

    def get_frame_lineage(self, frame_id: str) -> list[dict[str, Any]]:
        lineage: list[dict[str, Any]] = []
        current = self.get_frame(frame_id)
        while current:
            lineage.append(current)
            parent_id = current.get("parent_frame_id")
            current = self.get_frame(parent_id) if parent_id else None
        lineage.reverse()
        return lineage

    def get_frame_entry_snapshot(self, frame_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
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
        out = dict(row)
        out["state_json"] = json.loads(out["state_json"])
        out["prev_snapshot_id"] = graph_row["prev_snapshot_id"] if graph_row else None
        out["branch_origin_snapshot_id"] = graph_row["branch_origin_snapshot_id"] if graph_row else None
        return out

    def get_latest_snapshot_for_frame(self, frame_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
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
        out = dict(row)
        out["state_json"] = json.loads(out["state_json"])
        out["prev_snapshot_id"] = graph_row["prev_snapshot_id"] if graph_row else None
        out["branch_origin_snapshot_id"] = graph_row["branch_origin_snapshot_id"] if graph_row else None
        return out

    def create_snapshot(
        self,
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
        snapshot_uid = snapshot_id or str(uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO snapshots (
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
                    json.dumps(state),
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

    def load_snapshot(self, snapshot_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
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
        out = dict(row)
        out["state_json"] = json.loads(out["state_json"])
        out["prev_snapshot_id"] = graph_row["prev_snapshot_id"] if graph_row else None
        out["branch_origin_snapshot_id"] = graph_row["branch_origin_snapshot_id"] if graph_row else None
        return out

    def list_snapshots(self, run_id: str, frame_id: Optional[str] = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM snapshots WHERE run_id = ?"
        params: list[Any] = [run_id]
        if frame_id is not None:
            query += " AND frame_id = ?"
            params.append(frame_id)
        query += " ORDER BY created_at, snapshot_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            output = []
            for row in rows:
                item = dict(row)
                item["state_json"] = json.loads(item["state_json"])
                graph_row = conn.execute(
                    "SELECT prev_snapshot_id, branch_origin_snapshot_id FROM snapshot_graph WHERE snapshot_id = ?",
                    (item["snapshot_id"],),
                ).fetchone()
                item["prev_snapshot_id"] = graph_row["prev_snapshot_id"] if graph_row else None
                item["branch_origin_snapshot_id"] = graph_row["branch_origin_snapshot_id"] if graph_row else None
                output.append(item)
        return output

    def get_snapshot_chain(self, snapshot_id: str) -> list[dict[str, Any]]:
        chain: list[dict[str, Any]] = []
        current_id = snapshot_id
        while current_id:
            snapshot = self.load_snapshot(current_id)
            if not snapshot:
                break
            chain.append(snapshot)
            current_id = snapshot.get("prev_snapshot_id")
        chain.reverse()
        return chain

    def list_workflow_node_instances(self, run_id: str, node_name: str) -> list[dict[str, Any]]:
        frames = self.list_frames(run_id)
        matches = []
        for frame in frames:
            metadata = frame.get("metadata_json", {})
            if metadata.get("workflow_node_name") == node_name:
                matches.append(frame)
        return sorted(matches, key=lambda frame: int(frame.get("metadata_json", {}).get("workflow_instance_id") or 0))

    def record_event(
        self,
        run_id: str,
        frame_id: Optional[str],
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> str:
        event_id = str(uuid4())
        payload = payload or {}
        with self._connect() as conn:
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
                    json.dumps(payload),
                    _utc_now(),
                ),
            )
        return event_id

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY event_seq",
                (run_id,),
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["payload_json"] = json.loads(item["payload_json"])
            output.append(item)
        return output

    def list_frame_events(self, frame_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE frame_id = ? ORDER BY event_seq",
                (frame_id,),
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["payload_json"] = json.loads(item["payload_json"])
            output.append(item)
        return output

    def assess_frame_replay(self, frame_id: str) -> dict[str, Any]:
        blockers: list[dict[str, Any]] = []
        tool_events = [
            event for event in self.list_frame_events(frame_id)
            if event["event_type"] == "tool_completed"
        ]
        for event in tool_events:
            payload = event["payload_json"]
            replay_policy = payload.get("replay_policy", "allow")
            side_effect_type = payload.get("side_effect_type", "pure")
            if replay_policy == "deny":
                blockers.append(
                    {
                        "tool_name": payload.get("tool_name"),
                        "side_effect_type": side_effect_type,
                        "replay_policy": replay_policy,
                        "reason": "Tool execution is marked non-replayable.",
                    }
                )
        return {
            "frame_id": frame_id,
            "safe": len(blockers) == 0,
            "blockers": blockers,
            "tool_event_count": len(tool_events),
        }

    def resume_from_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        snapshot = self.load_snapshot(snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")
        state = dict(snapshot["state_json"])
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

    def fork_run_from_snapshot(
        self,
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

        frame_id_map: dict[str, str] = {}
        for index, frame in enumerate(lineage):
            new_status = "completed" if index < len(lineage) - 1 else "pending"
            new_frame_id = self.create_frame(
                run_id=new_run_id,
                frame_type=frame["frame_type"],
                frame_name=frame["frame_name"],
                status=new_status,
                parent_frame_id=frame_id_map.get(frame.get("parent_frame_id")),
                caller_frame_id=frame_id_map.get(frame.get("caller_frame_id")),
                return_to_frame_id=frame_id_map.get(frame.get("return_to_frame_id")),
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

        root_source_id = lineage[0]["frame_id"]
        new_root_frame_id = frame_id_map[root_source_id]
        self.set_run_root_frame(new_run_id, new_root_frame_id)

        cloned_state = json.loads(json.dumps(snapshot["state_json"]))
        cloned_state["run_id"] = new_run_id
        cloned_state["frame_id"] = frame_id_map[target_frame["frame_id"]]
        if cloned_state.get("root_frame_id") in frame_id_map:
            cloned_state["root_frame_id"] = frame_id_map[cloned_state["root_frame_id"]]
        if cloned_state.get("parent_frame_id") in frame_id_map:
            cloned_state["parent_frame_id"] = frame_id_map[cloned_state["parent_frame_id"]]
        if cloned_state.get("return_to_frame_id") in frame_id_map:
            cloned_state["return_to_frame_id"] = frame_id_map[cloned_state["return_to_frame_id"]]

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

    def restart_frame(self, frame_id: str, *, allow_unsafe_replay: bool = False) -> dict[str, Any]:
        entry_snapshot = self.get_frame_entry_snapshot(frame_id)
        if not entry_snapshot:
            raise ValueError(f"Frame '{frame_id}' does not have a frame_entry snapshot")
        return self.fork_run_from_snapshot(
            entry_snapshot["snapshot_id"],
            allow_unsafe_replay=allow_unsafe_replay,
            metadata={"restart_from_frame_id": frame_id},
        )

    def add_artifact(
        self,
        run_id: str,
        artifact_type: str,
        payload: dict[str, Any],
        *,
        frame_id: Optional[str] = None,
    ) -> str:
        artifact_id = str(uuid4())
        with self._connect() as conn:
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
                    json.dumps(payload),
                    _utc_now(),
                ),
            )
        return artifact_id

    def save_checkpoint(self, scope: str, payload: dict[str, Any], uid: Optional[str] = None) -> str:
        """
        Backward-compatible wrapper.

        Legacy checkpoints are persisted as snapshots. The state payload remains
        available through load_checkpoint(), but new code should use
        create_snapshot()/load_snapshot().
        """
        run_id = payload.get("run_id") or self.create_run(
            entry_type="legacy_checkpoint",
            entry_name=payload.get("agent_name") or scope,
            metadata={"compat_scope": scope},
        )
        frame_id = payload.get("frame_id") or self.create_frame(
            run_id=run_id,
            frame_type=scope,
            frame_name=payload.get("agent_name") or scope,
            status="paused",
            position={"scope": scope},
            metadata={"compat_checkpoint": True},
        )
        state = dict(payload)
        state.setdefault("scope", scope)
        state.setdefault("run_id", run_id)
        state.setdefault("frame_id", frame_id)
        return self.create_snapshot(
            run_id,
            frame_id,
            snapshot_kind="legacy_checkpoint",
            state=state,
            resume_strategy="exact",
            snapshot_id=uid,
        )

    def load_checkpoint(self, uid: str) -> Optional[dict[str, Any]]:
        snapshot = self.load_snapshot(uid)
        if not snapshot:
            return None
        state = snapshot["state_json"]
        state.setdefault("_snapshot_meta", {
            "snapshot_id": snapshot["snapshot_id"],
            "run_id": snapshot["run_id"],
            "frame_id": snapshot["frame_id"],
            "snapshot_kind": snapshot["snapshot_kind"],
            "resume_strategy": snapshot["resume_strategy"],
            "created_at": snapshot["created_at"],
        })
        return state

    def delete_checkpoint(self, uid: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM snapshots WHERE snapshot_id = ?", (uid,))

    @staticmethod
    def _decode_frame_row(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if not row:
            return None
        out = dict(row)
        for key in ("position_json", "inputs_json", "outputs_json", "resolved_config_json", "metadata_json"):
            out[key] = json.loads(out[key])
        return out
