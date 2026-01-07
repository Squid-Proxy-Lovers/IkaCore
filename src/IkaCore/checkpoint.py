import json
import sqlite3
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
from datetime import datetime, timezone


class CheckpointStore:
    """Simple SQLite-based checkpoint store for agents and stages."""

    def __init__(self, db_path: str = "checkpoints.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid TEXT NOT NULL UNIQUE,
                    scope TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at DATETIME NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_uid ON checkpoints(uid)"
            )

    def save_checkpoint(self, scope: str, payload: dict[str, Any], uid: Optional[str] = None) -> str:
        checkpoint_uid = uid or str(uuid4())
        payload_json = json.dumps(payload)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (uid, scope, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    checkpoint_uid,
                    scope,
                    payload_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return checkpoint_uid

    def load_checkpoint(self, uid: str) -> Optional[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT payload_json FROM checkpoints WHERE uid = ? LIMIT 1
                """,
                (uid,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def delete_checkpoint(self, uid: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM checkpoints WHERE uid = ?", (uid,))
