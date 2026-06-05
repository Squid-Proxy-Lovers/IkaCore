"""Tests for the lightweight checkpoint persistence layer."""

from IkaCore.checkpoint import CheckpointStore


class TestCheckpointStore:
    def test_save_load_and_delete_round_trip(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        store = CheckpointStore(str(db_path))
        payload = {
            "scope": "agent",
            "agent_name": "Tester",
            "remaining_steps": 4,
            "last_content": "hello",
        }

        uid = store.save_checkpoint("agent", payload)
        assert uid

        loaded = store.load_checkpoint(uid)
        assert loaded == payload

        store.delete_checkpoint(uid)
        assert store.load_checkpoint(uid) is None

    def test_save_checkpoint_with_explicit_uid_overwrites(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        store = CheckpointStore(str(db_path))
        uid = "fixed-uid"

        first = {"scope": "stage", "value": 1}
        second = {"scope": "stage", "value": 2}

        store.save_checkpoint("stage", first, uid=uid)
        store.save_checkpoint("stage", second, uid=uid)

        loaded = store.load_checkpoint(uid)
        assert loaded == second
