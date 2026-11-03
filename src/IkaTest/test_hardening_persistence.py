"""Storage observability, corrupt lineage and replay rejection boundaries."""

import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from IkaCore.runtime_control import RuntimeControl
from IkaCore.snapshot_privacy import sanitize
from IkaCore.snapshot_session import SnapshotSession, record_boundary
from IkaCore.snapshot_store import SnapshotStore


def test_event_queries_artifacts_and_descendant_replay_validation(tmp_path):
    store = SnapshotStore(str(tmp_path / "state.db"))
    run = store.create_run("agent", "root")
    root = store.create_frame(run, "agent", "root")
    child = store.create_frame(run, "tool", "child", parent_frame_id=root, metadata={"replay_policy": "allow"})
    sibling = store.create_frame(run, "tool", "sibling")
    store.record_event(run, child, "tool_completed", {"tool_name": "child", "replay_policy": "allow"})
    store.record_event(run, sibling, "tool_completed", {"tool_name": "sibling"})
    assert len(store.list_frame_events(child)) == 1
    assert store.list_frame_events("missing") == []
    assert store.assess_frame_replay(root)["safe"]
    artifact = store.add_artifact(run, "report", {"value": 7}, frame_id=child)
    with closing(sqlite3.connect(store.db_path)) as conn:
        assert conn.execute("SELECT artifact_id FROM artifacts").fetchone()[0] == artifact
    with pytest.raises(ValueError, match="not found"):
        store.assess_frame_replay("missing")
    other_run = store.create_run("agent", "other")
    other_frame = store.create_frame(other_run, "agent", "other")
    other = store.create_snapshot(other_run, other_frame, "frame_entry", {})
    for snapshot in ("missing", other):
        with pytest.raises(ValueError, match="must belong"):
            store.assess_frame_replay(root, after_snapshot_id=snapshot)


def test_corrupt_lineage_is_refused_and_missing_link_stops_chain(tmp_path):
    store = SnapshotStore(str(tmp_path / "state.db"))
    run = store.create_run("agent", "root")
    root = store.create_frame(run, "agent", "root")
    first = store.create_snapshot(run, root, "frame_entry", {})
    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute("UPDATE snapshot_graph SET prev_snapshot_id = ? WHERE snapshot_id = ?", (first, first))
        conn.commit()
    with pytest.raises(ValueError, match="cyclic"):
        store.get_snapshot_chain(first)
    assert store.get_snapshot_chain("missing") == []
    assert store.get_frame_lineage("missing") == []
    with pytest.raises(ValueError, match="not found"):
        store.fork_run_from_snapshot("missing")


def test_boundary_without_run_or_frame_is_noop(tmp_path):
    store = SnapshotStore(str(tmp_path / "state.db"))
    agent = SimpleNamespace(maxsteps=1)
    session = SnapshotSession(agent, store, RuntimeControl())
    assert record_boundary(session, "pre_model", {"remaining_steps": 1}) is None


@pytest.mark.parametrize("value,expected", [
    ("https://user:password@example.test/path?key=secret&label=safe", "https://example.test/path?key=%5BREDACTED%5D&label=safe"),
    ("http://[invalid", "[REDACTED URL]"),
    (("plain", {"api_key": "secret"}), ["plain", {"api_key": "[REDACTED]"}]),
])
def test_privacy_filters_credentials_in_url_userinfo_and_nested_sequences(value, expected):
    assert sanitize(value) == expected


def test_corrupt_frame_lineage_and_orphan_snapshot_are_refused(tmp_path):
    store = SnapshotStore(str(tmp_path / "state.db"))
    run = store.create_run("agent", "root")
    frame = store.create_frame(run, "agent", "root")
    snapshot = store.create_snapshot(run, frame, "frame_entry", {})
    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute("UPDATE frames SET parent_frame_id = frame_id WHERE frame_id = ?", (frame,))
        conn.commit()
    with pytest.raises(ValueError, match="cyclic runtime frame"):
        store.get_frame_lineage(frame)
    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute("DELETE FROM frames WHERE frame_id = ?", (frame,))
        conn.commit()
    with pytest.raises(ValueError, match=r"Frame.*not found"):
        store.fork_run_from_snapshot(snapshot)


def test_fork_refuses_interruption_with_wrong_root_lineage(tmp_path):
    from IkaCore.snapshot_replay import _clone_workflow_interruptions

    store = SnapshotStore(str(tmp_path / "state.db"))
    run = store.create_run("workflow", "root")
    root = store.create_frame(run, "workflow", "root")
    other = store.create_frame(run, "agent", "other")
    child = store.create_snapshot(run, other, "human_input", {"root_frame_id": other})
    state = {"interrupted_instances": {"node:0": child}}
    with pytest.raises(ValueError, match="must belong"):
        _clone_workflow_interruptions(store, {"run_id": run, "frame_id": root}, state, {}, "target")
