from pathlib import Path

import pytest

from IkaCore.snapshot_store import SnapshotStore as CheckpointStore


class TestRuntimeStore:
    def test_runtime_store_persists_runs_frames_snapshots_and_events(self, tmp_path: Path):
        store = CheckpointStore(str(tmp_path / "runtime.db"))
        run_id = store.create_run("agent", "root")
        root_frame = store.create_frame(run_id, "agent", "root")
        store.set_run_root_frame(run_id, root_frame)
        child_frame = store.create_frame(
            run_id,
            "subagent_call",
            "child",
            parent_frame_id=root_frame,
            caller_frame_id=root_frame,
            return_to_frame_id=root_frame,
        )
        snapshot_id = store.create_snapshot(
            run_id,
            child_frame,
            "boundary",
            {"scope": "agent", "hello": "world"},
            label="child-boundary",
        )
        store.record_event(run_id, child_frame, "child_completed", {"ok": True})

        snapshot = store.load_snapshot(snapshot_id)
        assert snapshot is not None
        assert snapshot["state_json"]["hello"] == "world"

        tree = store.get_call_tree(run_id)
        assert len(tree) == 1
        assert tree[0]["frame_id"] == root_frame
        assert len(tree[0]["children"]) == 1
        assert tree[0]["children"][0]["frame_id"] == child_frame
        assert tree[0]["children"][0]["return_to_frame_id"] == root_frame

        events = store.list_events(run_id)
        assert len(events) == 1
        assert events[0]["event_type"] == "child_completed"
        assert events[0]["payload_json"]["ok"] is True

    def test_fork_run_from_snapshot_clones_lineage_into_new_run(self, tmp_path: Path):
        store = CheckpointStore(str(tmp_path / "runtime.db"))
        run_id = store.create_run("agent", "root")
        root_frame = store.create_frame(run_id, "agent", "root", position={"level": 0})
        store.set_run_root_frame(run_id, root_frame)
        child_frame = store.create_frame(
            run_id,
            "stage",
            "child",
            parent_frame_id=root_frame,
            caller_frame_id=root_frame,
            return_to_frame_id=root_frame,
            position={"level": 1},
        )
        snapshot_id = store.create_snapshot(
            run_id,
            child_frame,
            "frame_entry",
            {
                "scope": "stage",
                "run_id": run_id,
                "frame_id": child_frame,
                "root_frame_id": root_frame,
                "parent_frame_id": root_frame,
                "return_to_frame_id": root_frame,
            },
        )

        forked = store.fork_run_from_snapshot(snapshot_id)
        assert forked["run_id"] != run_id
        assert forked["frame_id"] != child_frame

        new_run = store.get_run(forked["run_id"])
        assert new_run["branch_from_snapshot_id"] == snapshot_id
        frames = store.list_frames(forked["run_id"])
        assert len(frames) == 2
        root_clone = next(frame for frame in frames if frame["parent_frame_id"] is None)
        child_clone = next(frame for frame in frames if frame["parent_frame_id"] == root_clone["frame_id"])
        assert child_clone["frame_name"] == "child"
        assert child_clone["return_to_frame_id"] == root_clone["frame_id"]

    def test_snapshot_chain_uses_linked_history(self, tmp_path: Path):
        store = CheckpointStore(str(tmp_path / "runtime.db"))
        run_id = store.create_run("agent", "root")
        frame_id = store.create_frame(run_id, "agent", "root")
        first = store.create_snapshot(run_id, frame_id, "frame_entry", {"step": 0})
        second = store.create_snapshot(run_id, frame_id, "pre_model", {"step": 1})
        third = store.create_snapshot(run_id, frame_id, "post_model", {"step": 2})

        chain = store.get_snapshot_chain(third)
        assert [item["snapshot_id"] for item in chain] == [first, second, third]
        assert chain[1]["prev_snapshot_id"] == first
        assert chain[2]["prev_snapshot_id"] == second

    def test_assess_frame_replay_blocks_non_replayable_tool_events(self, tmp_path: Path):
        store = CheckpointStore(str(tmp_path / "runtime.db"))
        run_id = store.create_run("agent", "root")
        frame_id = store.create_frame(run_id, "agent", "root")
        store.record_event(
            run_id,
            frame_id,
            "tool_completed",
            {
                "tool_name": "write_to_db",
                "side_effect_type": "side_effecting",
                "replay_policy": "deny",
            },
        )

        assessment = store.assess_frame_replay(frame_id)
        assert assessment["safe"] is False
        assert assessment["blockers"][0]["tool_name"] == "write_to_db"

    def test_restart_frame_requires_allow_unsafe_for_denied_replay(self, tmp_path: Path):
        store = CheckpointStore(str(tmp_path / "runtime.db"))
        run_id = store.create_run("agent", "root")
        frame_id = store.create_frame(run_id, "agent", "root")
        store.set_run_root_frame(run_id, frame_id)
        store.create_snapshot(
            run_id,
            frame_id,
            "frame_entry",
            {"scope": "agent", "run_id": run_id, "frame_id": frame_id, "root_frame_id": frame_id},
        )
        store.record_event(
            run_id,
            frame_id,
            "tool_completed",
            {
                "tool_name": "write_to_db",
                "side_effect_type": "side_effecting",
                "replay_policy": "deny",
            },
        )

        with pytest.raises(ValueError) as exc:
            store.restart_frame(frame_id)
        assert "non-replayable" in str(exc.value)

        replay = store.restart_frame(frame_id, allow_unsafe_replay=True)
        assert replay["run_id"] != run_id
        assert replay["snapshot_id"] is not None


def test_snapshot_store_keeps_legacy_checkpoint_round_trips(tmp_path):
    from IkaCore.checkpoint import CheckpointStore as LegacyStore
    path = str(tmp_path / 'legacy.db')
    legacy = LegacyStore(path)
    legacy.save_checkpoint('agent', {'unchanged': True}, uid='old-uid')
    store = CheckpointStore(path)
    assert store.load_checkpoint('old-uid') == {'unchanged': True}
    store.save_checkpoint('agent', {'unchanged': False}, uid='old-uid')
    assert legacy.load_checkpoint('old-uid') == {'unchanged': False}
    store.delete_checkpoint('old-uid')
    assert legacy.load_checkpoint('old-uid') is None


def test_snapshot_metadata_redacts_credentials_without_changing_ordinary_urls(tmp_path):
    store = CheckpointStore(str(tmp_path / 'runtime.db'))
    run = store.create_run('agent', 'test', metadata={'api_key': 'private-value'})
    frame = store.create_frame(run, 'agent', 'test', resolved_config={'api_url': 'https://example.invalid?scope=read%20only'})
    snapshot = store.create_snapshot(run, frame, 'boundary', {'api_key': 'private-value', 'content': 'sk-' + 'x' * 40})
    state = store.load_snapshot(snapshot)['state_json']
    assert state['api_key'] == '[REDACTED]' and state['content'] == '[REDACTED]'
    assert store.get_run(run)['metadata_json']['api_key'] == '[REDACTED]'
    assert store.get_frame(frame)['resolved_config_json']['api_url'] == 'https://example.invalid?scope=read%20only'
    assert b'private-value' not in Path(store.db_path).read_bytes()
    with pytest.raises(ValueError, match='redacted'):
        store.resume_from_snapshot(snapshot)
    assert store.resume_from_snapshot(snapshot, allow_redacted_state=True)['state']['api_key'] == '[REDACTED]'


def test_non_secret_json_normalization_does_not_block_replay(tmp_path):
    store = CheckpointStore(str(tmp_path / 'runtime.db'))
    run = store.create_run('agent', 'test')
    frame = store.create_frame(run, 'agent', 'test')
    snapshot = store.create_snapshot(run, frame, 'boundary', {'tuple': (1, 2)})
    assert store.resume_from_snapshot(snapshot)['state']['tuple'] == [1, 2]


def test_concurrent_events_have_unique_monotonic_sequence_numbers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    store = CheckpointStore(str(tmp_path / 'runtime.db'))
    run = store.create_run('agent', 'test')
    frame = store.create_frame(run, 'agent', 'test')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.record_event(run, frame, 'progress', {'index': i}), range(30)))
    assert [event['event_seq'] for event in store.list_events(run)] == list(range(1, 31))


def test_unknown_tool_replay_is_denied_until_explicitly_allowed(tmp_path):
    store = CheckpointStore(str(tmp_path / 'runtime.db'))
    run = store.create_run('agent', 'test')
    frame = store.create_frame(run, 'agent', 'test')
    snapshot = store.create_snapshot(run, frame, 'frame_entry', {})
    store.record_event(run, frame, 'tool_completed', {'tool_name': 'external-operation'})
    with pytest.raises(ValueError, match='non-replayable'):
        store.fork_run_from_snapshot(snapshot)
    assert store.fork_run_from_snapshot(snapshot, allow_unsafe_replay=True)['run_id'] != run


def test_snapshot_rejects_frames_from_a_different_run(tmp_path):
    store = CheckpointStore(str(tmp_path / 'runtime.db'))
    first = store.create_run('agent', 'first')
    second = store.create_run('agent', 'second')
    frame = store.create_frame(first, 'agent', 'test')
    with pytest.raises(ValueError, match='belong'):
        store.create_snapshot(second, frame, 'boundary', {})


def test_snapshot_links_are_immutable_and_cannot_cross_runs(tmp_path):
    store = CheckpointStore(str(tmp_path / 'runtime.db'))
    first = store.create_run('agent', 'first')
    second = store.create_run('agent', 'second')
    one = store.create_frame(first, 'agent', 'one')
    two = store.create_frame(second, 'agent', 'two')
    snapshot = store.create_snapshot(first, one, 'boundary', {'value': 1}, snapshot_id='stable')
    with pytest.raises(ValueError, match='immutable'):
        store.create_snapshot(first, one, 'boundary', {'value': 2}, snapshot_id=snapshot)
    for previous in (snapshot, 'missing'):
        with pytest.raises(ValueError, match='predecessor'):
            store.create_snapshot(second, two, 'boundary', {}, prev_snapshot_id=previous)
    with pytest.raises(ValueError, match='origin'):
        store.create_snapshot(first, one, 'boundary', {}, branch_origin_snapshot_id='missing')
    assert store.load_snapshot(snapshot)['state_json'] == {'value': 1}


def test_replay_assessment_includes_unfinished_nested_tools(tmp_path):
    store = CheckpointStore(str(tmp_path / 'runtime.db'))
    run = store.create_run('agent', 'root')
    root = store.create_frame(run, 'agent', 'root')
    stage = store.create_frame(run, 'stage', 'stage', parent_frame_id=root)
    store.create_frame(run, 'tool', 'write', parent_frame_id=stage, status='failed')
    assert not store.assess_frame_replay(root)['safe']
    with pytest.raises(ValueError, match='not found'):
        store.assess_frame_replay('missing')
