"""Durability and isolation of the single snapshot persistence backend."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from IkaCore.runtime_control import RuntimeControl, RuntimePauseRequested
from IkaCore.snapshot_session import SnapshotSession, record_boundary
from IkaCore.snapshot_store import SnapshotStore


def test_nested_transactions_rollback_only_failed_scope(tmp_path):
    store = SnapshotStore(str(tmp_path / 'runtime.db'))
    with store.transaction():
        run = store.create_run('agent', 'outer')
        with pytest.raises(ValueError, match='rollback'):
            with store.transaction():
                discarded = store.create_frame(run, 'agent', 'discarded')
                raise ValueError('rollback')
        with store.transaction():
            retained = store.create_frame(run, 'agent', 'retained')
    assert store.get_frame(discarded) is None
    assert store.get_frame(retained)['run_id'] == run
    with pytest.raises(ValueError):
        with store.transaction():
            rolled_back = store.create_run('agent', 'rolled back')
            raise ValueError('rollback')
    assert store.get_run(rolled_back) is None
    assert not hasattr(store._connections, 'connection')


def test_snapshot_boundary_is_atomic_and_pause_is_committed(tmp_path, monkeypatch):
    store = SnapshotStore(str(tmp_path / 'runtime.db'))
    run = store.create_run('agent', 'test')
    frame = store.create_frame(run, 'agent', 'test')
    agent = SimpleNamespace(maxsteps=2, name='test', message_history={}, memory_access={}, _total_usage={}, _total_cost={})
    session = SnapshotSession(agent, store, RuntimeControl(pause_points={'pre_model'}), run_id=run, root_frame_id=frame)
    real_record = store.record_event
    def fail(*args):
        raise sqlite3.OperationalError('simulated disk failure')
    monkeypatch.setattr(store, 'record_event', fail)
    with pytest.raises(sqlite3.OperationalError):
        record_boundary(session, 'pre_model', {})
    assert store.list_snapshots(run) == [] and store.list_events(run) == []
    assert session.last_snapshot_id is None
    monkeypatch.setattr(store, 'record_event', real_record)
    with pytest.raises(RuntimePauseRequested) as paused:
        record_boundary(session, 'pre_model', {})
    reopened = SnapshotStore(store.db_path)
    assert reopened.load_snapshot(paused.value.snapshot_id)
    assert reopened.get_run(run)['status'] == 'paused'
    assert reopened.get_frame(frame)['status'] == 'paused'
    assert len(reopened.list_events(run)) == 1


def test_transactions_use_one_connection_and_isolate_workers(tmp_path, monkeypatch):
    store = SnapshotStore(str(tmp_path / 'runtime.db'))
    connections = []
    original = sqlite3.connect
    def tracked(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection
    monkeypatch.setattr(sqlite3, 'connect', tracked)
    with store.transaction():
        run = store.create_run('agent', 'test')
        frame = store.create_frame(run, 'agent', 'test')
        store.create_snapshot(run, frame, 'frame_entry', {})
        store.record_event(run, frame, 'created')
    assert len(connections) == 1
    def worker(index):
        with store.transaction():
            store.record_event(run, frame, str(index))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(20)))
    assert len(connections) == 21
    events = store.list_events(run)
    assert [event['event_seq'] for event in events] == list(range(1, 22))
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match=r'closed|thread'):
            connection.execute('SELECT 1')
    with original(store.db_path) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'checkpoints'").fetchall() == []
