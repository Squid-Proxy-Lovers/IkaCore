"""Real abrupt termination and fresh-interpreter recovery, in both journal modes."""

import json
import os
import signal
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from IkaCore.snapshot_store import SnapshotStore

WORKER = Path(__file__).with_name("hardening_crash_worker.py")


def run(*args):
    return subprocess.run([sys.executable, str(WORKER), *map(str, args)], capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("journal", ["delete", "wal"])
@pytest.mark.parametrize("method", ["exit", "kill"])
@pytest.mark.parametrize("point", ["pre_model", "event:post_model", "post_model", "after:post_model", "after:pre_model", "during_tool", "agent_completed", "after:agent_completed"])
def test_killed_agent_recovers_only_committed_boundaries(tmp_path, journal, method, point):
    path = tmp_path / "runtime.db"
    SnapshotStore(str(path))
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA journal_mode=" + journal).fetchone()[0] == journal
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    killed = run("crash", path, point, method)
    assert killed.returncode == (73 if method == "exit" else -signal.SIGKILL), killed.stderr
    before = path.with_suffix(".effects").read_text().count("effect") if path.with_suffix(".effects").exists() else 0
    recovered = run("recover", path)
    assert recovered.returncode == 0, recovered.stderr
    outcome = json.loads(recovered.stdout)
    after = path.with_suffix(".effects").read_text().count("effect") if path.with_suffix(".effects").exists() else 0
    if point in {"during_tool", "post_model", "event:post_model"}:
        assert outcome["denied"] and outcome["calls"] == 0
        assert after == before == 1
    else:
        assert not outcome["denied"] and outcome["result"] == "done"
        assert after == 1
        assert outcome["calls"] == (1 if point in {"pre_model", "after:pre_model"} else 0)
        assert outcome["usage"]["input_tokens"] == 3
        assert outcome["usage"]["output_tokens"] == 2


@pytest.mark.parametrize("journal", ["delete", "wal"])
@pytest.mark.parametrize("point", ["workflow_entry", "workflow_instance_completed", "workflow_completed"])
def test_killed_parallel_workflow_resumes_consistently(tmp_path, journal, point):
    path = tmp_path / "workflow.db"
    SnapshotStore(str(path))
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=" + journal)
    killed = run("workflow-crash", path, point, "kill")
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    recovered = run("workflow-recover", path)
    assert recovered.returncode == 0, recovered.stderr
    output = json.loads(recovered.stdout)
    assert output["nodes"] == ["join", "parallel", "root"]
    assert output["instances"] == ["join:0", "parallel:0", "parallel:1", "parallel:2", "root:0"]


@pytest.mark.parametrize("journal", ["delete", "wal"])
def test_two_process_writers_serialize_sequence_and_preserve_integrity(tmp_path, journal):
    path = tmp_path / "writers.db"
    store = SnapshotStore(str(path))
    run_id = store.create_run("agent", "writers")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=" + journal)
    processes = [subprocess.Popen([sys.executable, str(WORKER), "writer", str(path), run_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    try:
        for process in processes:
            _, error = process.communicate(timeout=30)
            assert process.returncode == 0, error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    events = store.list_events(run_id)
    assert [e["event_seq"] for e in events] == list(range(1, 61))
    assert len({e["payload_json"]["pid"] for e in events}) == 2
    assert all(e["payload_json"]["pid"] != os.getpid() for e in events)
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("journal", ["delete", "wal"])
@pytest.mark.parametrize("point", ["workflow_entry", "workflow_instance_completed", "workflow_completed"])
def test_parallel_workflow_crash_never_duplicates_nonreplayable_effects(tmp_path, journal, point):
    path = tmp_path / "effects.db"
    SnapshotStore(str(path))
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=" + journal)
    killed = run("workflow-effects-crash", path, point, "kill")
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    effects = path.with_suffix(".effects")
    before = effects.read_text().count("effect") if effects.exists() else 0
    recovered = run("workflow-effects-recover", path)
    assert recovered.returncode == 0, recovered.stderr
    result = json.loads(recovered.stdout)
    after = effects.read_text().count("effect")
    if result.get("denied"):
        assert result["calls"] == 0 and after == before
    else:
        assert result["nodes"] == ["join", "parallel", "root"] and after == 5
        assert result["calls"] == 5 - before
