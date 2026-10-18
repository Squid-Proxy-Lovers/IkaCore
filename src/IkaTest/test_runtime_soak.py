"""Opt-in offline sustained agent execution with snapshots and compaction."""

import gc
import json
import os
import sqlite3
import statistics
import threading
import time
import tracemalloc
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from IkaCore import ExecutionLimits, ExecutionPolicy, IkaBaseAgent, IkaTools, OutputConfig
from IkaModel.chat_interface.chat_request import _summarize_sync_if_near_budget
from IkaModel.chat_interface.tool_execution_sync import execute_tool
from IkaModel.request_control import request_controls
from IkaModel.runtime_errors import IkaRequestCancelled


@pytest.mark.soak
@pytest.mark.skipif(os.getenv("IKACORE_RUN_SOAK") != "1", reason="set IKACORE_RUN_SOAK=1 for 2,200-turn soak")
def test_thousands_of_turns_compact_snapshot_cancel_without_resource_growth(tmp_path):
    tool_count = 0
    def tool(args):
        nonlocal tool_count
        tool_count += 1
        return {"answer": args["index"]}
    obj = IkaBaseAgent("soak", prompt="Continue until complete", model_id="gpt-4o", api_key="test",
        tools=[IkaTools("lookup", "read fixture", {"type": "object"}, execute_function=tool, replay_policy="allow")],
        limits=ExecutionLimits(maxsteps=2205), policy=ExecutionPolicy(max_step_extensions=0),
        output=OutputConfig(cli=False, show_usage_level0=False))
    path = tmp_path / "soak.db"
    obj.enable_snapshots(str(path))
    summary_calls, turns = 0, 0
    retained, sizes, durations = [], [], []
    last_batch = time.monotonic()
    def summary(*a, **kw):
        nonlocal summary_calls
        summary_calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "Continue until complete. Prior observations verified."}}], "usage": {"total_tokens": 20}}, request=httpx.Request("POST", "https://example.test"))
    def chat(model, messages, history, **kw):
        nonlocal turns, last_batch
        turns += 1
        model.context_budget = 600
        execute_tool("lookup", {"index": turns}, kw["tool_executors"])
        history["messages"][str(turns)] = {"message": "observation " * 20, "tokens": 100}
        _summarize_sync_if_near_budget(model, history, None, messages)
        if turns % 220 == 0:
            gc.collect()
            retained.append(tracemalloc.get_traced_memory()[0])
            sizes.append(path.stat().st_size)
            durations.append(time.monotonic() - last_batch)
            last_batch = time.monotonic()
        if turns == 1100:
            raise IkaRequestCancelled("scripted mid-loop cancellation")
        if turns == 2201:
            call = {"function": {"name": "agent_end", "arguments": '{"input":"done"}'}}
            return {"content": "done", "tool_calls": [call], "executed_tool_calls": [call], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, "cost": {}}
        return {"content": "continue", "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, "cost": {}}
    # Warm the shared executor before measuring stable resource ownership.
    execute_tool("warm", {}, {"warm": lambda args: "ok"})
    gc.collect()
    initial_threads = {t.ident for t in threading.enumerate()}
    descriptor_dir = Path("/dev/fd")
    initial_fds = len(list(descriptor_dir.iterdir())) if descriptor_dir.exists() else None
    tracemalloc.start()
    started = last_batch = time.monotonic()
    try:
        with patch("IkaCore.agent_chat_support.chat", chat), patch("IkaModel.summarization.api_request_retry", summary):
            with pytest.raises(IkaRequestCancelled):
                obj.execution()
            assert turns == 1100
            # Compaction frequency follows the actual token estimate. A tool
            # checkpoint may now be the most recent boundary when cancellation
            # lands; recover the latest exact model/compaction checkpoint rather
            # than implicitly approving a mid-tool replay.
            session = obj._snapshot_runtime
            checkpoint = session.store.get_latest_snapshot_for_frame(session.root_frame_id)
            assert checkpoint["resume_strategy"] == "exact"
            saved = checkpoint["snapshot_id"]
            assert obj.resume_from_snapshot(saved)["final_message"] == "done"
            assert turns == tool_count == 2201
            complete = obj._snapshot_runtime.last_snapshot_id
            usage = dict(obj._total_usage)
            assert obj.resume_from_snapshot(complete)["final_message"] == "done"
            assert obj._total_usage == usage and turns == tool_count == 2201
            for _ in range(5):
                with request_controls(cancel_checker=lambda: True), pytest.raises(IkaRequestCancelled):
                    obj.execution()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        obj.shutdown()
    gc.collect()
    final_threads = {t.ident for t in threading.enumerate()}
    final_fds = len(list(descriptor_dir.iterdir())) if initial_fds is not None else None
    assert turns == tool_count == 2201 and summary_calls >= 200
    assert obj._total_usage["input_tokens"] == obj._total_usage["output_tokens"] == 2200
    assert obj._total_usage["total_tokens"] == 4400
    assert retained[-1] - retained[2] < 2_000_000
    assert peak < 20_000_000
    assert final_threads <= initial_threads
    assert final_fds is None or final_fds <= initial_fds + 1
    # Append-only audit data grows linearly; it is never promised to be bounded
    # for infinite runs. Limit bytes/turn and late-vs-early batch slope instead.
    assert sizes[-1] < turns * 150_000
    assert sizes[-1] - sizes[7] < 2 * (sizes[3] - sizes[1])
    assert statistics.median(durations[-3:]) < 4 * statistics.median(durations[:3])
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        statuses = dict(conn.execute("SELECT status, COUNT(*) FROM runs GROUP BY status"))
        assert statuses == {"cancelled": 5, "completed": 1}
        snapshots = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    report = {"completed_turns": 2200, "cancellations": 6, "turns": turns, "tools": tool_count, "compactions": summary_calls, "seconds": time.monotonic() - started,
              "retained_bytes": retained, "peak_bytes": peak, "database_bytes": sizes, "batch_seconds": durations,
              "threads_before": len(initial_threads), "threads_after": len(final_threads),
              "fds_before": initial_fds, "fds_after": final_fds, "snapshots": snapshots, "usage": obj._total_usage}
    output = Path(os.getenv("IKACORE_SOAK_REPORT", str(tmp_path / "soak.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
