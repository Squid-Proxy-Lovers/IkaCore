"""Fresh-process crash/recovery fixture. Never imports a network provider."""

import argparse
import json
import os
import signal
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from IkaCore import ExecutionLimits, ExecutionPolicy, IkaBaseAgent, IkaTools, OutputConfig
from IkaCore.snapshot_store import SnapshotStore
from IkaCore.workflow import IkaWorkflow, WorkflowEdge, WorkflowNode
from IkaModel.chat_interface.tool_execution_sync import execute_tool


def die(method):
    if method == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(10)  # allow main-thread signal delivery when called in a tool worker
    else:
        os._exit(73)


def agent(path, point="", method="exit"):
    effect = path.with_suffix(".effects")
    def tool(args):
        with effect.open("a") as out:
            out.write("effect\n")
            out.flush()
            os.fsync(out.fileno())
        if point == "during_tool":
            die(method)
        return "effect completed"
    obj = IkaBaseAgent("durable", prompt="finish", model_id="gpt-4o", api_key="test",
        tools=[IkaTools("effect", "record effect", {"type": "object"}, execute_function=tool, replay_policy="deny")],
        limits=ExecutionLimits(maxsteps=3), policy=ExecutionPolicy(max_step_extensions=0),
        output=OutputConfig(cli=False, show_usage_level0=False))
    obj.enable_snapshots(str(path))
    return obj


def finished(name="done"):
    call = {"id": "end", "type": "function", "function": {"name": "agent_end", "arguments": json.dumps({"input": name})}}
    return {"content": name, "tool_calls": [call], "executed_tool_calls": [call],
            "usage": {"input_tokens": 3, "output_tokens": 2}, "cost": {}}


def mocked_chat(model, messages, history, **kwargs):
    # The real tool wrapper records start/completion frames. External effects are
    # deliberately non-idempotent so an unsafe replay would be observable.
    execute_tool("effect", {"value": 1}, kwargs["tool_executors"])
    return finished()


def crash_agent(args):
    obj = agent(args.path, args.point, args.method)
    store = obj._snapshot_runtime.store
    create, event = store.create_snapshot, store.record_event
    def snapshot(*a, **kw):
        result = create(*a, **kw)
        if a[2] == args.point:
            die(args.method)
        return result
    def record(run, frame, kind, payload=None):
        result = event(run, frame, kind, payload)
        if kind == "snapshot_created" and payload["boundary"] == args.point + ":committed":
            die(args.method)
        return result
    # For transaction crashes, exit after the snapshot INSERT or after the event
    # INSERT but before the context manager can commit.
    if args.point.startswith("event:"):
        def record(run, frame, kind, payload=None):
            result = event(run, frame, kind, payload)
            if kind == "snapshot_created" and payload["boundary"] == args.point[6:]:
                die(args.method)
            return result
    store.create_snapshot = snapshot
    store.record_event = record
    import IkaCore.snapshot_session as session
    original = session.record_boundary
    def boundary(state, label, payload):
        result = original(state, label, payload)
        if args.point == "after:" + label:
            die(args.method)
        return result
    with patch("IkaCore.snapshot_agent_hooks.record_boundary", boundary), patch("IkaCore.agent_snapshot_execution.record_boundary", boundary), patch("IkaCore.agent_chat_support.chat", mocked_chat):
        obj.execution()
    raise AssertionError("crash hook was not reached")


def recover_agent(args):
    store = SnapshotStore(str(args.path))
    with closing(sqlite3.connect(args.path)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        run = conn.execute("SELECT run_id FROM runs").fetchone()[0]
    snapshots = store.list_snapshots(run)
    events = store.list_events(run)
    boundaries = [s for s in snapshots if s["resume_strategy"] == "exact"]
    for snap in boundaries:
        assert sum(e["event_type"] == "snapshot_created" and e["payload_json"]["snapshot_id"] == snap["snapshot_id"] for e in events) == 1
    latest = boundaries[-1]
    obj = agent(args.path)
    count = 0
    def chat(*a, **kw):
        nonlocal count
        count += 1
        return mocked_chat(*a, **kw)
    denied = False
    with patch("IkaCore.agent_chat_support.chat", chat):
        try:
            result = obj.resume_from_snapshot(latest["snapshot_id"])
        except ValueError as exc:
            assert "unsafe-replay" in str(exc) or "non-replayable" in str(exc)
            denied = True
            result = {}
    if not denied:
        assert store.get_run(run)["status"] == "completed"
        assert store.get_frame(obj._snapshot_runtime.root_frame_id)["status"] == "completed"
    print(json.dumps({"denied": denied, "calls": count, "result": result.get("final_message"),
        "usage": obj._total_usage, "kind": latest["snapshot_kind"], "snapshots": len(snapshots)}))
    obj.shutdown()


def workflow(path, point="", method="exit"):
    nodes = [WorkflowNode("root", agent(path)), WorkflowNode("parallel", agent(path), instances=3), WorkflowNode("join", agent(path))]
    wf = IkaWorkflow("durable", "test", nodes, [WorkflowEdge("root", "parallel"), WorkflowEdge("parallel", "join")],
        max_parallel_workers=2, compress_hook=lambda contexts, agent: "context", output=OutputConfig(cli=False))
    store = wf.enable_snapshots(str(path))
    original = store.create_snapshot
    def snapshot(run, frame, kind, state, **kw):
        result = original(run, frame, kind, state, **kw)
        if kind == point and (kind != "workflow_instance_completed" or "parallel:0" in state["instance_results"]):
            die(method)
        return result
    store.create_snapshot = snapshot
    return wf


def workflow_run(args):
    crashing = args.mode.endswith("crash")
    effects = "effects" in args.mode
    wf = workflow(args.path, args.point if crashing else "", args.method)
    calls = 0
    def chat(*a, **kw):
        nonlocal calls
        calls += 1
        return mocked_chat(*a, **kw) if effects else finished()
    with patch("IkaCore.agent_chat_support.chat", chat):
        if crashing:
            wf.run(parallel=True)
            raise AssertionError("workflow crash hook not reached")
        with closing(sqlite3.connect(args.path)) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            row = conn.execute("SELECT snapshot_id FROM snapshots WHERE resume_strategy = 'workflow' ORDER BY created_at DESC LIMIT 1").fetchone()
        try:
            output = wf.resume_from_snapshot(row[0])
        except ValueError as error:
            if not effects or "non-replayable" not in str(error):
                raise
            assert calls == 0
            print(json.dumps({"denied": True, "calls": calls}))
        else:
            print(json.dumps({"nodes": sorted(output), "instances": sorted(wf._workflow_snapshot.results), "calls": calls}))
    wf.async_executor.shutdown()
    for node in wf.nodes:
        node.agent.shutdown()


def writer(args):
    store = SnapshotStore(str(args.path))
    for n in range(30):
        with store.transaction():
            store.record_event(args.point, None, "writer", {"pid": os.getpid(), "n": n})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("path", type=Path)
    parser.add_argument("point", nargs="?", default="")
    parser.add_argument("method", nargs="?", default="exit")
    args = parser.parse_args()
    if args.mode == "crash":
        crash_agent(args)
    elif args.mode == "recover":
        recover_agent(args)
    elif args.mode.startswith("workflow-"):
        workflow_run(args)
    else:
        writer(args)
