from unittest.mock import MagicMock, patch

import pytest

from IkaCore import IkaBaseAgent
from IkaCore.runtime_control import RuntimeControl
from IkaCore.workflow import IkaWorkflow, WorkflowEdge, WorkflowNode


def workflow(tmp_path, *, asynchronous=False, control=None):
    calls = []
    nodes = []
    for name in ("root", "left", "right", "join"):
        agent = IkaBaseAgent(name=name, description="test", prompt="task", model_id="gpt-4o", api_key="test")
        def clone(name=name, **kwargs):
            new = IkaBaseAgent(name=name, description="test", prompt="task", model_id="gpt-4o", api_key="test")
            def execute():
                calls.append(name)
                return {"final_message": name, "summary": name}
            new.execution = execute
            return new
        agent.clone_for_run = clone
        nodes.append(WorkflowNode(name, agent, instances=2 if name == "left" else 1))
    wf = IkaWorkflow("test", "test", nodes, [WorkflowEdge("root", n) for n in ("left", "right")]
                     + [WorkflowEdge(n, "join") for n in ("left", "right")],
                     compress_hook=lambda contexts, agent: " | ".join(contexts), max_parallel_workers=4)
    store = wf.enable_snapshots(str(tmp_path / "runtime.db"), control)
    return wf, store, calls


@pytest.mark.parametrize("asynchronous", [False, True])
def test_workflow_snapshots_store_instances_and_reuse_completed_outputs(tmp_path, asynchronous):
    wf, store, calls = workflow(tmp_path)
    with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
        first = wf.run("brief", use_async=asynchronous)
        info = wf.snapshot_runtime_info()
        second = wf.resume_from_snapshot(info["last_snapshot_id"])
    assert set(first) == set(second) == {"root", "left", "right", "join"}
    assert calls.count("left") == 2 and len(calls) == 5
    assert len(store.list_workflow_node_instances(info["run_id"], "left")) == 4
    assert store.get_run(info["run_id"])["status"] == "completed"
    wf.async_executor.shutdown()


def test_selective_replay_invalidates_descendants_and_preserves_other_instances(tmp_path):
    control = RuntimeControl(parallel_replay={"left": [1]})
    wf, store, calls = workflow(tmp_path, control=control)
    wf.run()
    calls.clear()
    wf.resume_from_snapshot(wf.snapshot_runtime_info()["last_snapshot_id"])
    assert calls == ["left", "join"]
    wf.async_executor.shutdown()


def test_workflow_entry_pause_has_no_execution_and_resumes_once(tmp_path):
    wf, store, calls = workflow(tmp_path, control=RuntimeControl(pause_points={"workflow_entry"}))
    assert wf.run() == {} and calls == []
    info = wf.snapshot_runtime_info()
    assert info["paused"] and store.get_run(info["run_id"])["status"] == "paused"
    assert len(wf.resume_from_snapshot(info["last_snapshot_id"])) == 4
    wf.async_executor.shutdown()


def test_replay_rejects_changed_graph_and_invalid_selection(tmp_path):
    wf, store, calls = workflow(tmp_path)
    wf.run()
    snapshot = wf.snapshot_runtime_info()["last_snapshot_id"]
    wf.nodes[0].agent.prompt = "changed task"
    with pytest.raises(ValueError, match="configuration"):
        wf.resume_from_snapshot(snapshot)
    wf.nodes[0].agent.prompt = "task"
    wf._workflow_snapshot.control.parallel_replay = {"left": [99]}
    with pytest.raises(ValueError, match="existing"):
        wf.resume_from_snapshot(snapshot)
    wf.async_executor.shutdown()


def test_replay_checks_nested_tool_effects_before_selected_instance_is_rerun(tmp_path):
    wf, store, calls = workflow(tmp_path, control=RuntimeControl(parallel_replay={"left": [1]}))
    wf.run()
    info = wf.snapshot_runtime_info()
    left = store.list_workflow_node_instances(info["run_id"], "left")[1]
    child = store.create_frame(info["run_id"], "tool", "external", parent_frame_id=left["frame_id"], status="completed")
    store.record_event(info["run_id"], child, "tool_completed", {"tool_name": "external", "replay_policy": "deny"})
    with pytest.raises(ValueError, match="non-replayable"):
        wf.resume_from_snapshot(info["last_snapshot_id"])
    wf.resume_from_snapshot(info["last_snapshot_id"], allow_unsafe_replay=True)
    wf.async_executor.shutdown()


def test_worker_failures_are_recorded_and_dependents_are_blocked(tmp_path):
    wf, store, calls = workflow(tmp_path)
    def clone(**kwargs):
        agent = IkaBaseAgent(name="root", description="test", prompt="task", model_id="gpt-4o", api_key="test")
        agent.execution = MagicMock(side_effect=ValueError("expected"))
        return agent
    wf.nodes[0].agent.clone_for_run = clone
    with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
        assert wf.run(use_async=True) == {}
    info = wf.snapshot_runtime_info()
    instances = store.list_workflow_node_instances(info["run_id"], "root")
    assert instances[0]["status"] == "failed"
    wf.async_executor.shutdown()


def test_parallel_snapshot_results_remain_complete(tmp_path):
    wf, store, calls = workflow(tmp_path)
    with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
        wf.run(use_async=True)
    info = wf.snapshot_runtime_info()
    snapshot = store.load_snapshot(info["last_snapshot_id"])
    assert set(snapshot["state_json"]["instance_results"]) == {"root:0", "left:0", "left:1", "right:0", "join:0"}
    wf.async_executor.shutdown()
