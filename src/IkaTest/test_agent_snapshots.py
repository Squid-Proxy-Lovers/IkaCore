import json

import pytest

from IkaCore import IkaBaseAgent, IkaStage, IkaTools
from IkaCore.runtime_control import RuntimeControl
from IkaModel.chat_interface.tool_execution_sync import execute_tool


def _agent(tmp_path, **kwargs):
    return IkaBaseAgent(name="Parent", description="test", prompt="do the task", model_id="gpt-4o",
                        api_key="test", maxsteps=2, show_usage_level0=False, **kwargs)


def _finished(text="done"):
    call = {"id": "end", "type": "function", "function": {"name": "agent_end", "arguments": json.dumps({"input": text})}}
    return {"content": text, "tool_calls": [call], "executed_tool_calls": [call], "content_before_tools": text,
            "usage": {"input_tokens": 1, "output_tokens": 2}, "cost": {}}


def test_default_execution_does_not_enable_snapshot_store(tmp_path, monkeypatch):
    monkeypatch.setattr("IkaCore.agent_chat_support.chat", lambda *args, **kwargs: _finished())
    agent = _agent(tmp_path)
    result = agent.execution()
    assert "runtime" not in result and agent._snapshot_runtime is None
    agent.shutdown()


def test_execution_creates_agent_and_boundary_snapshots_without_legacy_checkpoint_changes(tmp_path, monkeypatch):
    monkeypatch.setattr("IkaCore.agent_chat_support.chat", lambda *args, **kwargs: _finished())
    agent = _agent(tmp_path)
    store = agent.enable_snapshots(str(tmp_path / "runtime.db"))
    assert not agent.checkpoint and agent.checkpoint_store is None
    result = agent.execution()
    assert result["final_message"] == "done"
    run = result["runtime"]["run_id"]
    frame = result["runtime"]["frame_id"]
    assert store.get_run(run)["status"] == "completed"
    assert store.get_frame(frame)["status"] == "completed"
    kinds = {snapshot["snapshot_kind"] for snapshot in store.list_snapshots(run)}
    assert {"frame_entry", "pre_model", "post_model", "agent_completed"} <= kinds
    assert "api_key" not in store.get_frame(frame)["resolved_config_json"]
    agent.shutdown()


def test_pre_model_pause_and_resume_calls_provider_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("IkaCore.agent_chat_support.chat", lambda *args, **kwargs: calls.append(True) or _finished())
    agent = _agent(tmp_path)
    agent.enable_snapshots(str(tmp_path / "runtime.db"), RuntimeControl(pause_points={"pre_model"}))
    paused = agent.execution()
    assert paused["status"] == "paused" and not calls
    resumed = agent.resume_from_snapshot(paused["checkpoint_uid"])
    assert resumed["final_message"] == "done" and calls == [True]
    agent.shutdown()


def test_post_model_resume_reuses_pending_response_without_double_usage(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("IkaCore.agent_chat_support.chat", lambda *args, **kwargs: calls.append(True) or _finished())
    agent = _agent(tmp_path)
    agent.enable_snapshots(str(tmp_path / "runtime.db"), RuntimeControl(pause_points={"post_model"}))
    paused = agent.execution()
    assert paused["status"] == "paused" and calls == [True]
    usage_before = dict(agent._total_usage)
    resumed = agent.resume_from_snapshot(paused["checkpoint_uid"])
    assert resumed["final_message"] == "done" and calls == [True]
    assert agent._total_usage == usage_before
    agent.shutdown()


def test_staged_execution_records_stage_frames(tmp_path, monkeypatch):
    monkeypatch.setattr("IkaCore.agent_chat_support.chat", lambda *args, **kwargs: _finished())
    agent = _agent(tmp_path, Stages=[IkaStage(name="phase", prompt="do this phase", tools=[])])
    store = agent.enable_snapshots(str(tmp_path / "runtime.db"))
    result = agent.execution()
    frames = store.list_frames(result["runtime"]["run_id"])
    assert any(frame["frame_type"] == "stage" and frame["frame_name"] == "phase" for frame in frames)
    agent.shutdown()


def test_nested_subagent_records_call_tree_and_tool_replay_policy(tmp_path, monkeypatch):
    child = IkaBaseAgent(name="Child", description="child", prompt="child task", model_id="gpt-4o", api_key="test")
    parent = _agent(tmp_path, subagents=[child])
    store = parent.enable_snapshots(str(tmp_path / "runtime.db"))

    def chat(model, messages, history, **kwargs):
        if "Child" in kwargs["tool_executors"]:
            execute_tool("Child", {"input": "child task"}, kwargs["tool_executors"])
            return _finished("parent done")
        return _finished("child done")

    monkeypatch.setattr("IkaCore.agent_chat_support.chat", chat)
    result = parent.execution()
    frames = store.list_frames(result["runtime"]["run_id"])
    assert any(frame["frame_name"] == "Child" and frame["frame_type"] == "agent" for frame in frames)
    assert any(frame["frame_name"] == "Child" and frame["frame_type"] == "tool" for frame in frames)
    assert not store.assess_frame_replay(result["runtime"]["frame_id"])["safe"]
    parent.shutdown()
    child.shutdown()


def test_generator_tool_records_mid_tool_pause_and_requires_replay_decision(tmp_path, monkeypatch):
    def generate(args):
        yield {"__ika_checkpoint__": {"label": "middle", "payload": {"progress": 1}}}
        return "done"

    tool = IkaTools("generate", "test", {"type": "object"}, execute_function=generate, replay_policy="deny")
    agent = _agent(tmp_path, tools=[tool])
    store = agent.enable_snapshots(str(tmp_path / "runtime.db"), RuntimeControl(pause_points={"tool:middle"}))

    def chat(model, messages, history, **kwargs):
        execute_tool("generate", {}, kwargs["tool_executors"])
        return _finished()

    monkeypatch.setattr("IkaCore.agent_chat_support.chat", chat)
    result = agent.execution()
    assert result["status"] == "paused"
    snapshot = store.load_snapshot(result["checkpoint_uid"])
    assert snapshot["state_json"]["tool_checkpoint"]
    assert snapshot["state_json"]["tool_payload"] == {"progress": 1}
    with pytest.raises(ValueError, match="unsafe-replay"):
        agent.resume_from_snapshot(result["checkpoint_uid"])
    agent.shutdown()


def test_completed_snapshot_returns_saved_output_and_new_execution_uses_new_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("IkaCore.agent_chat_support.chat", lambda *args, **kwargs: calls.append(True) or _finished())
    agent = _agent(tmp_path)
    store = agent.enable_snapshots(str(tmp_path / "runtime.db"))
    first = agent.execution()
    assert agent.resume_from_snapshot(first["runtime"]["last_snapshot_id"])["final_message"] == "done"
    assert len(calls) == 1
    second = agent.execution()
    assert second["runtime"]["run_id"] != first["runtime"]["run_id"]
    assert len(calls) == 2
    assert len(store.list_frames(first["runtime"]["run_id"])) == 1
    agent.shutdown()


def test_failed_and_cancelled_execution_do_not_leave_running_snapshot_frames(tmp_path, monkeypatch):
    from IkaModel import IkaRequestCancelled, request_controls
    agent = _agent(tmp_path)
    store = agent.enable_snapshots(str(tmp_path / "runtime.db"))
    def fail(*args, **kwargs):
        raise ValueError("expected")
    monkeypatch.setattr("IkaCore.agent_chat_support.chat", fail)
    with pytest.raises(ValueError):
        agent.execution()
    session = agent._snapshot_runtime
    assert store.get_run(session.run_id)["status"] == "failed"
    with request_controls(cancel_checker=lambda: True), pytest.raises(IkaRequestCancelled):
        agent.execution()
    assert store.get_run(session.run_id)["status"] == "cancelled"
    assert store.get_frame(session.root_frame_id)["status"] == "cancelled"
    agent.shutdown()


def test_agent_fork_and_restart_leave_original_run_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr("IkaCore.agent_chat_support.chat", lambda *args, **kwargs: _finished())
    agent = _agent(tmp_path)
    store = agent.enable_snapshots(str(tmp_path / "runtime.db"), RuntimeControl(pause_points={"pre_model"}))
    first = agent.execution()
    agent.clear_pause_requests()
    fork = agent.fork_from_snapshot(first["checkpoint_uid"])
    assert fork["runtime"]["run_id"] != first["runtime"]["run_id"]
    assert store.get_run(first["runtime"]["run_id"])["status"] == "paused"
    restarted = agent.restart_from_frame(fork["runtime"]["frame_id"])
    assert restarted["final_message"] == "done"
    assert restarted["runtime"]["run_id"] != fork["runtime"]["run_id"]
    agent.shutdown()


def test_agent_replay_requires_explicit_snapshot_enablement(tmp_path):
    agent = _agent(tmp_path)
    for method in (agent.resume_from_snapshot, agent.fork_from_snapshot, agent.restart_from_frame):
        with pytest.raises(ValueError, match="enable_snapshots"):
            method("missing")
    agent.shutdown()


def test_staged_human_input_snapshot_resumes_with_user_answer(tmp_path, monkeypatch):
    agent = _agent(tmp_path, Stages=[IkaStage('phase', 'ask then finish', [], hitl=True)])
    store = agent.enable_snapshots(str(tmp_path / 'runtime.db'))
    messages_seen = []
    def chat(model, messages, history, **kwargs):
        messages_seen.append(messages[0]['content'])
        if len(messages_seen) == 1:
            return {'interrupted': True, 'status': 'awaiting_user_input', 'content': 'Question?',
                    'tool_calls': [], 'executed_tool_calls': [], 'usage': {}, 'cost': {},
                    'interrupt_data': {'question': 'Question?', 'stage_index': 0, 'remaining_steps': 2}}
        return _finished()
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    first = agent.execution()
    assert first['status'] == 'awaiting_user_input'
    assert store.get_run(first['runtime']['run_id'])['status'] == 'paused'
    snapshot = store.load_snapshot(first['checkpoint_uid'])
    assert snapshot['state_json']['scope'] == 'hitl'
    second = agent.resume_from_snapshot(first['checkpoint_uid'], resume_input='User answer')
    assert second['final_message'] == 'done'
    assert messages_seen[-1] == 'User answer'
    agent.shutdown()
