"""Control, cancellation and lifecycle branches with observable outcomes."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from IkaCore import ExecutionPolicy, IkaBaseAgent, IkaStage, IkaTools, OutputConfig
from IkaCore.agent_execution_helpers import build_stage_content_prompt, resolve_stage_model_overrides
from IkaCore.agent_helper_support import validate_final_answer_checks_for_history
from IkaCore.agent_runtime_payloads import string_value
from IkaCore.execution_types import StagedExecutionState
from IkaCore.workflow import IkaWorkflow, WorkflowEdge, WorkflowNode
from IkaCore.workflow_context_cache import compress_once
from IkaCore.workflow_dataflow import _DataflowNodePhases
from IkaCore.workflow_snapshot_state import interrupted_snapshot
from IkaModel import connection_scope as connections
from IkaModel import http_configuration, runtime_options
from IkaModel.base import AgentEndException, HumanInputRequired
from IkaModel.chat_interface import tool_execution_async as tools
from IkaModel.chat_interface.tool_execution_sync import ToolCallPlan
from IkaModel.runtime_errors import IkaRequestCancelled


def agent(**kwargs):
    return IkaBaseAgent("edge", prompt="task", model_id="gpt-4o", api_key="test", output=OutputConfig(cli=False, show_usage_level0=False), **kwargs)


@pytest.mark.parametrize("text", ["{ }", "[ ]"])
def test_agent_end_whitespace_json_is_still_empty(text):
    obj = agent()
    try:
        with pytest.raises(ValueError, match="empty JSON"):
            obj.controls._validate_agent_end_text(text)
        assert obj.controls._target_stage_from_change_call({}, [0], 1) is None
    finally:
        obj.shutdown()


def test_agent_schema_fallbacks_and_structured_submission_policy():
    obj = agent(tools=[IkaTools("submit_fixture", "submit", {"type": "object"}, execute_function=lambda args: "ok")])
    try:
        assert "submit_fixture" in obj.tool_runtime._simple_agent_end_config(["submit_fixture"])[0]
        assert obj.tool_runtime._tool_schema_from_parameters(None) == ({}, [])
        assert obj.tool_runtime._tool_schema_from_parameters("bad") == ({}, [])
        assert obj.tool_runtime._property_from_parameter(7) == {"type": "string", "description": "7"}
    finally:
        obj.shutdown()


def test_stage_overrides_budget_and_clone_snapshot_settings(tmp_path):
    stage = IkaStage("phase", "step", [], api_url="https://proxy.test", api_key="other", model_id="other")
    obj = agent(Stages=[stage])
    obj.context_budget = 1234
    obj.enable_snapshots(str(tmp_path / "state.db"))
    try:
        overrides = resolve_stage_model_overrides(obj, stage)
        assert overrides["api_url"] == "https://proxy.test" and overrides["api_key"] == "other"
        model = obj.preparation._build_stage_barebone_model(stage, 0, "system", [], [], "task")
        assert model.context_budget == 1234
        obj.apply_workflow_stage_wiring({})
        clone = obj.clone_for_run()
        try:
            assert clone._snapshot_runtime.store.db_path == str(tmp_path / "state.db")
        finally:
            clone.shutdown()
        assert obj._last_api_call_ts == 0
        with pytest.raises(RuntimeError, match="without a next_agent"):
            obj.execution_runtime._run_next_agent_execution()
        assert obj.execution_runtime._apply_staged_resume_state(None, {"scope": "unknown"}, StagedExecutionState(0, 1, "")) is None
    finally:
        obj.shutdown()
    assert "CURRENT STAGE" in build_stage_content_prompt("", [], stage, 0)
    assert string_value(7) == "7"


def test_final_check_callable_object_with_no_type_hint_support():
    class Check:
        """Accept a saved history."""
        def __call__(self, history):
            return bool(history)
    check = Check()
    assert validate_final_answer_checks_for_history({"value": 1}, [check]) == [check]


def test_connection_scope_rejects_closed_context_and_tracks_ca_file_changes(monkeypatch, tmp_path):
    assert connections.scoped_client("url", {}, 1, False) is None
    token = connections._state.set(connections._ConnectionState(closed=True))
    try:
        with pytest.raises(RuntimeError, match="closed"):
            connections.scoped_client("url", {}, 1, False)
    finally:
        connections._state.reset(token)
    ca = tmp_path / "ca.pem"
    ca.write_text("first")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    first = http_configuration._env_key()
    ca.write_text("a changed certificate")
    assert http_configuration._env_key() != first
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing"))
    assert {name: stamp for name, _, stamp in http_configuration._env_key()}["SSL_CERT_FILE"] is None


@pytest.mark.parametrize("error", [IkaRequestCancelled("cancel"), AgentEndException({"content": "done"}), HumanInputRequired({"question": "approve?"}), RuntimeError("broken"), None])
def test_async_parallel_tools_keep_interrupt_and_error_semantics(monkeypatch, error):
    mocked = AsyncMock(side_effect=error) if error else AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(tools, "async_execute_tool", mocked)
    plan = ToolCallPlan([], parallel_calls=[("fixture", {}, "id")])
    counts = {}
    if isinstance(error, (IkaRequestCancelled, AgentEndException)):
        with pytest.raises(type(error)):
            asyncio.run(tools._execute_parallel_tool_plan_async(plan, {}, 1, [], 0, counts))
    else:
        interrupt = asyncio.run(tools._execute_parallel_tool_plan_async(plan, {}, 1, [], 0, counts))
        if isinstance(error, HumanInputRequired):
            assert interrupt == {"question": "approve?"}
        elif error:
            assert "broken" in json.loads(plan.tool_call_id_to_result["id"])["error"]
        else:
            assert json.loads(plan.tool_call_id_to_result["id"]) == {"ok": True}
    assert counts == {"fixture": 1}


def test_async_tool_cooperative_cancellation_is_not_converted_to_a_result():
    async def cancelled(args):
        raise IkaRequestCancelled("cancelled by tool")
    with pytest.raises(IkaRequestCancelled):
        asyncio.run(tools.async_execute_tool("fixture", {}, {"fixture": cancelled}))


def test_summary_reuse_client_policy_and_tool_rate_limits(monkeypatch):
    obj = agent(policy=ExecutionPolicy())
    try:
        with runtime_options(reuse_connections=True):
            assert obj.chat._get_chat_client() is None
        enforce = Mock()
        monkeypatch.setattr(obj.chat, "_enforce_rate_limit", enforce)
        obj.chat._enforce_rate_limit_tool("fixture")
        enforce.assert_called_once_with(None, "_last_tool_ts_fixture")
    finally:
        obj.shutdown()


def test_workflow_helper_rejections_and_scheduler_activation(tmp_path):
    obj = agent()
    wf = IkaWorkflow("edge", "test", [WorkflowNode("root", obj), WorkflowNode("child", obj)], [WorkflowEdge("root", "child")], compress_hook=lambda contexts, agent: "|".join(contexts), output=OutputConfig(cli=False))
    try:
        assert wf.snapshot_runtime_info() == {}
        with pytest.raises(ValueError, match="enable_snapshots"):
            wf.resume_from_snapshot("missing")
        assert wf._async_context_for_node(wf.nodes[0], "root", {}, "initial") == "initial"
        ready, pending = set(), {"child"}
        wf._activate_async_dependents({"success": False}, {}, set(), ready, pending)
        assert ready == set() and pending == {"child"}
        wf._activate_unblocked_pending_nodes(pending, {"root"}, ready)
        assert ready == {"child"} and pending == set()
        assert compress_once(wf, wf.nodes[0], ["a", "b"]) == "a|b"
        with pytest.raises(NotImplementedError):
            _DataflowNodePhases._track(None, None, "kind", "root")
        session = SimpleNamespace(store=SimpleNamespace(load_snapshot=lambda _: None))
        with pytest.raises(ValueError, match="missing"):
            interrupted_snapshot(session, "missing")
    finally:
        wf.async_executor.shutdown()
        obj.shutdown()


@pytest.mark.parametrize("options", [{"timeout": -1}, {"timeout": float("inf")}, {"replay_policy": "maybe"}])
def test_tool_metadata_validation_refuses_invalid_execution_policies(options):
    with pytest.raises(ValueError):
        IkaTools("fixture", "fixture", {"type": "object"}, execute_function=lambda args: "ok", **options)


def test_impossible_snapshot_enable_override_fails_clearly(monkeypatch):
    obj = agent()
    try:
        monkeypatch.setattr(obj, "enable_snapshots", lambda: None)
        with pytest.raises(ValueError, match="could not be enabled"):
            obj.configure_runtime_control()
    finally:
        obj.shutdown()


def test_memory_tools_without_backends_publish_no_tools():
    from IkaMem.agent_tools import MemoryTools
    assert MemoryTools("none").tools() == []


def test_unreachable_dependency_is_rejected_by_sync_scheduler():
    from IkaCore.workflow_sync_dataflow import run_sync_dataflow
    wf = SimpleNamespace(start_node="root", _next_reachable_nodes={"root"}, _node_dependencies={"root": {"missing"}})
    with pytest.raises(ValueError, match="unreachable"):
        run_sync_dataflow(wf, None)


def test_forced_final_summary_falls_back_to_last_content(monkeypatch):
    from IkaCore import agent_chat_support, agent_policy
    obj = agent()
    try:
        monkeypatch.setattr(agent_policy, "run_summarization", lambda *a, **kw: "   ")
        assert obj.policy_runtime._finalize_simple_after_max_steps("system", [], Mock(), None, "fallback") == ("fallback", "fallback")
        obj.summarize_final = True
        obj.message_history["summary"]["message"] = 7
        monkeypatch.setattr(agent_chat_support, "summarise_message_history", lambda *a, **kw: "")
        assert obj.chat._build_final_output("", SimpleNamespace(model_id="fixture"))["final_message"] == "7"
    finally:
        obj.shutdown()


def test_workflow_compression_preserves_cancellation_and_invalid_wiring_is_ignored(monkeypatch):
    from IkaCore.workflow_core import WorkflowCore

    obj = agent()
    try:
        wf = SimpleNamespace(edges=[WorkflowEdge("root", "child", edge_type="child")])
        WorkflowCore._bind_stage_wiring(wf)
        monkeypatch.setattr("IkaCore.workflow_core.summarise_message_history", Mock(side_effect=IkaRequestCancelled("stop")))
        with pytest.raises(IkaRequestCancelled):
            WorkflowCore._default_compress_hook(wf, ["upstream"], obj)
    finally:
        obj.shutdown()
