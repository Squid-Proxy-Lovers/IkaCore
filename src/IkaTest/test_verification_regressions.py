"""Public-boundary regressions from the v2 verification repros."""

from contextlib import ExitStack

import pytest

from IkaCore import (
    AgentTopology,
    ExecutionLimits,
    ExecutionPolicy,
    IkaBaseAgent,
    IkaStage,
    IkaTools,
    IkaWorkflow,
    OutputConfig,
    WorkflowEdge,
    WorkflowNode,
)
from IkaCore.testing import MockProvider, provider_response


@pytest.fixture
def agents():
    with ExitStack() as resources:
        def make(name, asynchronous=False, **kwargs):
            model_id = kwargs.pop("model_id", "gpt-4o-mini")
            agent = IkaBaseAgent(
                name, prompt=name + " task", model_id=model_id, api_key="fixture",
                output=OutputConfig(cli=False), limits=ExecutionLimits.bounded(),
                policy=ExecutionPolicy.bounded(async_requests=asynchronous), **kwargs,
            )
            resources.callback(agent.shutdown)
            return agent
        yield make


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("snapshots", [False, True])
@pytest.mark.parametrize("delegated", [False, True])
def test_workflow_approval_blocks_dependent_tools(agents, tmp_path, parallel, asynchronous, snapshots, delegated):
    effects = []
    child = agents("child", asynchronous, stages=[IkaStage("approve", "Ask approval", [], hitl=True)])
    approval = agents("parent", asynchronous, topology=AgentTopology(subagents=(child,))) if delegated else child
    effect = IkaTools("record_effect", "Record effect", {"type": "object"},
                      execute_function=lambda args: effects.append("ran") or "done")
    sink = agents("sink", asynchronous, tools=[effect])
    workflow = IkaWorkflow("approval", "test", [WorkflowNode("approval", approval), WorkflowNode("action", sink)],
                           [WorkflowEdge("approval", "action")], compress_hook=lambda contexts, agent: "\n".join(contexts),
                           output=OutputConfig(cli=False))
    if snapshots:
        workflow.enable_snapshots(str(tmp_path / "workflow.db"))
    def respond(url, payload):
        names = {tool.get("name") or tool.get("function", {}).get("name") for tool in payload.get("tools", [])}
        if "child" in names:
            action = ("child", {"input": "needs approval"})
        elif "ask_user" in names:
            action = ("ask_user", {"question": "Approve operation?"})
        elif "record_effect" in names and not effects:
            action = ("record_effect", {})
        else:
            action = ("agent_end", {"input": "done"})
        return provider_response(url, payload, action)
    try:
        with MockProvider(responder=respond) as mock:
            result = workflow.run(parallel=parallel)
        assert not effects
        assert result == {}  # Interrupted nodes are not completed results.
        assert result.status in {"paused", "awaiting_user_input"}
        assert result.interrupt_data["resumable"] is snapshots
        assert len(mock.requests) == (2 if delegated else 1)
        if not snapshots:
            assert "enable_snapshots" in result.interrupt_data["guidance"]
            with pytest.raises(ValueError, match="enable_snapshots"):
                workflow.resume_from_snapshot("unavailable", resume_input="YES")
    finally:
        workflow.async_executor.shutdown()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("snapshots", [False, True])
def test_later_empty_truncation_stays_interrupted(agents, tmp_path, asynchronous, snapshots):
    agent = agents("truncation", asynchronous)
    if snapshots:
        agent.enable_snapshots(str(tmp_path / "agent.db"))
    calls = 0
    def respond(url, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return provider_response(url, payload, "Working on it")
        if calls == 2:
            return {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [], "usage": {"input_tokens": 10, "output_tokens": 100, "total_tokens": 110}}
        return provider_response(url, payload, "FORCED_SUMMARY_MUST_NOT_BECOME_SUCCESS")
    with MockProvider(responder=respond) as mock:
        result = agent.execution()
    assert result.get("status") in {"paused", "awaiting_user_input"}
    assert result["interrupt_data"]["type"] == "provider_response_truncated"
    assert len(mock.requests) == 2
    assert result["usage"]["output_tokens"] >= 100
    if snapshots:
        saved = agent._snapshot_runtime.store.load_snapshot(result["snapshot_id"])["state_json"]
        assert saved["interrupt_data"]["type"] == "provider_response_truncated"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("restore", ["same", "fresh", "fork"])
@pytest.mark.parametrize("model_id", ["gpt-4o-mini", "claude-sonnet-4-5", "gemini-2.5-flash", "deepseek-chat"])
def test_small_cap_preserves_delegated_continuation(agents, tmp_path, asynchronous, repeat, restore, model_id):
    from IkaModel import request_controls

    def create():
        child = agents("child", asynchronous, stages=[IkaStage("approve", "Approval", [], hitl=True)])
        return agents("parent", asynchronous, model_id=model_id, topology=AgentTopology(subagents=(child,)))
    parent = create()
    db = str(tmp_path / "pause.db")
    store = parent.enable_snapshots(db)
    actions = [("child", {"input": "need approval"}), ("ask_user", {"question": "Approve?"})]
    if repeat:
        actions.append(("ask_user", {"question": "Approve again?"}))
    actions += [("agent_end", {"input": "CHILD_APPROVED_OUTPUT"}), ("agent_end", {"input": "parent done"})]
    scripted = iter(actions)
    with request_controls(tool_result_max_chars=128), MockProvider(
        responder=lambda u, p: provider_response(u, p, next(scripted))
    ) as mock:
        paused = parent.execution()
        if restore == "fresh":
            parent = create()
            parent.enable_snapshots(db)
        snapshot_id = paused["snapshot_id"]
        if restore == "fork":
            snapshot_id = store.fork_run_from_snapshot(snapshot_id, allow_unsafe_replay=True)["snapshot_id"]
        if repeat:
            paused = parent.resume_from_snapshot(snapshot_id, resume_input="YES", allow_unsafe_replay=restore == "fork")
            snapshot_id = paused["snapshot_id"]
        result = parent.resume_from_snapshot(snapshot_id, resume_input="YES", allow_unsafe_replay=restore == "fork")
    assert result["final_message"] == "parent done"
    payload = mock.requests[-1]
    if "input" in payload:
        outputs = [item["output"] for item in payload["input"] if item.get("type") == "function_call_output"]
    elif "contents" in payload:
        outputs = [part["functionResponse"]["response"]["result"] for message in payload["contents"]
                   for part in message["parts"] if "functionResponse" in part]
    else:
        outputs = []
        for message in payload["messages"]:
            if message.get("role") == "tool":
                outputs.append(message["content"])
            elif isinstance(message.get("content"), list):
                outputs.extend(block["content"] for block in message["content"] if block.get("type") == "tool_result")
    assert any("CHILD_APPROVED_OUTPUT" in output for output in outputs)
    assert all(len(output) <= 128 for output in outputs)
    assert "__ika_interrupt__" not in str(mock.requests[-1])
    assert "CHILD_APPROVED_OUTPUT" in str(parent.transcript)


def test_legacy_delegated_snapshot_still_resumes(agents, tmp_path):
    child = agents("child", stages=[IkaStage("approve", "Approval", [], hitl=True)])
    parent = agents("parent", topology=AgentTopology(subagents=(child,)))
    store = parent.enable_snapshots(str(tmp_path / "legacy.db"))
    actions = iter([("child", {"input": "approval"}), ("ask_user", {"question": "Approve?"}),
                    ("agent_end", {"input": "legacy approved"}), ("agent_end", {"input": "done"})])
    with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))) as mock:
        paused = parent.execution()
        snapshot = store.load_snapshot(paused["snapshot_id"])
        state = snapshot["state_json"]
        state["interrupt_data"].pop("tool_result_continuation")
        legacy = store.create_snapshot(snapshot["run_id"], snapshot["frame_id"], "human_input", state)
        result = parent.resume_from_snapshot(legacy, resume_input="YES")
    assert result["final_message"] == "done"
    assert "legacy approved" in str(mock.requests[-1])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_batch_keeps_continuation_identity_outside_capped_text(asynchronous, parallel):
    import asyncio

    from IkaModel import request_controls
    from IkaModel.base import HumanInputRequired
    from IkaModel.chat_interface.tool_execution_async import async_execute_tool_calls
    from IkaModel.chat_interface.tool_execution_sync import execute_tool_calls

    def pause(args):
        raise HumanInputRequired({"question": "Approve?", "delegated": {"snapshot_id": "identity-must-survive"}})
    calls = [{"id": "normal", "function": {"name": "normal", "arguments": "{}"}},
             {"id": "approval", "function": {"name": "pause", "arguments": "{}"}}]
    kwargs = {"tool_metadata": {"normal": {"parallel": parallel}, "pause": {"parallel": parallel}}}
    with request_controls(tool_result_max_chars=1):
        args = (calls, {"normal": lambda args: "normal", "pause": pause}, "openai")
        batch = asyncio.run(async_execute_tool_calls(*args, **kwargs)) if asynchronous else execute_tool_calls(*args, **kwargs)
    assert batch[4]["delegated"]["snapshot_id"] == "identity-must-survive"
    assert batch[4]["tool_result_continuation"] == {"tool_call_id": "approval", "call_index": 1}
    assert all(len(output) <= 1 for output in batch[1])
