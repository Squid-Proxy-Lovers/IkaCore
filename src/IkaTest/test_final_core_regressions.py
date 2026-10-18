"""Regressions from the final v2 core review, exercised through public runtimes."""

import json
import threading
from contextlib import ExitStack

import httpx
import pytest

from IkaCore import (
    AgentTopology,
    ExecutionLimits,
    ExecutionPolicy,
    IkaBaseAgent,
    IkaStage,
    IkaWorkflow,
    OutputConfig,
    ProviderConfig,
    WorkflowNode,
)
from IkaCore.testing import MockProvider, provider_response
from IkaModel import IkaRequestCancelled, IkaRequestDeadlineExceeded


@pytest.fixture
def make_agent():
    with ExitStack() as resources:
        def make(name="review", prompt="FIRST_TURN", **kwargs):
            defaults = {"output": OutputConfig(cli=False), "limits": ExecutionLimits.bounded(),
                            "policy": ExecutionPolicy.bounded()}
            defaults.update(kwargs)
            agent = IkaBaseAgent(name, prompt=prompt, model_id="gpt-4o-mini", api_key="fixture", **defaults)
            resources.callback(agent.shutdown)
            return agent
        yield make


def end(text):
    return ("agent_end", {"input": text})


def test_parallel_chains_isolate_next_agent_and_close_owned_clients(make_agent):
    barrier = threading.Barrier(2)
    clients = []
    def factory():
        client = httpx.Client()
        clients.append(client)
        return client
    def synchronize(event):
        if event.kind == "agent_init" and event.data.get("description", "").startswith("Step "):
            barrier.wait(timeout=5)

    downstream = make_agent("next", "NEXT_TASK", provider=ProviderConfig.bounded(client_factory=factory),
                            output=OutputConfig(cli=False, subscribers=(synchronize,)))
    parent = make_agent("parent", topology=AgentTopology(next_agent=downstream))
    workflow = IkaWorkflow("chain-isolation", "review", [
        WorkflowNode("node", parent, instances=2, instance_inputs=["WORK_ALPHA", "WORK_BETA"])
    ], [], output=OutputConfig(cli=False))
    downstream_requests = []
    def respond(url, payload):
        wire = json.dumps(payload)
        if "NEXT_TASK" in wire:
            downstream_requests.append(wire)
            answer = ("ALPHA" if "ALPHA" in wire else "") + ("BETA" if "BETA" in wire else "")
        else:
            answer = "ALPHA" if "WORK_ALPHA" in wire else "BETA"
        return provider_response(url, payload, end(answer))

    try:
        with MockProvider(responder=respond):
            result = workflow.run(parallel=True, concurrency=2)["node"]
        assert {r["final_message"] for r in result.instance_outputs.values()} == {"ALPHA", "BETA"}
        assert len(downstream_requests) == 2
        assert all(("ALPHA" in wire) != ("BETA" in wire) for wire in downstream_requests)
        assert not downstream.transcript["messages"]
        assert len(clients) == 3
        assert not clients[0].is_closed and all(c.is_closed for c in clients[1:])
    finally:
        workflow.async_executor.shutdown()
        for client in clients:
            client.close()


def test_chain_closes_owned_downstream_on_error(make_agent):
    clients = []
    def factory():
        client = httpx.Client()
        clients.append(client)
        return client
    downstream = make_agent("next", "NEXT_TASK", provider=ProviderConfig.bounded(client_factory=factory))
    parent = make_agent("parent", topology=AgentTopology(next_agent=downstream))
    def respond(url, payload):
        if "NEXT_TASK" in json.dumps(payload):
            raise RuntimeError("downstream failure")
        return provider_response(url, payload, end("upstream done"))

    try:
        with MockProvider(responder=respond), pytest.raises(RuntimeError, match="downstream failure"):
            parent.execution()
        assert len(clients) == 2
        assert not clients[0].is_closed and clients[1].is_closed
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("restore", [False, True])
def test_backward_stage_transitions_exhaust_overall_budget(make_agent, tmp_path, restore):
    stages = [IkaStage("work", "Work", [], stage_max_step=1),
              IkaStage("review", "Review", [], stage_max_step=1, allowed_back_to=[0])]
    options = {"stages": stages, "limits": ExecutionLimits.bounded(maxsteps=2, max_tool_rounds=1)}
    agent = make_agent(**options)
    db = str(tmp_path / "budget.db")
    store = agent.enable_snapshots(db)
    if restore:
        agent.request_pause_at("post_model")
    actions = iter([("stage_end", {"reason": "ready"}), ("change_stage", {"stage_index": 0, "reason": "revise"}),
                    ("stage_end", {"reason": "again"}), end("unexpected extra turn")])
    with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))) as mock:
        result = agent.execution()
        if restore:
            other = make_agent(**options)
            other.enable_snapshots(db)
            result = other.resume_from_snapshot(result["snapshot_id"])
    assert len(mock.requests) == 2
    states = [s["state_json"] for s in store.list_snapshots(result["runtime"]["run_id"])]
    assert all(s.get("remaining_steps", 0) >= 0 for s in states)


def test_validation_correction_does_not_reset_consumed_budget(make_agent):
    agent = make_agent(limits=ExecutionLimits.bounded(maxsteps=2, max_tool_rounds=1),
                       policy=ExecutionPolicy.bounded(final_answer_checks=(lambda result: False,),
                                                      validation_retries=1, validation_extra_steps=0))
    actions = iter([end("first"), ("stage_end", {"reason": "no correction yet"}), end("third turn")])
    def respond(url, payload):
        return provider_response(url, payload, next(actions) if payload.get("tools") else "forced final")
    with MockProvider(responder=respond) as mock:
        agent.execution()
    assert len([p for p in mock.requests if p.get("tools")]) == 2


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
def test_delegated_hitl_resumes_child_before_parent(make_agent, tmp_path, staged, use_async):
    def create():
        child = make_agent("child", "CHILD_TASK", stages=[IkaStage("approve", "Get approval", [], hitl=True)],
                           policy=ExecutionPolicy.bounded(use_async=use_async))
        topology = ({"stages": [IkaStage("parent-stage", "Delegate approval", [], subagents=[child])]}
                    if staged else {"topology": AgentTopology(subagents=(child,))})
        return make_agent("parent", "PARENT_TASK", policy=ExecutionPolicy.bounded(use_async=use_async), **topology)
    parent = create()
    db = str(tmp_path / "delegated.db")
    store = parent.enable_snapshots(db)
    actions = iter([("child", {"input": "need approval"}), ("ask_user", {"question": "Approve launch?"}),
                    end("child approved"), end("parent done")])
    with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))) as mock:
        paused = parent.execution()
        assert paused.get("status") == "awaiting_user_input"
        assert len(mock.requests) == 2
        assert store.get_run(paused["runtime"]["run_id"])["status"] == "paused"
        identity = paused["interrupt_data"]["delegated"]
        assert identity["snapshot_id"]
        other = create()
        other.enable_snapshots(db)
        still_paused = other.resume_from_snapshot(paused["snapshot_id"])
        assert still_paused["status"] == "awaiting_user_input"
        assert len(mock.requests) == 2
        result = other.resume_from_snapshot(still_paused["snapshot_id"], resume_input="YES, launch")
    assert result["final_message"] == "parent done"
    assert len(mock.requests) == 4
    assert "YES, launch" in json.dumps(mock.requests[2])
    assert "child approved" in json.dumps(mock.requests[3])
    assert all(f["status"] == "completed" for f in store.list_frames(result["runtime"]["run_id"])
               if f["frame_type"] == "agent")


def test_delegated_hitl_without_snapshots_reports_nonresumable_pause(make_agent):
    child = make_agent("child", stages=[IkaStage("approve", "Get approval", [], hitl=True)])
    parent = make_agent("parent", topology=AgentTopology(subagents=(child,)))
    actions = iter([("child", {"input": "need approval"}), ("ask_user", {"question": "Approve?"}),
                    end("should not finish")])
    with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))) as mock:
        result = parent.execution()
    assert result["status"] == "awaiting_user_input"
    assert result["interrupt_data"]["resumable"] is False
    assert "enable_snapshots" in result["interrupt_data"]["error"]
    assert len(mock.requests) == 2


def test_delegated_repeated_questions_update_pending_child_identity(make_agent, tmp_path):
    child = make_agent("child", stages=[IkaStage("approve", "Get approval", [], hitl=True)])
    parent = make_agent("parent", topology=AgentTopology(subagents=(child,)))
    parent.enable_snapshots(str(tmp_path / "repeat.db"))
    actions = iter([("child", {"input": "approval"}), ("ask_user", {"question": "First?"}),
                    ("ask_user", {"question": "Second?"}), end("all approved"), end("parent done")])
    with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))) as mock:
        first = parent.execution()
        second = parent.resume_from_snapshot(first["snapshot_id"], resume_input="first yes")
        assert second["status"] == "awaiting_user_input"
        assert second["interrupt_data"]["question"] == "Second?"
        assert first["interrupt_data"]["delegated"]["snapshot_id"] != second["interrupt_data"]["delegated"]["snapshot_id"]
        result = parent.resume_from_snapshot(second["snapshot_id"], resume_input="second yes")
    assert result["final_message"] == "parent done"
    assert "all approved" in json.dumps(mock.requests[-1])
    assert "__ika_interrupt__" not in json.dumps(mock.requests[-1])


def test_delegated_control_pause_resumes_saved_child_response(make_agent, tmp_path):
    child = make_agent("child", "CHILD_TASK")
    child.enable_snapshots(str(tmp_path / "template.db"))
    child.request_pause_at("post_model")
    parent = make_agent("parent", topology=AgentTopology(subagents=(child,)))
    # Parent control is inherited. Pause only the child using an explicit clone override.
    native = child.clone_for_run
    def clone(**kwargs):
        run = native(**kwargs)
        original = run.execution
        def execute():
            run.request_pause_at("post_model")
            return original()
        run.execution = execute
        return run
    child.clone_for_run = clone
    parent.enable_snapshots(str(tmp_path / "pause.db"))
    actions = iter([("child", {"input": "task"}), end("saved child output"), end("parent done")])
    with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))) as mock:
        paused = parent.execution()
        assert paused["status"] == "paused"
        child.clone_for_run = native
        result = parent.resume_from_snapshot(paused["snapshot_id"])
    assert result["final_message"] == "parent done"
    assert len(mock.requests) == 3
    assert "saved child output" in json.dumps(mock.requests[-1])


@pytest.mark.parametrize("parallel", [False, True])
def test_workflow_resumes_delegated_child_in_parent_instance(make_agent, tmp_path, parallel):
    child = make_agent("child", stages=[IkaStage("approve", "Get approval", [], hitl=True)])
    parent = make_agent("parent", topology=AgentTopology(subagents=(child,)))
    workflow = IkaWorkflow("nested", "review", [WorkflowNode("node", parent)], [], output=OutputConfig(cli=False))
    workflow.enable_snapshots(str(tmp_path / "workflow.db"))
    actions = iter([("child", {"input": "approval"}), ("ask_user", {"question": "Approve?"}),
                    end("approved child"), end("done")])
    try:
        with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))) as mock:
            workflow.run(parallel=parallel)
            snapshot = workflow._workflow_snapshot.last_snapshot_id
            result = workflow.resume_from_snapshot(snapshot, resume_input="YES")
        assert result["node"].output["final_message"] == "done"
        assert len(mock.requests) == 4
        assert "YES" in json.dumps(mock.requests[2])
    finally:
        workflow.async_executor.shutdown()


def test_fork_preserves_nested_delegated_continuations(make_agent, tmp_path):
    leaf = make_agent("leaf", stages=[IkaStage("approve", "Get approval", [], hitl=True)])
    child = make_agent("child", topology=AgentTopology(subagents=(leaf,)))
    parent = make_agent("parent", topology=AgentTopology(subagents=(child,)))
    store = parent.enable_snapshots(str(tmp_path / "fork.db"))
    actions = iter([("child", {"input": "delegate"}), ("leaf", {"input": "approval"}),
                    ("ask_user", {"question": "Approve?"}), end("approved"), end("child done"), end("parent done")])
    with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))):
        paused = parent.execution()
        result = parent.fork_from_snapshot(paused["snapshot_id"], resume_input="YES", allow_unsafe_replay=True)
    assert result["final_message"] == "parent done"
    assert result["runtime"]["run_id"] != paused["runtime"]["run_id"]
    assert store.get_run(paused["runtime"]["run_id"])["status"] == "paused"


@pytest.mark.parametrize("error_type", [IkaRequestCancelled, IkaRequestDeadlineExceeded])
@pytest.mark.parametrize("use_async", [False, True])
def test_delegated_request_controls_propagate(make_agent, tmp_path, error_type, use_async):
    child = make_agent("child", "CHILD_TASK", policy=ExecutionPolicy.bounded(use_async=use_async))
    parent = make_agent("parent", topology=AgentTopology(subagents=(child,)),
                        policy=ExecutionPolicy.bounded(use_async=use_async))
    store = parent.enable_snapshots(str(tmp_path / "cancel.db"))
    calls = []
    def respond(url, payload):
        calls.append(payload)
        if len(calls) == 1:
            return provider_response(url, payload, ("child", {"input": "delegated work"}))
        if len(calls) == 2:
            raise error_type("child cancelled")
        return provider_response(url, payload, end("should not succeed"))
    with MockProvider(responder=respond), pytest.raises(error_type, match="child cancelled"):
        parent.execution()
    assert len(calls) == 2
    assert "child cancelled" not in json.dumps(parent.transcript)
    session = parent._snapshot_runtime
    assert store.get_run(session.run_id)["status"] == "cancelled"


@pytest.mark.parametrize("operation", ["save_short", "search_short", "save_long", "search_long", "filter"])
def test_memory_operations_propagate_request_controls(operation):
    from IkaMem.agent_memory import MemoryOperations

    class Backend:
        def save(self, *args, **kwargs):
            raise IkaRequestCancelled("memory cancelled")
        def search(self, *args, **kwargs):
            if operation == "filter":
                return []
            raise IkaRequestCancelled("memory cancelled")
    memory = MemoryOperations("test", Backend(), Backend())
    def cancel_filter(results):
        raise IkaRequestCancelled("memory cancelled")
    calls = {
        "save_short": lambda: memory.save_short_term("data"),
        "search_short": lambda: memory.search_short_term("query"),
        "save_long": lambda: memory.save_long_term({"task": "task", "output": "answer"}),
        "search_long": lambda: memory.search_long_term("query"),
        "filter": lambda: memory.search_long_term("query", filter_func=cancel_filter),
    }
    with pytest.raises(IkaRequestCancelled, match="memory cancelled"):
        calls[operation]()


def test_output_sink_propagates_request_controls():
    from IkaCore.cli_output import _route_to_sink, set_output_sink

    def cancel(ctx, rendered):
        raise IkaRequestCancelled("sink cancelled")
    set_output_sink(cancel)
    try:
        with pytest.raises(IkaRequestCancelled, match="sink cancelled"):
            _route_to_sink(None, "rendered")
    finally:
        set_output_sink(None)


@pytest.mark.parametrize("boundary", ["frame_entry", "pre_model", "post_model"])
@pytest.mark.parametrize("turn_number", [2, 3])
def test_fresh_restore_preserves_pending_later_chat_turn(make_agent, tmp_path, boundary, turn_number):
    agent = make_agent()
    db = str(tmp_path / "chat.db")
    agent.enable_snapshots(db)
    with MockProvider(responder=lambda u, p: provider_response(u, p, end("previous answer"))):
        agent.execution()
        if turn_number == 3:
            agent.send("SECOND_COMPLETED_TURN")
    message = f"TURN_{turn_number}_MUST_APPEAR_ONCE"
    agent.request_pause_at(boundary)
    with MockProvider(responder=lambda u, p: provider_response(u, p, end("saved answer"))):
        paused = agent.send(message)
    assert paused["status"] == "paused"
    fresh = make_agent()
    fresh.enable_snapshots(db)
    with MockProvider(responder=lambda u, p: provider_response(u, p, end("restored answer"))) as mock:
        result = fresh.resume_from_snapshot(paused["snapshot_id"])
    assert fresh.prompt == message
    assert result["final_message"] == ("saved answer" if boundary == "post_model" else "restored answer")
    inputs = [item["message"] for item in fresh.transcript["messages"].values() if item.get("type") == "user_input"]
    assert sum(message in item for item in inputs) == 1
    if boundary != "post_model":
        assert message in json.dumps(mock.requests[-1])
    else:
        assert not mock.requests


@pytest.mark.parametrize("outcome", ["success", "error", "cancel", "pause"])
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
def test_delegated_clones_close_only_owned_clients(make_agent, tmp_path, outcome, borrowed, use_async):
    from IkaModel.base import HumanInputRequired

    clients = []
    def factory():
        client = httpx.AsyncClient() if use_async else httpx.Client()
        clients.append(client)
        return client
    caller = factory() if borrowed else None
    provider = ProviderConfig.bounded(client=caller) if borrowed else ProviderConfig.bounded(client_factory=factory)
    child = make_agent("child", provider=provider, policy=ExecutionPolicy.bounded(use_async=use_async),
                       stages=[IkaStage("approval", "Get approval", [], hitl=True)] if outcome == "pause" else [])
    parent = make_agent("parent", topology=AgentTopology(subagents=(child,)))
    parent.enable_snapshots(str(tmp_path / "clients.db"))
    def respond(url, payload):
        if outcome == "error":
            raise RuntimeError("child failed")
        if outcome == "cancel":
            raise IkaRequestCancelled("child cancelled")
        return provider_response(url, payload, ("ask_user", {"question": "Approve?"}) if outcome == "pause" else end("done"))
    executor = parent.tool_runtime._build_subagent_executor(child)
    try:
        with MockProvider(responder=respond):
            if outcome in {"cancel", "pause"}:
                error = IkaRequestCancelled if outcome == "cancel" else HumanInputRequired
                with pytest.raises(error):
                    executor({"input": "task"})
            else:
                result = executor({"input": "task"})
                assert ("child failed" in result) if outcome == "error" else result == "done"
        assert not clients[0].is_closed
        if borrowed:
            assert len(clients) == 1
        else:
            assert len(clients) == 2 and clients[1].is_closed
        child.shutdown()
        assert clients[0].is_closed is not borrowed
    finally:
        if use_async:
            from IkaModel.async_runner import run_async
            for client in clients:
                run_async(client.aclose)
        else:
            for client in clients:
                client.close()


def test_delegated_result_replacement_preserves_unrelated_content():
    from IkaCore.snapshot_agent_state import _replace_interrupt

    pending = {"__ika_interrupt__": True, "delegated": {"snapshot_id": "child"}}
    assert _replace_interrupt({"functionResponse": {"response": pending}}, "child", "approved") == {
        "functionResponse": {"response": {"result": "approved"}}}
    for value in ["untrusted __ika_interrupt__ text", json.dumps({"other": "__ika_interrupt__"}),
                  json.dumps({"__ika_interrupt__": True, "delegated": {"snapshot_id": "other"}})]:
        assert _replace_interrupt(value, "child", "approved") == value


@pytest.mark.parametrize("invalid", ["missing_agent", "missing_snapshot", "other_run", "wrong_child", "missing_root", "wrong_parent"])
def test_delegated_resume_rejects_missing_or_foreign_child(make_agent, tmp_path, invalid):
    child = make_agent("child", stages=[IkaStage("approve", "Get approval", [], hitl=True)])
    parent = make_agent("parent", topology=AgentTopology(subagents=(child,)))
    store = parent.enable_snapshots(str(tmp_path / "invalid.db"))
    actions = iter([("child", {"input": "approval"}), ("ask_user", {"question": "Approve?"})])
    with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))):
        paused = parent.execution()
    saved = store.load_snapshot(paused["snapshot_id"])
    state = saved["state_json"]
    if invalid == "missing_agent":
        parent.subagents = []
    elif invalid == "missing_snapshot":
        state["interrupt_data"]["delegated"]["snapshot_id"] = "missing"
    elif invalid == "other_run":
        run = store.create_run("agent", "unrelated")
        frame = store.create_frame(run, "agent", "unrelated")
        foreign = store.create_snapshot(run, frame, "human_input", {})
        state["interrupt_data"]["delegated"]["snapshot_id"] = foreign
    else:
        child_snapshot = store.load_snapshot(state["interrupt_data"]["delegated"]["snapshot_id"])
        child_state = child_snapshot["state_json"]
        if invalid == "missing_root":
            child_state["root_frame_id"] = "missing"
        elif invalid == "wrong_child":
            child_state["root_frame_id"] = saved["frame_id"]
        else:
            frame = store.create_frame(saved["run_id"], "agent", "child")
            child_state["root_frame_id"] = frame
            child_snapshot["frame_id"] = frame
        bad_child = store.create_snapshot(saved["run_id"], child_snapshot["frame_id"], "human_input", child_state)
        state["interrupt_data"]["delegated"]["snapshot_id"] = bad_child
    snapshot = store.create_snapshot(saved["run_id"], saved["frame_id"], "human_input", state)
    with MockProvider() as mock, pytest.raises(ValueError, match=r"no longer configured|belong to the parent"):
        parent.resume_from_snapshot(snapshot, resume_input="YES")
    assert not mock.requests



def test_delegated_resume_without_tool_frame_identity(make_agent, tmp_path):
    child = make_agent("child", stages=[IkaStage("approve", "Get approval", [], hitl=True)])
    parent = make_agent("parent", topology=AgentTopology(subagents=(child,)))
    store = parent.enable_snapshots(str(tmp_path / "no-tool-frame.db"))
    actions = iter([("child", {"input": "approval"}), ("ask_user", {"question": "Approve?"}),
                    end("approved"), end("done")])
    with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))):
        paused = parent.execution()
        snapshot = store.load_snapshot(paused["snapshot_id"])
        state = snapshot["state_json"]
        state["interrupt_data"]["delegated"]["tool_frame_id"] = None
        historical = store.create_snapshot(snapshot["run_id"], snapshot["frame_id"], "human_input", state)
        result = parent.resume_from_snapshot(historical, resume_input="YES")
    assert result["final_message"] == "done"


@pytest.mark.parametrize("boundary", ["pre_model", "post_model"])
@pytest.mark.parametrize("staged", [False, True])
def test_restored_validation_correction_keeps_remaining_allowance(make_agent, tmp_path, boundary, staged):
    options = {"limits": ExecutionLimits.bounded(maxsteps=2, max_tool_rounds=1),
               "policy": ExecutionPolicy.bounded(final_answer_checks=(lambda result: False,),
                                                  validation_retries=1, validation_extra_steps=0),
               "stages": [IkaStage("answer", "Answer", [])] if staged else []}
    agent = make_agent(**options)
    db = str(tmp_path / "correction.db")
    agent.enable_snapshots(db)
    calls = []
    def respond(url, payload):
        if not payload.get("tools"):
            return provider_response(url, payload, "forced final")
        calls.append(payload)
        if len(calls) == 1:
            agent.request_pause_at(boundary)
            return provider_response(url, payload, end("initial"))
        return provider_response(url, payload, "partial correction")
    with MockProvider(responder=respond):
        paused = agent.execution()
        fresh = make_agent(**options)
        fresh.enable_snapshots(db)
        fresh.resume_from_snapshot(paused["snapshot_id"])
    assert len(calls) == 2

