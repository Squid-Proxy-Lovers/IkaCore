"""Cross-branch contracts: public turns, durable state, bounded execution and ownership."""
from __future__ import annotations

import asyncio
import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from IkaCore import (
    ExecutionLimits,
    ExecutionPolicy,
    IkaBaseAgent,
    IkaStage,
    IkaWorkflow,
    OutputConfig,
    ProviderConfig,
    RuntimeControl,
    WorkflowNode,
)
from IkaCore.testing import MockProvider, provider_response
from IkaModel import IkaRequestDeadlineExceeded, request_controls
from IkaModel.compaction import retained_compaction_entries


def agent(**kwargs):
    return IkaBaseAgent("merged", prompt="initial task", model_id="gpt-4o-mini", api_key="fixture",
                        output=OutputConfig(cli=False), **kwargs)


def end(text):
    return ("agent_end", {"input": text})


def test_latest_send_instruction_survives_compaction_before_old_pending_exchange():
    reasoning = {"type": "reasoning", "encrypted_content": "opaque"}
    call = {"type": "function_call", "call_id": "pending", "name": "lookup", "arguments": "{}"}
    history = {"messages": {
        "stale": {"type": "user_input", "message": "old task"},
        "latest": {"type": "user_input", "message": "preserve this constraint"},
        "noise": {"type": "assistant", "message": "noise"},
        "reasoning": {"type": "tool", "message": json.dumps(reasoning)},
        "call": {"type": "assistant_with_tools", "message": json.dumps(call)},
    }}
    saved = copy.deepcopy(history)
    retained = retained_compaction_entries(history)
    assert list(retained) == ["latest", "reasoning", "call"]
    assert history == saved


@pytest.mark.parametrize("parallel", [False, True])
def test_borrowed_client_config_reaches_deadline_private_pool_through_clone(parallel):
    observed = []
    def respond(request):
        observed.append(request)
        return provider_response(str(request.url), json.loads(request.content), end("done"))
    with httpx.Client(transport=httpx.MockTransport(respond), cookies={"fixture": "retained"},
                      headers={"X-Caller": "preserved"}, auth=("user", "password")) as caller:
        a = agent(provider=ProviderConfig(client=caller))
        wf = IkaWorkflow("client", "ownership", [WorkflowNode("node", a)], [], output=OutputConfig(cli=False))
        try:
            with request_controls(timeout=2, max_retries=1):
                result = wf.run(parallel=parallel)["node"]
            assert result.output["final_message"] == "done"
            assert len(observed) == 1
            request = observed[0]
            assert request.headers["X-Caller"] == "preserved"
            assert request.headers["cookie"] == "fixture=retained"
            assert request.headers["authorization"].startswith("Basic ")
            assert not caller.is_closed
        finally:
            wf.async_executor.shutdown()
            a.shutdown()
        assert not caller.is_closed


def test_deadline_returns_without_closing_borrowed_client_or_late_cleanup():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def respond(request):
        entered.set()
        release.wait(2)
        finished.set()
        return provider_response(str(request.url), {}, end("late"))
    with httpx.Client(transport=httpx.MockTransport(respond)) as caller:
        a = agent(provider=ProviderConfig(client=caller))
        try:
            with request_controls(timeout=0.1, max_retries=1), pytest.raises(IkaRequestDeadlineExceeded):
                a.execution()
            assert entered.is_set() and not caller.is_closed
        finally:
            release.set()
            assert finished.wait(2)
            a.shutdown()
        assert not caller.is_closed


def test_lowercase_stages_and_removed_alias():
    stage = IkaStage("answer", "answer", [])
    a = agent(stages=[stage])
    try:
        assert a.stages == [stage] and not hasattr(a, "Stages")
        a.stages = []
        assert a.stages == []
        with pytest.raises(TypeError, match="Stages"):
            agent(Stages=[])
    finally:
        a.shutdown()


def test_owned_async_client_is_closed_and_borrowed_async_client_is_retained():
    owned = SimpleNamespace(aclose=AsyncMock())
    a = agent(provider=ProviderConfig(client_factory=lambda: owned))
    a.shutdown()
    owned.aclose.assert_awaited_once()
    caller = httpx.AsyncClient()
    b = agent(provider=ProviderConfig(client=caller))
    b.shutdown()
    assert not caller.is_closed
    asyncio.run(caller.aclose())


@pytest.mark.parametrize("state", ["interrupt", "resume", "paused_frame"])
def test_send_rejects_all_forms_of_pending_turn(tmp_path, state):
    a = agent()
    try:
        if state == "paused_frame":
            store = a.enable_snapshots(str(tmp_path / "state.db"))
            run = store.create_run("agent", a.name)
            frame = store.create_frame(run, "agent", a.name)
            a._snapshot_runtime.root_frame_id = frame
            store.update_frame(frame, status="paused")
        elif state == "interrupt":
            a.state._last_interrupt = {"status": "awaiting_user_input"}
        else:
            a.state._resume_state = {"scope": "agent"}
        with pytest.raises(ValueError, match="pending turn"):
            a.send("new")
    finally:
        a.shutdown()


@pytest.mark.parametrize("message", ["", "  ", None, 3])
def test_send_rejects_empty_or_nontext_turns(message):
    a = agent()
    try:
        with pytest.raises(ValueError, match="nonempty"):
            a.send(message)
        assert a.prompt == "initial task" and not a.transcript["messages"]
    finally:
        a.shutdown()


def test_validation_waits_for_staged_human_answer(tmp_path):
    checked = []
    def valid(result):
        checked.append(result["final_message"])
        return result["final_message"] == "approved"
    a = agent(stages=[IkaStage("approve", "ask before answering", [], hitl=True)],
              policy=ExecutionPolicy.bounded(final_answer_checks=(valid,)))
    a.enable_snapshots(str(tmp_path / "human.db"))
    try:
        with MockProvider(responder=lambda url, payload: provider_response(url, payload, ("ask_user", {"question": "Approve?"}))):
            waiting = a.execution()
        assert waiting["status"] == "awaiting_user_input"
        assert waiting["validation"]["status"] == "not_run" and checked == []
        with MockProvider(responder=lambda url, payload: provider_response(url, payload, end("approved"))):
            result = a.fork_from_snapshot(waiting["snapshot_id"], resume_input="YES")
        assert result["validation"]["status"] == "passed" and checked == ["approved"]
    finally:
        a.shutdown()


def test_fork_helpers_require_persistence_before_writing():
    a = agent()
    wf = IkaWorkflow("fork", "test", [WorkflowNode("node", a)], [], output=OutputConfig(cli=False))
    try:
        with pytest.raises(ValueError, match="enable_snapshots"):
            a.fork_from_snapshot("missing")
        with pytest.raises(ValueError, match="enable_snapshots"):
            wf.fork_from_snapshot("missing")
    finally:
        wf.async_executor.shutdown()
        a.shutdown()


def test_validation_rejects_corrupt_saved_retry_counter_and_restores_budget():
    a = agent(limits=ExecutionLimits(maxsteps=2),
              policy=ExecutionPolicy.bounded(final_answer_checks=(lambda result: True,)))
    a.state.validation_attempts = 4
    try:
        with pytest.raises(RuntimeError, match="Invalid validation retry state"):
            a.execution()
        assert a.maxsteps == 2 and a.state.validation_attempts == 0
    finally:
        a.shutdown()


def test_workflow_fork_retains_output_and_transcript_for_each_instance(tmp_path):
    a = agent()
    wf = IkaWorkflow("fork", "test", [WorkflowNode("node", a, instances=2)], [], output=OutputConfig(cli=False))
    wf.enable_snapshots(str(tmp_path / "fork.db"))
    try:
        with MockProvider(responder=lambda url, payload: provider_response(url, payload, end("answer"))):
            first = wf.run(parallel=True)["node"]
        snapshot = wf.snapshot_runtime_info()["last_snapshot_id"]
        with MockProvider() as mock:
            forked = wf.fork_from_snapshot(snapshot)["node"]
        assert not mock.requests
        assert forked.instance_outputs == first.instance_outputs
        assert forked.instance_transcripts == first.instance_transcripts
        assert set(forked.instance_outputs) == {0, 1}
    finally:
        wf.async_executor.shutdown()
        a.shutdown()


@pytest.mark.parametrize("parallel", [False, True])
def test_worker_cancellation_preserves_type_and_cancels_other_inflight_work(parallel):
    from IkaModel import IkaRequestCancelled
    a = agent()
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    calls = 0
    def work():
        nonlocal calls
        with lock:
            index = calls
            calls += 1
        if index == 0:
            if parallel:
                assert entered.wait(2)
            raise IkaRequestCancelled("cancel from worker")
        entered.set()
        release.wait(2)
        return {"final_message": "late"}
    a.execution = work
    wf = IkaWorkflow("cancel", "workers", [WorkflowNode("node", a, instances=2)], [],
                     output=OutputConfig(cli=False))
    try:
        with pytest.raises(IkaRequestCancelled, match="cancel from worker"):
            wf.run(parallel=parallel)
        assert not wf._results
    finally:
        release.set()
        wf.async_executor.shutdown()
        a.shutdown()


def test_parallel_workflow_and_removed_aliases():
    a = agent()
    wf = IkaWorkflow("alias", "test", [WorkflowNode("node", a)], [], output=OutputConfig(cli=False))
    try:
        with MockProvider(responder=lambda url, payload: provider_response(url, payload, end("answer"))):
            result = wf.run(parallel=True)["node"]
        assert result.output["final_message"] == "answer"
        assert not hasattr(result, "history") and not hasattr(wf, "run_async")
        with pytest.raises(TypeError, match="use_async"):
            wf.run(use_async=True)
    finally:
        wf.async_executor.shutdown()
        a.shutdown()


def test_failed_child_keeps_original_interruption_for_workflow_retry(tmp_path):
    from IkaCore.snapshot_store import SnapshotStore
    from IkaCore.workflow_snapshot_state import WorkflowSnapshotSession
    from IkaCore.workflow_snapshots import _reuse_resumed_children
    store = SnapshotStore(str(tmp_path / "failed-child.db"))
    run = store.create_run("workflow", "parent")
    root = store.create_frame(run, "workflow", "parent")
    child = store.create_frame(run, "agent", "child", parent_frame_id=root)
    snapshot = store.create_snapshot(run, child, "human_input", {"root_frame_id": child})
    store.update_frame(child, status="failed")
    session = WorkflowSnapshotSession(None, store, RuntimeControl(), run_id=run, root_frame_id=root,
                                      interrupted_instances={"node:0": snapshot})
    _reuse_resumed_children(session)
    assert session.interrupted_instances == {"node:0": snapshot}
    assert not session.replay_cache


@pytest.mark.parametrize("entry_type", ["user_input", "stage_input", "hitl_input"])
def test_compaction_estimate_counts_mirrored_first_input_once(entry_type):
    from IkaModel.chat_interface.chat_request import get_total_tokens
    history = {"system": {"message": "system", "tokens": 2},
               "first_input": {"message": "opening turn", "tokens": 3},
               "summary": {"message": "summary", "tokens": 4},
               "messages": {
                   "opening": {"type": entry_type, "message": "opening turn", "tokens": 3},
                   "next": {"type": "user_input", "message": "next turn", "tokens": 5},
               }}
    assert get_total_tokens(history) == 14
    # Different input text and assistant text are not duplicates of the user view.
    history["first_input"]["message"] = "different instruction"
    assert get_total_tokens(history) == 17
    history["first_input"]["message"] = "opening turn"
    history["messages"]["opening"]["type"] = "assistant"
    assert get_total_tokens(history) == 17


def test_compaction_events_follow_agent_scope_and_workflow_instance():
    from IkaCore import RuntimeOutput, output_scope
    outer_events, agent_events = [], []
    a = IkaBaseAgent("scoped", prompt="task", model_id="gpt-4o-mini", api_key="fixture",
                     limits=ExecutionLimits.bounded(context_budget=1),
                     output=OutputConfig(cli=False, subscribers=(agent_events.append,)))
    wf = IkaWorkflow("events", "test", [WorkflowNode("node", a, instances=2)], [],
                     output=OutputConfig(cli=False))
    def respond(url, payload):
        return provider_response(url, payload, end("answer") if payload.get("tools") else "summary")
    try:
        with output_scope(RuntimeOutput((outer_events.append,))), MockProvider(responder=respond):
            wf.run(parallel=True)
        compactions = [event for event in agent_events if event.kind == "compaction"]
        assert {event.instance_id for event in compactions} == {0, 1}
        assert all(event.data["count"] == 1 for event in compactions)
        assert not any(event.kind == "compaction" for event in outer_events)
        assert a.compaction_count == 0  # Worker clone counters do not mutate the template.
    finally:
        wf.async_executor.shutdown()
        a.shutdown()


@pytest.mark.parametrize("entry_type", [None, [], {}])
def test_compaction_estimate_tolerates_malformed_history_entry_types(entry_type):
    from IkaModel.chat_interface.chat_request import get_total_tokens
    history = {"first_input": {"message": "opening", "tokens": 3},
               "messages": {"entry": {"type": entry_type, "message": "opening", "tokens": 3}}}
    assert get_total_tokens(history) == 6
