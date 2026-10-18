"""Public contracts replacing the application dogfood workarounds."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest

from IkaCore import (
    ExecutionLimits,
    ExecutionPolicy,
    FinalAnswerValidationError,
    IkaBaseAgent,
    IkaStage,
    IkaTools,
    IkaWorkflow,
    OutputConfig,
    ProviderConfig,
    RuntimeControl,
    WorkflowEdge,
    WorkflowError,
    WorkflowNode,
)
from IkaCore.testing import MockProvider, provider_response
from IkaModel import request_controls, runtime_options
from IkaModel.request_interface import async_api_request_retry


def helper(**kwargs):
    return IkaBaseAgent("test", prompt="INITIAL_TASK", model_id="gpt-4o-mini", api_key="fixture",
        output=OutputConfig(cli=False), provider=kwargs.pop("provider", ProviderConfig.bounded()),
        limits=kwargs.pop("limits", ExecutionLimits.bounded()), policy=kwargs.pop("policy", ExecutionPolicy.bounded()), **kwargs)


def scripted(actions):
    responses = iter(actions)
    return MockProvider(responder=lambda url, payload: provider_response(url, payload, next(responses)))


def end(text):
    return ("agent_end", {"input": text})


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("fail_closed", [False, True])
def test_exhausted_checks_have_explicit_status_or_raise(staged, fail_closed):
    checked = []
    def reject(result):
        checked.append(result["final_message"])
        assert {"final_message", "summary", "usage", "cost", "model_id", "validation"} <= result.keys()
        return False
    a = helper(stages=[IkaStage("answer", "Answer", [], stage_max_step=1)] if staged else [],
               policy=ExecutionPolicy.bounded(final_answer_checks=(reject,), fail_closed=fail_closed))
    with scripted([end("bad"), end("still bad")]) as mock:
        if fail_closed:
            with pytest.raises(FinalAnswerValidationError) as caught:
                a.execution()
            result = caught.value.result
        else:
            result = a.execution()
    assert checked == ["bad", "still bad"]
    assert result["validation"] == {"status": "failed", "attempts": 2, "failed_checks": ["reject"]}
    assert len(mock.requests) == 2
    assert a.maxsteps == 6
    assert a.prompt == "INITIAL_TASK"
    a.shutdown()


def test_staged_correction_preserves_evidence_without_replaying_tools():
    effects = []
    tool = IkaTools("fetch", "Fetch evidence", {"query": {"type": "string"}}, execute_function=lambda _: effects.append("read") or "FACT")
    def valid(result):
        return result["final_message"] == "correct FACT"
    a = helper(stages=[IkaStage("research", "Fetch evidence then answer", [tool], stage_max_step=3)],
               policy=ExecutionPolicy.bounded(final_answer_checks=(valid,), fail_closed=True))
    with scripted([("fetch", {"query": "evidence"}), end("bad FACT"), end("correct FACT")]) as mock:
        result = a.execution()
    assert effects == ["read"]
    assert result["validation"]["status"] == "passed" and result["validation"]["attempts"] == 2
    last = json.dumps(mock.requests[-1])
    assert "FACT" in last and "valid" in last
    assert [t["name"] for t in mock.requests[-1]["tools"]] == ["agent_end"]
    a.shutdown()


@pytest.mark.parametrize("check", [lambda: True, lambda a, b: True])
def test_validator_signature_rejected_without_probe(check):
    with pytest.raises(ValueError, match="one positional AgentResult"):
        helper(policy=ExecutionPolicy.bounded(final_answer_checks=(check,)))


def test_async_validator_rejected_without_coroutine_creation():
    async def check(result):
        return True
    with pytest.raises(ValueError, match="synchronous"):
        helper(policy=ExecutionPolicy.bounded(final_answer_checks=(check,)))


@pytest.mark.parametrize("model,url,responses", [
    ("gpt-4o-mini", "https://api.openai.com/v1/responses", True),
    ("gpt-4o-mini", "https://api.openai.com/v1/chat/completions", False),
    ("gpt-5.6-sol", "https://chatgpt.com/backend-api/codex/responses", True),
    ("claude-sonnet-4-5", "https://api.anthropic.com/v1/messages", True),
    ("gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent", True),
    ("deepseek-chat", "https://api.deepseek.com/chat/completions", True),
    ("openrouter/auto", "https://openrouter.ai/api/v1/chat/completions", True),
])
def test_send_replays_each_user_turn_once_in_order(model, url, responses):
    a = IkaBaseAgent("chat", prompt="FIRST_UNIQUE", model_id=model, api_key="fixture", output=OutputConfig(cli=False),
                     provider=ProviderConfig(api_url=url, use_responses_api=responses))
    with scripted([end("first"), end("second"), end("third")]) as mock:
        a.execution()
        a.send("SECOND_UNIQUE")
        a.send("THIRD_UNIQUE")
    wire = json.dumps(mock.requests[-1])
    assert wire.count("FIRST_UNIQUE") == 1
    assert wire.count("SECOND_UNIQUE") == 1
    assert wire.count("THIRD_UNIQUE") == 1
    assert wire.index("FIRST_UNIQUE") < wire.index("SECOND_UNIQUE") < wire.index("THIRD_UNIQUE")
    assert any(e["type"] == "user_input" for e in a.transcript["messages"].values())
    copied = a.transcript
    copied["messages"].clear()
    assert a.transcript["messages"]
    a.shutdown()


def test_completed_snapshot_restores_conversation_before_send(tmp_path):
    a = helper()
    store = a.enable_snapshots(str(tmp_path / "chat.db"))
    with scripted([end("first"), end("second")]):
        a.execution()
        result = a.send("SECOND_UNIQUE")
    snapshot = result["runtime"]["last_snapshot_id"]
    expected = a.transcript
    a.shutdown()
    b = helper()
    b.enable_snapshots(store.db_path)
    with MockProvider() as no_requests:
        restored = b.resume_from_snapshot(snapshot)
    assert not no_requests.requests and restored["final_message"] == "second"
    assert b.transcript == expected
    with pytest.raises(ValueError, match="completed snapshot"):
        b.resume_from_snapshot(snapshot, resume_input="mistaken answer")
    with scripted([end("third")]) as mock:
        b.send("THIRD_UNIQUE")
    assert "SECOND_UNIQUE" in json.dumps(mock.requests[0])
    b.shutdown()


def test_pending_snapshot_must_resume_before_next_turn(tmp_path):
    a = helper()
    a.enable_snapshots(str(tmp_path / "pending.db"), RuntimeControl(pause_points={"post_model"}))
    with scripted([end("first")]):
        paused = a.send("PENDING_UNIQUE")
    with pytest.raises(ValueError, match="pending turn"):
        a.send("too early")
    a.clear_pause_requests()
    with MockProvider():
        assert a.resume_from_snapshot(paused["snapshot_id"])["final_message"] == "first"
    with scripted([end("second")]):
        assert a.send("next")["final_message"] == "second"
    a.shutdown()


def test_context_budget_events_and_counter_survive_snapshot(tmp_path):
    observed = []
    a = IkaBaseAgent("chat", prompt="task", model_id="gpt-4o-mini", api_key="fixture",
                     limits=ExecutionLimits.bounded(context_budget=1),
                     output=OutputConfig(cli=False, subscribers=(observed.append,)))
    a.enable_snapshots(str(tmp_path / "compact.db"))
    def respond(url, payload):
        return provider_response(url, payload, end("done") if payload.get("tools") else "Remember task")
    with MockProvider(responder=respond), runtime_options(compact_at_fraction=0.5):
        a.execution()
        result = a.send("New task")
    events = [e for e in observed if e.kind == "compaction"]
    assert a.compaction_count >= 1 and len(events) == a.compaction_count
    assert events[-1].data["count"] == a.compaction_count
    b = helper()
    b.enable_snapshots(str(tmp_path / "compact.db"))
    b.resume_from_snapshot(result["runtime"]["last_snapshot_id"])
    assert b.compaction_count == a.compaction_count
    a.shutdown()
    b.shutdown()


def test_client_ownership_and_clone_lifetime():
    borrowed = httpx.Client()
    a = helper(provider=ProviderConfig(client=borrowed))
    clone = a.clone_for_run()
    assert clone.client is borrowed
    clone.shutdown()
    a.shutdown()
    assert not borrowed.is_closed
    borrowed.close()
    owned = []
    def factory():
        owned.append(httpx.Client())
        return owned[-1]
    a = helper(provider=ProviderConfig(client_factory=factory))
    clone = a.clone_for_run()
    assert len(owned) == 2 and clone.client is not a.client
    clone.shutdown()
    assert owned[1].is_closed and not owned[0].is_closed
    a.shutdown()
    assert owned[0].is_closed
    with pytest.raises(ValueError, match="borrowed"):
        ProviderConfig(client=object(), client_factory=factory)


@pytest.mark.parametrize("parallel", [False, True])
def test_workflow_error_identifies_instance_and_blocks_dependents(parallel):
    a, b, c = helper(), helper(), helper()
    a.execution = Mock(return_value={"final_message": "seed", "summary": "seed"})
    b.execution = Mock(side_effect=ValueError("worker broke"))
    c.execution = Mock()
    wf = IkaWorkflow("wf", "test", [WorkflowNode("a", a), WorkflowNode("b", b), WorkflowNode("c", c)],
                     [WorkflowEdge("a", "b"), WorkflowEdge("b", "c")],
                     compress_hook=lambda contexts, target: " ".join(contexts), output=OutputConfig(cli=False))
    try:
        with pytest.raises(WorkflowError, match=r"b.*worker broke") as caught:
            wf.run(parallel=parallel)
        error = caught.value.errors[0]
        assert (error.node_name, error.instance_id, error.error_type) == ("b", 0, "ValueError")
        assert set(caught.value.partial_results) == {"a"}
        c.execution.assert_not_called()
    finally:
        wf.async_executor.shutdown()
        for item in (a, b, c):
            item.shutdown()


@pytest.mark.parametrize("parallel", [False, True])
def test_workflow_output_is_separate_from_transcript_and_replay(tmp_path, parallel):
    a = helper()
    wf = IkaWorkflow("wf", "test", [WorkflowNode("node", a)], [], output=OutputConfig(cli=False))
    wf.enable_snapshots(str(tmp_path / "wf.db"))
    try:
        with scripted([end("answer")]):
            results = wf.run(parallel=parallel)
        result = results["node"]
        assert result.output["final_message"] == result.final == "answer"
        assert "INITIAL_TASK" in result.transcript["first_input"]["message"]
        assert result.instance_transcripts[0] == result.transcript
        saved = wf.snapshot_runtime_info()["last_snapshot_id"]
        with MockProvider():
            replay = wf.resume_from_snapshot(saved)["node"]
        assert replay.transcript == result.transcript
    finally:
        wf.async_executor.shutdown()
        a.shutdown()


def test_mock_provider_native_envelopes_nested_scopes_and_async():
    envelope = {"output": [], "usage": {}}
    with MockProvider([envelope]) as outer:
        with MockProvider([envelope]) as inner:
            response = asyncio.run(async_api_request_retry("https://api.openai.com/v1/responses", {}, {"model": "x"}))
            assert response.json() == envelope and len(inner.requests) == 1
        assert not outer.requests
        asyncio.run(async_api_request_retry("https://api.openai.com/v1/responses", {}, {}))
        with pytest.raises(AssertionError, match="exhausted"):
            outer.request("https://fixture.invalid", {})
    with pytest.raises(AssertionError, match="not consumed"), MockProvider([envelope]):
        pass
    with pytest.raises(RuntimeError, match="twice"), MockProvider() as mock, mock:
        pass
    with pytest.raises(ValueError, match="Choose"):
        MockProvider([envelope], responder=lambda url, payload: envelope)


def test_bounded_presets_and_context_budget_units():
    assert ProviderConfig.bounded().max_tokens == 1024
    assert ExecutionLimits.bounded().maxsteps == 6
    assert ExecutionPolicy.bounded().max_step_extensions == ExecutionPolicy.bounded().max_stage_extensions == 0
    assert ExecutionPolicy.bounded().validation_retries == 1
    with pytest.raises(ValueError, match="positive token"):
        ExecutionLimits(context_budget=0)
    with pytest.raises(ValueError, match="nonnegative"):
        ExecutionPolicy(validation_retries=-1)
    with request_controls(timeout=0), MockProvider():
        from IkaModel import IkaRequestDeadlineExceeded
        with pytest.raises(IkaRequestDeadlineExceeded):
            helper().execution()


@pytest.mark.parametrize("staged", [False, True])
def test_validation_correction_pause_restores_retry_budget(tmp_path, staged):
    checked = []
    def valid(result):
        checked.append(result["final_message"])
        return result["final_message"] == "good"
    a = helper(stages=[IkaStage("answer", "Answer", [], stage_max_step=1)] if staged else [],
               policy=ExecutionPolicy.bounded(final_answer_checks=(valid,)))
    a.enable_snapshots(str(tmp_path / "validation.db"))
    def respond(url, payload):
        # Pause after generating the correction, before its validator runs.
        if "Correct your previous final answer" in json.dumps(payload):
            a.request_pause_at("post_model")
            return provider_response(url, payload, end("good"))
        return provider_response(url, payload, end("bad"))
    with MockProvider(responder=respond):
        paused = a.execution()
    assert paused["status"] == "paused" and checked == ["bad"]
    a.clear_pause_requests()
    with MockProvider():
        result = a.resume_from_snapshot(paused["snapshot_id"])
    assert result["final_message"] == "good"
    assert checked == ["bad", "good"] and result["validation"]["attempts"] == 2
    assert a.maxsteps == 6
    a.shutdown()


def test_direct_client_assignment_is_borrowed_and_cloned():
    a = helper()
    with httpx.Client() as caller:
        a.client = caller
        clone = a.clone_for_run()
        assert clone.client is caller
        clone.shutdown()
        a.shutdown()
        assert not caller.is_closed


def test_prompt_assignment_replaces_restored_and_workflow_state(tmp_path):
    a = helper()
    a.inject_workflow_context("OLD_UPSTREAM")
    a.enable_snapshots(str(tmp_path / "replace.db"))
    with scripted([end("old")]):
        result = a.execution()
    with MockProvider():
        a.resume_from_snapshot(result["runtime"]["last_snapshot_id"])
    a.prompt = "FRESH_TASK"
    assert not a.transcript["messages"]
    with scripted([end("fresh")]) as mock:
        a.execution()
    payload = json.dumps(mock.requests[0])
    assert "FRESH_TASK" in payload and "OLD_UPSTREAM" not in payload and "INITIAL_TASK" not in payload
    a.shutdown()


def test_workflow_concurrency_bounds_active_instances_and_validates_aliases():
    import threading
    import time
    lock = threading.Lock()
    active = peak = 0
    a = helper()
    def work():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {"final_message": "done"}
    a.execution = work
    wf = IkaWorkflow("wf", "bounded workers", [WorkflowNode("node", a, instances=3)], [], output=OutputConfig(cli=False))
    try:
        for _ in range(2):
            result = wf.run(concurrency=2)["node"]
            assert len(result.instance_outputs) == 3
            assert result.history is result.output  # Deprecated output alias remains usable.
        assert peak == 2
        with pytest.raises(ValueError, match="Conflicting"):
            wf.run(parallel=True, use_async=False)
        with pytest.raises(ValueError, match="positive"):
            wf.run(concurrency=0)
        from IkaCore.workflow import AsyncWorkflowExecutor
        caller = AsyncWorkflowExecutor(max_workers=1)
        other = IkaWorkflow("wf", "borrowed executor", [WorkflowNode("node", a)], [], async_executor=caller,
                            output=OutputConfig(cli=False))
        try:
            with pytest.raises(ValueError, match="caller-owned"):
                other.run(concurrency=2)
            assert caller.executor.submit(lambda: "still usable").result() == "still usable"
        finally:
            caller.shutdown()
    finally:
        wf.async_executor.shutdown()
        a.shutdown()


@pytest.mark.parametrize("cancel", [False, True])
def test_workflow_context_worker_errors_have_node_identity_and_cancellation_type(cancel):
    from IkaModel import IkaRequestCancelled
    a, b = helper(), helper()
    a.execution = Mock(return_value={"final_message": "upstream"})
    b.execution = Mock()
    error = IkaRequestCancelled("stop") if cancel else ValueError("bad context")
    def compress(contexts, target):
        raise error
    wf = IkaWorkflow("wf", "context errors", [WorkflowNode("a", a), WorkflowNode("b", b)], [WorkflowEdge("a", "b")],
                     compress_hook=compress, output=OutputConfig(cli=False))
    try:
        with pytest.raises(IkaRequestCancelled if cancel else WorkflowError) as caught:
            wf.run(parallel=True)
        if not cancel:
            detail = caught.value.errors[0]
            assert detail.node_name == "b" and detail.instance_id is None and detail.message == "bad context"
            assert set(caught.value.partial_results) == {"a"}
        b.execution.assert_not_called()
    finally:
        wf.async_executor.shutdown()
        a.shutdown()
        b.shutdown()
