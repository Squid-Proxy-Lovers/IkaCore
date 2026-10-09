import threading
from unittest.mock import MagicMock, patch

from IkaCore.agents import IkaBaseAgent
from IkaCore.workflow import AsyncWorkflowExecutor, IkaWorkflow, WorkflowEdge, WorkflowNode


def _agent(name):
    return IkaBaseAgent(name=name, description=f"{name} desc", prompt=f"{name} prompt", model_id="gpt-4o", api_key="k")


def _output(name):
    return {"final_message": f"{name} final", "summary": f"{name} summary"}


def _clone_with(agent, execution):
    agent.execution = execution  # sync runs call the node's own agent; async runs use clones

    def clone_for_run(**_kwargs):
        clone = _agent(agent.name)
        clone.execution = execution
        return clone

    agent.clone_for_run = clone_for_run
    return agent


def test_dependent_starts_when_its_own_dependencies_finish_not_when_the_wave_does():
    # root -> a1 -> a2 and root -> slow. slow only finishes once a2 has started, which is impossible
    # if a2 has to wait for every node running alongside a1 (i.e. for slow) to finish first.
    a2_started = threading.Event()
    slow_saw_a2 = []

    def slow_execution():
        slow_saw_a2.append(a2_started.wait(timeout=5))
        return _output("slow")

    def a2_execution():
        a2_started.set()
        return _output("a2")

    agents = {
        "root": _clone_with(_agent("root"), lambda: _output("root")),
        "a1": _clone_with(_agent("a1"), lambda: _output("a1")),
        "a2": _clone_with(_agent("a2"), a2_execution),
        "slow": _clone_with(_agent("slow"), slow_execution),
    }
    workflow = IkaWorkflow(
        name="dataflow",
        description="d",
        nodes=[WorkflowNode(name, agent) for name, agent in agents.items()],
        edges=[WorkflowEdge("root", "a1"), WorkflowEdge("a1", "a2"), WorkflowEdge("root", "slow")],
        compress_hook=lambda contexts, agent: "\n".join(contexts),
        async_executor=AsyncWorkflowExecutor(max_workers=4),
    )
    with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
        results = workflow.run(initial_context="go", use_async=True)

    assert slow_saw_a2 == [True]
    assert set(results) == {"root", "a1", "a2", "slow"}


def test_context_is_prepared_once_per_node_and_shared_by_its_instances():
    calls = []

    def compress(contexts, agent):
        calls.append(tuple(contexts))
        return " | ".join(contexts)

    root = _clone_with(_agent("root"), lambda: _output("root"))
    seen_contexts = []

    def fan_clone(**_kwargs):
        clone = _agent("fan")
        original = clone.inject_workflow_context

        def record(context):
            seen_contexts.append(context)
            original(context)

        clone.inject_workflow_context = record
        clone.execution = lambda: _output("fan")
        return clone

    fan = _agent("fan")
    fan.clone_for_run = fan_clone
    workflow = IkaWorkflow(
        name="instances",
        description="d",
        nodes=[WorkflowNode("root", root), WorkflowNode("fan", fan, instances=4)],
        edges=[WorkflowEdge("root", "fan")],
        compress_hook=compress,
        async_executor=AsyncWorkflowExecutor(max_workers=4),
    )
    with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
        results = workflow.run(initial_context="go", use_async=True)

    assert calls.count(("root summary",)) == 1  # not once per instance
    assert seen_contexts == ["root summary"] * 4
    assert results["fan"].summary.count("fan summary") == 4


# ---------------------------------------------------------------------------
# Summary-call reductions: initial context passthrough, one summary per edge, shared identical summaries
# ---------------------------------------------------------------------------


def _chain_workflow(compress_hook=None, async_executor=None):
    root = _clone_with(_agent("root"), lambda: _output("root"))
    follower = _clone_with(_agent("follower"), lambda: _output("follower"))
    kwargs = {"compress_hook": compress_hook} if compress_hook else {}
    if async_executor:
        kwargs["async_executor"] = async_executor
    return IkaWorkflow(name="chain", description="d", nodes=[WorkflowNode("root", root), WorkflowNode("follower", follower)],
                       edges=[WorkflowEdge("root", "follower")], **kwargs)


def test_initial_context_reaches_start_node_verbatim_without_summarising():
    calls = []

    def compress(contexts, agent):
        calls.append(list(contexts))
        return "SUMMARY(" + " | ".join(contexts) + ")"

    for use_async in (False, True):
        calls.clear()
        workflow = _chain_workflow(compress, AsyncWorkflowExecutor(max_workers=2) if use_async else None)
        with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
            workflow.run(initial_context="the user's own brief", use_async=use_async)
        assert calls == [["root summary"]], (use_async, calls)  # only the real edge is summarised, once


def test_sync_edges_are_summarised_once_at_the_consumer():
    seen = []
    root = _agent("root")
    root.execution = MagicMock(return_value=_output("root"))
    follower = _agent("follower")
    follower.execution = MagicMock(return_value=_output("follower"))
    follower.inject_workflow_context = seen.append
    calls = []
    workflow = IkaWorkflow(
        name="sync", description="d", nodes=[WorkflowNode("root", root), WorkflowNode("follower", follower)],
        edges=[WorkflowEdge("root", "follower")],
        compress_hook=lambda contexts, agent: calls.append(list(contexts)) or "S(" + ",".join(contexts) + ")",
    )
    workflow.run()
    assert calls == [["root summary"]]  # previously: [["root summary"]] from the producer, then [["S(root summary)"]]
    assert seen == ["S(root summary)"]


def test_default_summaries_are_shared_only_between_identical_requests():
    from IkaCore import workflow_core

    def build(key_for_child):
        root = _clone_with(_agent("root"), lambda: _output("root"))
        kids = []
        for i in range(3):
            kid = _agent(f"kid{i}")
            kid.api_key = key_for_child(i)
            kids.append(_clone_with(kid, lambda i=i: _output(f"kid{i}")))
        return IkaWorkflow(
            name="fan", description="d",
            nodes=[WorkflowNode("root", root)] + [WorkflowNode(k.name, k) for k in kids],
            edges=[WorkflowEdge("root", k.name) for k in kids],
            async_executor=AsyncWorkflowExecutor(max_workers=4),
            summarize_context_above_tokens=None,  # always summarise, so the sharing path is exercised
        )

    for key_for_child, expected in ((lambda i: "same-key", 1), (lambda i: f"key-{i}", 3)):
        with patch.object(workflow_core, "summarise_message_history", return_value="shared") as summarise, \
             patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
            build(key_for_child).run(use_async=True)
        assert summarise.call_count == expected


def test_custom_compress_hooks_are_never_shared():
    calls = []
    root = _clone_with(_agent("root"), lambda: _output("root"))
    kids = [_clone_with(_agent(f"kid{i}"), lambda i=i: _output(f"kid{i}")) for i in range(3)]
    workflow = IkaWorkflow(
        name="fan", description="d", nodes=[WorkflowNode("root", root)] + [WorkflowNode(k.name, k) for k in kids],
        edges=[WorkflowEdge("root", k.name) for k in kids],
        compress_hook=lambda contexts, agent: calls.append(agent.name) or "x",
        async_executor=AsyncWorkflowExecutor(max_workers=4),
    )
    with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
        workflow.run(use_async=True)
    assert sorted(calls) == ["kid0", "kid1", "kid2"]


def test_default_hook_passes_short_context_through_and_summarises_long_context():
    from IkaCore import workflow_core

    agent = _agent("agent")

    def hook_result(threshold, contexts):
        workflow = IkaWorkflow(name="t", description="d", nodes=[WorkflowNode("agent", agent)], edges=[],
                               summarize_context_above_tokens=threshold)
        with patch.object(workflow_core, "summarise_message_history", return_value="SUMMARY") as summarise:
            return workflow._default_compress_hook(contexts, agent), summarise.call_count

    short, long_text = ["alpha", "beta"], ["x" * 4400]
    assert hook_result(1000, short) == ("alpha\n\nbeta", 0)  # under threshold: verbatim, no model call
    assert hook_result(1000, long_text) == ("SUMMARY", 1)     # ~1100 tokens: summarised
    assert hook_result(None, short) == ("SUMMARY", 1)         # None keeps the always-summarise behavior


def test_injected_context_reaches_the_model_alongside_the_agents_own_task():
    # Regression: context used to replace a simple agent's task and be overwritten by a staged agent's prompt.
    from IkaCore.stages import IkaStage

    simple = _agent("simple")
    simple.prompt = "You are the north analyst. Audit the north region."
    simple.inject_workflow_context("Planner notes: focus on Q3.")
    first_message = simple._prepare_simple_runtime().messages[0]["content"]
    assert "You are the north analyst" in first_message and "Planner notes: focus on Q3." in first_message

    staged = IkaBaseAgent(name="staged", description="d", prompt="Audit the south region.", model_id="gpt-4o",
                          api_key="k", Stages=[IkaStage("Gather", "Gather balances.", [])])
    staged.inject_workflow_context("Planner notes: focus on Q3.")
    content = staged._prepare_stage_runtime(0, 10, None).content_prompt
    assert "Audit the south region." in content and "Planner notes: focus on Q3." in content
