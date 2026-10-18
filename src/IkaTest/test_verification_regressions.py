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
            agent = IkaBaseAgent(
                name, prompt=name + " task", model_id="gpt-4o-mini", api_key="fixture",
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
