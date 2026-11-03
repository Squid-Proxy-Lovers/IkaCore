"""Chained-agent variants of interrupted prerequisite execution."""

import pytest

from IkaCore import AgentTopology, ExecutionLimits, ExecutionPolicy, IkaBaseAgent, IkaTools, OutputConfig
from IkaCore.testing import MockProvider, provider_response
from IkaModel.base import HumanInputRequired


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("snapshots", [False, True])
def test_next_agent_waits_for_tool_approval(tmp_path, asynchronous, snapshots):
    def create(name, **kwargs):
        return IkaBaseAgent(name, prompt=name + " task", model_id="gpt-4o-mini", api_key="fixture",
                            output=OutputConfig(cli=False), limits=ExecutionLimits.bounded(),
                            policy=ExecutionPolicy.bounded(async_requests=asynchronous), **kwargs)
    def approve(args):
        raise HumanInputRequired({"question": "Approve?"})
    approval = IkaTools("approval", "Request approval", {"input": {"type": "string", "required": True}}, execute_function=approve)
    downstream = create("downstream")
    parent = create("parent", tools=[approval], topology=AgentTopology(next_agent=downstream))
    if snapshots:
        parent.enable_snapshots(str(tmp_path / "chain.db"))
    actions = iter([("approval", {"input": "needs approval"}),
                    ("agent_end", {"input": "UNAPPROVED_DOWNSTREAM_EXECUTED"})])
    try:
        with MockProvider(responder=lambda u, p: provider_response(u, p, next(actions))) as mock:
            result = parent.execution()
        assert len(mock.requests) == 1
        assert result["status"] == "awaiting_user_input"
    finally:
        parent.shutdown()
        downstream.shutdown()
