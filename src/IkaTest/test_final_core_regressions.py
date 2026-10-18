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
