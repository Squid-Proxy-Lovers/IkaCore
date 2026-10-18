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
            defaults = dict(output=OutputConfig(cli=False), limits=ExecutionLimits.bounded(),
                            policy=ExecutionPolicy.bounded())
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
