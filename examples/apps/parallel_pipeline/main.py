"""Parallel workflow with fan-in, cooperative Ctrl-C and a borrowed HTTPX client."""
from __future__ import annotations

import signal
import threading
from typing import Any, ClassVar

import httpx

from examples.apps.common import agent, emit, end, parser, session
from IkaCore import IkaBaseAgent, IkaWorkflow, OutputConfig, WorkflowEdge, WorkflowNode
from IkaModel import request_controls
from IkaModel.runtime_errors import IkaRequestCancelled, IkaRequestDeadlineExceeded


class BorrowedClientAgent(IkaBaseAgent):
    """Workflow clones drop .client; explicitly inject our shared thread-safe client.

    Workflow parallel mode runs synchronous agents in worker threads. A sync HTTPX
    client can be shared across those threads; an AsyncClient cannot span their loops.
    """
    shared_client: ClassVar[httpx.Client]

    def clone_for_run(self, **kwargs: Any) -> IkaBaseAgent:
        clone = super().clone_for_run(**kwargs)
        clone.client = self.shared_client
        return clone

    def shutdown(self) -> None:
        if self.client is self.shared_client:
            self.client = None  # The outer context owns close(), including on cancellation.
        super().shutdown()


def main() -> None:
    cli = parser(__doc__)
    cli.add_argument("--cancel-demo", action="store_true", help="deliver SIGINT before dispatch; no model call")
    args = cli.parse_args()
    cancelled = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda signum, frame: cancelled.set())
    scripts = {"seed": [end("OFFICE_PILOT")], "cost": [end("COST:100")],
               "risk": [end("RISK:LOW")], "join": [end("COST:100 RISK:LOW")]} if not args.cancel_demo else {}
    try:
        with httpx.Client(timeout=45) as client, session(args, scripts):
            BorrowedClientAgent.shared_client = client
            tasks = [agent(args, "seed", "Call agent_end with exactly OFFICE_PILOT.", agent_type=BorrowedClientAgent),
                     agent(args, "cost", "For upstream OFFICE_PILOT call agent_end with exactly COST:100.", agent_type=BorrowedClientAgent),
                     agent(args, "risk", "For upstream OFFICE_PILOT call agent_end with exactly RISK:LOW.", agent_type=BorrowedClientAgent),
                     agent(args, "join", "Read BOTH upstream results. Call agent_end with exactly COST:100 RISK:LOW.", agent_type=BorrowedClientAgent)]
            joined_contexts = []

            def compress(contexts, target):
                if target.name.startswith("join"):
                    joined_contexts.extend(contexts)
                return "\n\n".join(contexts)

            workflow = IkaWorkflow("parallel", "Cost and risk analysis", [WorkflowNode(t.name, t) for t in tasks],
                                  [WorkflowEdge("seed", "cost"), WorkflowEdge("seed", "risk"),
                                   WorkflowEdge("cost", "join"), WorkflowEdge("risk", "join")],
                                  compress_hook=compress, max_parallel_workers=2, output=OutputConfig(cli=False))
            try:
                # Expired requests must fail before dispatch (and incur no live calls).
                with request_controls(timeout=0):
                    try:
                        tasks[0].execution()
                    except IkaRequestDeadlineExceeded:
                        pass
                    else:
                        raise AssertionError("expired deadline ignored")
                if args.cancel_demo:
                    signal.raise_signal(signal.SIGINT)
                    with request_controls(cancel_checker=cancelled.is_set):
                        try:
                            tasks[0].execution()
                        except IkaRequestCancelled:
                            pass
                        else:
                            raise AssertionError("SIGINT cancellation ignored")
                    emit("parallel_pipeline", cancelled=True, caller_client_open=not client.is_closed)
                else:
                    with request_controls(timeout=70, cancel_checker=cancelled.is_set):
                        results = workflow.run(use_async=True)
                    if cancelled.is_set():
                        raise IkaRequestCancelled("Pipeline cancelled")
                    assert results["join"].final == "COST:100 RISK:LOW"
                    assert {"COST:100", "RISK:LOW"} <= set(joined_contexts)
                    assert not client.is_closed
                    emit("parallel_pipeline", fan_in=True, caller_client_open=True, workers=2, deadline_seconds=70)
            finally:
                workflow.async_executor.shutdown()
                for task in tasks:
                    task.shutdown()
                assert not client.is_closed
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    main()
