"""
Workflow semantics tests.

These lock down the v2 behavior where child edges only wire subagents into a
stage and are never auto-executed as separate workflow nodes.
"""
import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

mock_ika_mem = MagicMock()
with patch.dict("sys.modules", {"IkaMem": mock_ika_mem}):
    from IkaCore.agents import IkaBaseAgent
from IkaCore.stages import IkaStage
from IkaCore.workflow import AsyncWorkflowExecutor, IkaWorkflow, WorkflowEdge, WorkflowNode, WorkflowResult


def _agent(name: str, stages=None):
    return IkaBaseAgent(
        name=name,
        description=f"{name} description",
        prompt=f"{name} prompt",
        model_id="gpt-4o",
        api_key="test-key",
        Stages=stages or [],
    )


def _compress_hook(contexts, agent):
    return "\n".join([context for context in contexts if context])


class InlineWorkflowExecutor:
    max_workers = 1

    def __init__(self):
        self.scheduled = []

    def schedule_node(self, node_name, agent, context="", instance_id=0):
        import concurrent.futures

        self.scheduled.append(
            {
                "node_name": node_name,
                "agent_name": agent.name,
                "prompt": agent.prompt,
                "context": context,
                "instance_id": instance_id,
            }
        )
        future = concurrent.futures.Future()
        if context:
            agent.inject_workflow_context(context)
        try:
            future.set_result(
                {
                    "node_name": node_name,
                    "instance_id": instance_id,
                    "result": agent.execution(),
                    "success": True,
                }
            )
        except Exception as exc:
            future.set_result(
                {
                    "node_name": node_name,
                    "instance_id": instance_id,
                    "result": {"error": str(exc)},
                    "success": False,
                }
            )
        return future

    def wait_for_completion(self, futures, timeout=None):
        return [future.result() for future in futures]

    def drain(self):
        return None


class TestWorkflowChildEdges:
    def test_child_edge_only_wires_stage_and_does_not_execute_child(self):
        parent_stage = IkaStage("ParentStage", "Do parent work", [])
        parent = _agent("parent", stages=[parent_stage])
        child = _agent("child")
        follower = _agent("follower")

        parent.execution = MagicMock(return_value={"final_message": "parent final", "summary": "parent summary"})
        child.execution = MagicMock(side_effect=AssertionError("child workflow node should not execute"))
        follower.execution = MagicMock(return_value={"final_message": "follower final", "summary": "follower summary"})

        workflow = IkaWorkflow(
            name="wf",
            description="workflow",
            nodes=[
                WorkflowNode(name="parent", agent=parent),
                WorkflowNode(name="child", agent=child),
                WorkflowNode(name="follower", agent=follower),
            ],
            edges=[
                WorkflowEdge(source="parent", target="child", edge_type="child", stage_index=0),
                WorkflowEdge(source="parent", target="follower", edge_type="next"),
            ],
            compress_hook=_compress_hook,
        )

        results = workflow.run()

        assert "parent" in results
        assert "follower" in results
        assert "child" not in results
        assert child.execution.call_count == 0
        assert parent.Stages[0].subagents == [child]

    def test_async_workflow_ignores_child_only_nodes(self):
        parent_stage = IkaStage("ParentStage", "Do parent work", [])
        parent = _agent("parent", stages=[parent_stage])
        child = _agent("child")
        follower = _agent("follower")

        parent.execution = MagicMock(return_value={"final_message": "parent final", "summary": "parent summary"})
        child.execution = MagicMock(side_effect=AssertionError("child workflow node should not execute"))
        follower.execution = MagicMock(return_value={"final_message": "follower final", "summary": "follower summary"})

        workflow = IkaWorkflow(
            name="wf-async",
            description="workflow",
            nodes=[
                WorkflowNode(name="parent", agent=parent),
                WorkflowNode(name="child", agent=child),
                WorkflowNode(name="follower", agent=follower),
            ],
            edges=[
                WorkflowEdge(source="parent", target="child", edge_type="child", stage_index=0),
                WorkflowEdge(source="parent", target="follower", edge_type="next"),
            ],
            compress_hook=_compress_hook,
        )

        results = workflow.run(use_async=True)

        assert "parent" in results
        assert "follower" in results
        assert "child" not in results
        assert child.execution.call_count == 0
        assert workflow._next_reachable_nodes == {"parent", "follower"}

    def test_child_edge_requires_stage_index(self):
        parent = _agent("parent", stages=[IkaStage("ParentStage", "Do parent work", [])])
        child = _agent("child")

        with pytest.raises(ValueError, match="must define stage_index"):
            IkaWorkflow(
                name="wf-invalid",
                description="workflow",
                nodes=[
                    WorkflowNode(name="parent", agent=parent),
                    WorkflowNode(name="child", agent=child),
                ],
                edges=[WorkflowEdge(source="parent", target="child", edge_type="child")],
                compress_hook=_compress_hook,
            )


class TestWorkflowValidationDisplayAndContext:
    def test_workflow_rejects_invalid_graph_configuration(self):
        agent = _agent("agent")
        other = _agent("other")

        with pytest.raises(ValueError, match="edge_type"):
            WorkflowEdge(source="agent", target="other", edge_type="bad")
        with pytest.raises(ValueError, match="instances"):
            WorkflowNode("bad-instances", agent, instances=0)
        with pytest.raises(ValueError, match="instance_inputs"):
            WorkflowNode("too-many-inputs", agent, instances=1, instance_inputs=["one", "two"])

        with pytest.raises(ValueError, match="at least one node"):
            IkaWorkflow(name="empty", description="workflow", nodes=[], edges=[])

        with pytest.raises(ValueError, match="start_node"):
            IkaWorkflow(
                name="bad-start",
                description="workflow",
                nodes=[WorkflowNode("agent", agent)],
                edges=[],
                start_node="missing",
            )

        with pytest.raises(ValueError, match="Edge source"):
            IkaWorkflow(
                name="bad-source",
                description="workflow",
                nodes=[WorkflowNode("agent", agent)],
                edges=[WorkflowEdge("missing", "agent")],
            )

        with pytest.raises(ValueError, match="Edge target"):
            IkaWorkflow(
                name="bad-target",
                description="workflow",
                nodes=[WorkflowNode("agent", agent)],
                edges=[WorkflowEdge("agent", "missing")],
            )

        staged = _agent("staged", stages=[IkaStage("S", "P", [])])
        staged.next_agent = other
        with pytest.raises(ValueError, match="cannot declare subagents"):
            IkaWorkflow(
                name="bad-staged-topology",
                description="workflow",
                nodes=[WorkflowNode("staged", staged), WorkflowNode("other", other)],
                edges=[],
            )

    def test_repr_shows_stage_wiring_edges_and_cycles_without_recursing_forever(self):
        parent = _agent("parent", stages=[IkaStage("S", "P", [])])
        child = _agent("child")
        workflow = IkaWorkflow(
            name="display",
            description="workflow display",
            nodes=[WorkflowNode("parent", parent), WorkflowNode("child", child)],
            edges=[
                WorkflowEdge("parent", "child", edge_type="child", stage_index=0),
                WorkflowEdge("child", "parent", edge_type="next"),
            ],
            compress_hook=_compress_hook,
            start_node="parent",
        )

        rendered = repr(workflow)

        assert "IkaWorkflow: display" in rendered
        assert "parent <-> child (child) [stage 0]" in rendered
        assert "child --> parent (next)" in rendered
        assert "(cycle)" in rendered
        assert workflow._node_index["parent"].stage_wiring[0]["subagents"] == [child]

    def test_default_compress_hook_summarizes_or_falls_back_to_merged_context(self):
        agent = _agent("agent")
        workflow = IkaWorkflow(
            name="compress",
            description="workflow",
            nodes=[WorkflowNode("agent", agent)],
            edges=[],
        )

        assert workflow._default_compress_hook([], agent) == ""

        summarizer = MagicMock(return_value="short summary")
        with patch("IkaCore.workflow_core.summarise_message_history", summarizer):
            assert workflow._default_compress_hook(["one", "", "two"], agent) == "short summary"
        summarizer.assert_called_once()

        with patch("IkaCore.workflow_core.summarise_message_history", side_effect=RuntimeError("summary failed")):
            assert workflow._default_compress_hook(["one", "two"], agent) == "one\n\ntwo"

    def test_sync_run_propagates_initial_and_upstream_context_and_detects_cycles(self):
        root = _agent("root")
        follower = _agent("follower")
        root.execution = MagicMock(return_value={"final_message": "root final", "summary": "root summary"})
        follower.execution = MagicMock(return_value={"final_message": "follower final", "summary": "follower summary"})
        workflow = IkaWorkflow(
            name="sync-context",
            description="workflow",
            nodes=[WorkflowNode("root", root), WorkflowNode("follower", follower)],
            edges=[WorkflowEdge("root", "follower")],
            compress_hook=_compress_hook,
        )

        results = workflow.run(initial_context="initial")

        assert set(results) == {"root", "follower"}
        assert "initial" in root.message_history["first_input"]["message"]
        assert "root summary" in follower.message_history["first_input"]["message"]
        assert workflow._run_node("root", {}) is results["root"]

        workflow._results.pop("follower")
        workflow._visiting.add("follower")
        with pytest.raises(ValueError, match="Cycle detected"):
            workflow._run_node("follower", {})


class TestWorkflowAsyncSemantics:
    def test_async_instances_preserve_instance_inputs_and_aggregate_summaries(self):
        worker = _agent("worker")
        worker.execution = MagicMock(
            side_effect=[
                {"final_message": "first final", "summary": ""},
                {"final_message": "second final", "summary": "second summary"},
            ]
        )
        executor = InlineWorkflowExecutor()
        workflow = IkaWorkflow(
            name="wf-instances",
            description="workflow",
            nodes=[
                WorkflowNode(
                    name="worker",
                    agent=worker,
                    instances=2,
                    instance_inputs=["first prompt", "second prompt"],
                )
            ],
            edges=[],
            compress_hook=_compress_hook,
            async_executor=executor,
        )

        with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
            results = workflow.run(use_async=True)

        assert [item["prompt"] for item in executor.scheduled] == ["first prompt", "second prompt"]
        assert "[Instance 0]: first final" in results["worker"].summary
        assert "[Instance 1]: second summary" in results["worker"].summary

    def test_async_success_propagates_context_to_dependents(self):
        root = _agent("root")
        follower = _agent("follower")
        root.execution = MagicMock(return_value={"final_message": "root final", "summary": "root summary"})
        follower.execution = MagicMock(return_value={"final_message": "follower final", "summary": "follower summary"})
        executor = InlineWorkflowExecutor()
        workflow = IkaWorkflow(
            name="wf-context",
            description="workflow",
            nodes=[
                WorkflowNode(name="root", agent=root),
                WorkflowNode(name="follower", agent=follower),
            ],
            edges=[WorkflowEdge(source="root", target="follower", edge_type="next")],
            compress_hook=_compress_hook,
            async_executor=executor,
        )

        with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
            results = workflow.run(initial_context="initial context", use_async=True)

        scheduled_by_node = {item["node_name"]: item for item in executor.scheduled}
        assert scheduled_by_node["root"]["context"] == "initial context"
        assert scheduled_by_node["follower"]["context"] == "root summary"
        assert set(results) == {"root", "follower"}

    def test_async_failure_does_not_mark_dependency_complete_or_run_dependent(self):
        root = _agent("root")
        follower = _agent("follower")
        root.execution = MagicMock(side_effect=RuntimeError("root failed"))
        follower.execution = MagicMock(return_value={"final_message": "should not run", "summary": "bad"})
        executor = InlineWorkflowExecutor()
        workflow = IkaWorkflow(
            name="wf-failure",
            description="workflow",
            nodes=[
                WorkflowNode(name="root", agent=root),
                WorkflowNode(name="follower", agent=follower),
            ],
            edges=[WorkflowEdge(source="root", target="follower", edge_type="next")],
            compress_hook=_compress_hook,
            async_executor=executor,
        )

        with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
            results = workflow.run(use_async=True)

        assert results == {}
        assert [item["node_name"] for item in executor.scheduled] == ["root"]
        follower.execution.assert_not_called()

    def test_async_multi_instance_failure_blocks_dependents_and_node_result(self):
        root = _agent("root")
        follower = _agent("follower")
        root.execution = MagicMock(
            side_effect=[
                {"final_message": "first final", "summary": "first summary"},
                RuntimeError("second failed"),
            ]
        )
        follower.execution = MagicMock(return_value={"final_message": "should not run", "summary": "bad"})
        executor = InlineWorkflowExecutor()
        workflow = IkaWorkflow(
            name="wf-partial-failure",
            description="workflow",
            nodes=[
                WorkflowNode("root", root, instances=2),
                WorkflowNode("follower", follower),
            ],
            edges=[WorkflowEdge("root", "follower")],
            compress_hook=_compress_hook,
            async_executor=executor,
        )

        with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
            results = workflow.run(use_async=True)

        assert results == {}
        assert [item["node_name"] for item in executor.scheduled] == ["root", "root"]
        follower.execution.assert_not_called()

    def test_async_fanout_fanin_uses_all_upstream_contexts_before_join(self):
        root = _agent("root")
        left = _agent("left")
        right = _agent("right")
        join = _agent("join")
        root.execution = MagicMock(
            side_effect=[
                {"final_message": "root first", "summary": "root summary first"},
                {"final_message": "root second", "summary": "root summary second"},
            ]
        )
        left.execution = MagicMock(return_value={"final_message": "left final", "summary": "left summary"})
        right.execution = MagicMock(return_value={"final_message": "right final", "summary": "right summary"})
        join.execution = MagicMock(return_value={"final_message": "join final", "summary": "join summary"})
        executor = InlineWorkflowExecutor()
        workflow = IkaWorkflow(
            name="wf-fan",
            description="workflow",
            nodes=[
                WorkflowNode("root", root, instances=2, instance_inputs=["first input"]),
                WorkflowNode("left", left),
                WorkflowNode("right", right),
                WorkflowNode("join", join),
            ],
            edges=[
                WorkflowEdge("root", "left"),
                WorkflowEdge("root", "right"),
                WorkflowEdge("left", "join"),
                WorkflowEdge("right", "join"),
            ],
            compress_hook=_compress_hook,
            async_executor=executor,
        )

        with patch("IkaCore.workflow_async.get_cli_output", return_value=MagicMock()):
            results = workflow.run(initial_context="initial context", use_async=True)

        scheduled = {}
        for item in executor.scheduled:
            scheduled.setdefault(item["node_name"], []).append(item)
        assert [item["prompt"] for item in scheduled["root"]] == ["first input", "root prompt"]
        for node_name in ("left", "right"):
            context = scheduled[node_name][0]["context"]
            assert "root summary first" in context
            assert "root summary second" in context
        assert "left summary" in scheduled["join"][0]["context"]
        assert "right summary" in scheduled["join"][0]["context"]
        assert set(results) == {"root", "left", "right", "join"}

    def test_async_executor_schedules_sync_async_and_failed_agents(self):
        executor = AsyncWorkflowExecutor(max_workers=1)
        cli = MagicMock()
        sync_agent = _agent("sync")
        sync_agent.execution = MagicMock(return_value={"final_message": "sync done"})

        async_agent = _agent("async")
        async_agent.use_async = True

        async def async_execution():
            return {"final_message": "async done"}

        async_agent.async_execution = async_execution
        failing_agent = _agent("failing")
        failing_agent.execution = MagicMock(side_effect=RuntimeError("failed"))

        try:
            with patch("IkaCore.workflow_async_executor.get_cli_output", return_value=cli):
                futures = [
                    executor.schedule_node("sync", sync_agent, context="sync context", instance_id=1),
                    executor.schedule_node("async", async_agent, instance_id=2),
                    executor.schedule_node("failing", failing_agent, instance_id=3),
                ]
                results = executor.wait_for_completion(futures)
        finally:
            executor.shutdown()

        by_node = {result["node_name"]: result for result in results}
        assert by_node["sync"]["success"] is True
        assert by_node["sync"]["result"]["final_message"] == "sync done"
        assert by_node["async"]["success"] is True
        assert by_node["async"]["result"]["final_message"] == "async done"
        assert by_node["failing"]["success"] is False
        assert by_node["failing"]["result"]["error"] == "failed"
        assert "sync context" in sync_agent.message_history["first_input"]["message"]
        assert executor.pending_tasks == {}
        assert set(executor.completed_results) == {"sync_1", "async_2", "failing_3"}

    def test_async_executor_wait_handles_future_exceptions_and_drain_uses_pending_tasks(self):
        executor = AsyncWorkflowExecutor(max_workers=1)
        failed_future = concurrent.futures.Future()
        failed_future.set_exception(RuntimeError("future exploded"))

        try:
            results = executor.wait_for_completion([failed_future])
            pending_future = concurrent.futures.Future()
            pending_future.set_result({"node_name": "node", "instance_id": 0, "result": {}, "success": True})
            executor.pending_tasks["node_0"] = pending_future
            executor.wait_for_completion = MagicMock(return_value=[])

            executor.drain()
        finally:
            executor.shutdown()

        assert results == [{"result": {"error": "future exploded"}, "success": False}]
        executor.wait_for_completion.assert_called_once_with([pending_future])

    def test_async_state_scheduling_and_dependency_activation_helpers(self):
        root = _agent("root")
        mid = _agent("mid")
        leaf = _agent("leaf")
        executor = InlineWorkflowExecutor()
        workflow = IkaWorkflow(
            name="async-helpers",
            description="workflow",
            nodes=[
                WorkflowNode("root", root, instances=2, instance_inputs=["first"]),
                WorkflowNode("mid", mid),
                WorkflowNode("leaf", leaf),
            ],
            edges=[WorkflowEdge("root", "mid"), WorkflowEdge("mid", "leaf")],
            compress_hook=_compress_hook,
            async_executor=executor,
        )

        upstream_contexts, ready_nodes, completed_nodes, pending_nodes, node_futures = workflow._initial_async_state(
            "initial"
        )

        assert upstream_contexts == {"root": ["initial"]}
        assert ready_nodes == {"root"}
        assert completed_nodes == set()
        assert pending_nodes == {"mid", "leaf"}
        assert node_futures == {}
        assert workflow._instance_inputs_for_node(workflow._node_index["root"]) == ["first", None]
        assert workflow._async_context_for_node(workflow._node_index["root"], "root", {}, "initial") == "initial"

        cli = MagicMock()
        root.execution = MagicMock(
            side_effect=[
                {"final_message": "first final", "summary": "first summary"},
                {"final_message": "second final", "summary": "second summary"},
            ]
        )
        futures = workflow._schedule_async_node("root", upstream_contexts, "initial", cli, step=1)

        assert len(futures) == 2
        assert [item["prompt"] for item in executor.scheduled] == ["first", "root prompt"]
        assert executor.scheduled[0]["context"] == "initial"
        cli.workflow_status.assert_called_once()

        workflow._results = {}
        workflow._record_async_success(
            {"node_name": "root", "instance_id": 0, "result": {"final_message": "root final"}, "success": True}
        )
        workflow._record_async_success(
            {"node_name": "root", "instance_id": 1, "result": {"summary": "second summary"}, "success": True}
        )
        assert "[Instance 0]: root final" in workflow._results["root"].summary
        assert "[Instance 1]: second summary" in workflow._results["root"].summary

        ready_nodes = set()
        pending_nodes = {"mid", "leaf"}
        upstream_contexts = {}
        workflow._activate_async_dependents(
            {"node_name": "root", "instance_id": 0, "result": {"summary": "root summary"}, "success": False},
            upstream_contexts,
            completed_nodes=set(),
            ready_nodes=ready_nodes,
            pending_nodes=pending_nodes,
        )
        assert ready_nodes == set()
        assert upstream_contexts == {}

        completed_nodes = set()
        workflow._activate_async_dependents(
            {"node_name": "root", "instance_id": 0, "result": {"summary": "root summary"}, "success": True},
            upstream_contexts,
            completed_nodes=completed_nodes,
            ready_nodes=ready_nodes,
            pending_nodes=pending_nodes,
        )
        assert completed_nodes == {"root"}
        assert ready_nodes == {"mid"}
        assert pending_nodes == {"leaf"}
        assert upstream_contexts == {"mid": ["root summary"]}

        ready_nodes.clear()
        pending_nodes = {"leaf"}
        workflow._activate_unblocked_pending_nodes(pending_nodes, completed_nodes | {"mid"}, ready_nodes)
        assert ready_nodes == {"leaf"}
        assert pending_nodes == set()

    def test_process_async_results_skips_failures_and_cleans_node_futures(self):
        root = _agent("root")
        follower = _agent("follower")
        workflow = IkaWorkflow(
            name="process-results",
            description="workflow",
            nodes=[WorkflowNode("root", root), WorkflowNode("follower", follower)],
            edges=[WorkflowEdge("root", "follower")],
            compress_hook=_compress_hook,
            async_executor=InlineWorkflowExecutor(),
        )
        success_future = concurrent.futures.Future()
        success_future.set_result(
            {"node_name": "root", "instance_id": 0, "result": {"summary": "root summary"}, "success": True}
        )
        failure_future = concurrent.futures.Future()
        failure_future.set_result(
            {"node_name": "follower", "instance_id": 0, "result": {"error": "bad"}, "success": False}
        )
        node_futures = {"root": [success_future], "follower": [failure_future]}
        ready_nodes = set()
        pending_nodes = {"follower"}
        completed_nodes = set()
        upstream_contexts = {}

        workflow._process_async_results(
            [],
            upstream_contexts,
            completed_nodes,
            ready_nodes,
            pending_nodes,
            node_futures,
        )
        assert node_futures == {"root": [success_future], "follower": [failure_future]}

        workflow._process_async_results(
            [success_future, failure_future],
            upstream_contexts,
            completed_nodes,
            ready_nodes,
            pending_nodes,
            node_futures,
        )

        assert "root" in workflow._results
        assert "follower" not in workflow._results
        assert completed_nodes == {"root"}
        assert ready_nodes == {"follower"}
        assert node_futures == {}

    def test_create_agent_instance_and_run_node_async_apply_instance_overrides(self):
        stage = IkaStage("S", "P", [])
        worker = _agent("worker", stages=[stage])
        child = _agent("child")
        workflow = IkaWorkflow(
            name="async-node",
            description="workflow",
            nodes=[WorkflowNode("worker", worker, stage_wiring={0: {"subagents": [child]}})],
            edges=[],
            compress_hook=_compress_hook,
        )

        worker.execution = MagicMock(return_value={"final_message": "original", "summary": "original summary"})
        first = workflow._run_node_async("worker", {}, instance_id=0)
        cached = workflow._run_node_async("worker", {}, instance_id=0)

        assert first is cached
        assert isinstance(first, WorkflowResult)

        worker.execution = MagicMock(return_value={"final_message": "second", "summary": ""})
        second = workflow._run_node_async(
            "worker",
            {"worker": ["upstream"]},
            instance_id=2,
            instance_input="instance prompt",
        )
        assert second.final == "second"
        assert second.summary == "second"
        assert "worker_instance_2" not in workflow._results

        copy = workflow._create_agent_instance(worker, 3, instance_input="prompt override")
        assert copy.name == "worker_instance_3"
        assert copy.prompt == "prompt override"
