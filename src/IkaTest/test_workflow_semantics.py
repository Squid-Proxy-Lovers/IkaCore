"""
Workflow semantics tests.

These lock down the v2 behavior where child edges only wire subagents into a
stage and are never auto-executed as separate workflow nodes.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

mock_ika_mem = MagicMock()
with patch.dict("sys.modules", {"IkaMem": mock_ika_mem}):
    from IkaCore.agents import IkaBaseAgent
from IkaCore.stages import IkaStage
from IkaCore.workflow import IkaWorkflow, WorkflowEdge, WorkflowNode


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
