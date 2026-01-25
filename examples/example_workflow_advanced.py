"""
Advanced workflow example using:
- Sync: staged orchestrator, child edges with stage_index, next edge, custom compress_hook
- Async: instances, instance_inputs, run_async

Run: API_KEY=xxx python examples/example_workflow_advanced.py
     RUN_WORKFLOW=async  for async-only
     RUN_WORKFLOW=both   for sync then async
"""
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from IkaCore.agents import IkaBaseAgent
from IkaCore.stages import IkaStage
from IkaCore.workflow import (
    IkaWorkflow,
    WorkflowNode,
    WorkflowEdge,
    WorkflowCompressionHook,
)

API_KEY = os.getenv("API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "deepseek-chat")
if not API_KEY:
    raise ValueError("API_KEY is not set")


def truncate_compress(contexts: List[str], _agent: IkaBaseAgent) -> str:
    merged = "\n\n".join(c for c in contexts if c) if contexts else ""
    if not merged:
        return ""
    cap = 2000
    return merged[:cap] + ("..." if len(merged) > cap else "")


def build_sync_workflow(compress_hook: WorkflowCompressionHook) -> IkaWorkflow:
    """
    Sync workflow using:
    - Staged orchestrator (Stages) with child edges and stage_index
    - Child edges: researcher wired to stage 0, critic to stage 1
    - Next edge: orchestrator -> synthesizer
    - Custom compress_hook
    """
    stg0 = IkaStage(
        name="gather",
        prompt="Expand on the topic. If a research subagent is available, you may delegate to it. Then call stage_end.",
        tools=[],
        model_id=MODEL_ID,
        api_key=API_KEY,
        stage_max_step=20,
    )
    stg1 = IkaStage(
        name="refine",
        prompt="Critically refine the content. If a critic subagent is available, you may use it. Then call stage_end.",
        tools=[],
        model_id=MODEL_ID,
        api_key=API_KEY,
        stage_max_step=20,
    )

    orchestrator = IkaBaseAgent(
        name="orchestrator",
        description="Staged coordinator",
        system_prompt="You are a coordinator. Follow each stage and call stage_end when done.",
        prompt="Process the topic you receive: first gather and expand, then refine.",
        tools=[],
        Stages=[stg0, stg1],
        model_id=MODEL_ID,
        api_key=API_KEY,
        maxsteps=50,
        logging_level=2,
    )

    researcher = IkaBaseAgent(
        name="researcher",
        description="Expands and researches",
        system_prompt="You are a researcher. Reply in one short paragraph.",
        prompt="Expand on the topic given. Be concise.",
        tools=[],
        model_id=MODEL_ID,
        api_key=API_KEY,
        logging_level=2,
    )

    critic = IkaBaseAgent(
        name="critic",
        description="Critiques and refines",
        system_prompt="You are a critic. Reply in one short paragraph.",
        prompt="Critically refine or improve the text you receive. Be concise.",
        tools=[],
        model_id=MODEL_ID,
        api_key=API_KEY,
        logging_level=2,
    )

    synthesizer = IkaBaseAgent(
        name="synthesizer",
        description="Final synthesis",
        system_prompt="You are a synthesizer. Produce a short final answer.",
        prompt="Synthesize the information you receive into a clear, final answer in 2-3 sentences.",
        tools=[],
        model_id=MODEL_ID,
        api_key=API_KEY,
        logging_level=2,
    )

    nodes = [
        WorkflowNode(name="orchestrator", agent=orchestrator),
        WorkflowNode(name="researcher", agent=researcher),
        WorkflowNode(name="critic", agent=critic),
        WorkflowNode(name="synthesizer", agent=synthesizer),
    ]

    edges = [
        WorkflowEdge(source="orchestrator", target="researcher", edge_type="child", stage_index=0),
        WorkflowEdge(source="orchestrator", target="critic", edge_type="child", stage_index=1),
        WorkflowEdge(source="orchestrator", target="synthesizer", edge_type="next"),
    ]

    return IkaWorkflow(
        name="sync_advanced",
        description="Staged parent + child(stage_index) + next + custom compress",
        nodes=nodes,
        edges=edges,
        compress_hook=compress_hook,
        start_node="orchestrator",
    )


def build_async_workflow() -> IkaWorkflow:
    """
    Async workflow using:
    - instances=2 and instance_inputs on brainstorm (parallel runs with different prompts)
    - Next edge: brainstorm -> decider
    - run_async for dependency-based parallel execution
    """
    brainstorm = IkaBaseAgent(
        name="brainstorm",
        description="Brainstorms from multiple angles",
        system_prompt="You are a brainstormer. Reply in one short paragraph.",
        prompt="Analyze the topic you receive.",
        tools=[],
        model_id=MODEL_ID,
        api_key=API_KEY,
        logging_level=2,
    )

    decider = IkaBaseAgent(
        name="decider",
        description="Picks and summarizes",
        system_prompt="You are a decider. Produce a short conclusion.",
        prompt="From the analyses you receive, pick the best points and give a 2-3 sentence conclusion.",
        tools=[],
        model_id=MODEL_ID,
        api_key=API_KEY,
        logging_level=2,
    )

    nodes = [
        WorkflowNode(
            name="brainstorm",
            agent=brainstorm,
            instances=2,
            instance_inputs=[
                "Focus on technical or scientific aspects. Be concise.",
                "Focus on practical or societal impact. Be concise.",
            ],
        ),
        WorkflowNode(name="decider", agent=decider),
    ]

    edges = [
        WorkflowEdge(source="brainstorm", target="decider", edge_type="next"),
    ]

    return IkaWorkflow(
        name="async_advanced",
        description="instances + instance_inputs + run_async",
        nodes=nodes,
        edges=edges,
        start_node="brainstorm",
    )


def run_sync() -> None:
    workflow = build_sync_workflow(compress_hook=truncate_compress)
    results = workflow.run(initial_context="Explain the basics of machine learning and why it matters.")
    print("=== Sync workflow (staged + child+stage_index + next + custom compress) ===")
    for name, res in results.items():
        summary = (res.summary or "")[:300]
        childs = list(res.child_summaries.keys()) if res.child_summaries else []
        print(f"[{name}] summary: {summary}...")
        if childs:
            print(f"  child_summaries: {childs}")
    r = results.get("synthesizer")
    print("Final (synthesizer):", ((r.final or "")[:400] + "...") if r else "N/A")


def run_async() -> None:
    workflow = build_async_workflow()
    results = workflow.run(initial_context="The future of renewable energy.", use_async=True)
    print("=== Async workflow (instances + instance_inputs + run_async) ===")
    for name, res in results.items():
        summary = res.summary or ""
        # Check if both instances are present for multi-instance nodes
        has_instance_0 = "[Instance 0]:" in summary
        has_instance_1 = "[Instance 1]:" in summary
        if has_instance_0 and has_instance_1:
            print(f"[{name}] summary: (contains Instance 0 and Instance 1)")
            # Print first 600 chars to show both instances
            print(f"  {summary[:600]}...")
        else:
            print(f"[{name}] summary: {summary[:400]}...")
    decider = results.get("decider")
    if decider:
        print("Final (decider):", (decider.final or "")[:400], "...")


def main() -> None:
    which = (os.getenv("RUN_WORKFLOW") or "sync").strip().lower()
    # run_sync()
    run_async()


if __name__ == "__main__":
    main()
