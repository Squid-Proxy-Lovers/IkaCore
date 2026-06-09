import os
from typing import List

from IkaCore import IkaBaseAgent, IkaStage
from IkaCore.workflow import IkaWorkflow, WorkflowCompressionHook, WorkflowEdge, WorkflowNode


def truncate_compress(contexts: List[str], _agent: IkaBaseAgent) -> str:
    merged = "\n\n".join(c for c in contexts if c) if contexts else ""
    if not merged:
        return ""
    cap = 2000
    return merged[:cap] + ("..." if len(merged) > cap else "")


def build_sync_workflow(compress_hook: WorkflowCompressionHook, api_key: str, model_id: str) -> IkaWorkflow:
    stg0 = IkaStage(
        name="gather",
        prompt="Expand on the topic. If a research subagent is available, you may delegate to it. Then call stage_end.",
        tools=[],
        model_id=model_id,
        api_key=api_key,
        stage_max_step=20,
    )
    stg1 = IkaStage(
        name="refine",
        prompt="Critically refine the content. If a critic subagent is available, you may use it. Then call stage_end.",
        tools=[],
        model_id=model_id,
        api_key=api_key,
        stage_max_step=20,
    )

    orchestrator = IkaBaseAgent(
        name="orchestrator",
        description="Staged coordinator",
        system_prompt="You are a coordinator. Follow each stage and call stage_end when done.",
        prompt="Process the topic you receive: first gather and expand, then refine.",
        tools=[],
        Stages=[stg0, stg1],
        model_id=model_id,
        api_key=api_key,
        maxsteps=50,
        logging_level=2,
    )

    researcher = IkaBaseAgent(
        name="researcher",
        description="Expands and researches",
        system_prompt="You are a researcher. Reply in one short paragraph.",
        prompt="Expand on the topic given. Be concise.",
        tools=[],
        model_id=model_id,
        api_key=api_key,
        logging_level=2,
    )

    critic = IkaBaseAgent(
        name="critic",
        description="Critiques and refines",
        system_prompt="You are a critic. Reply in one short paragraph.",
        prompt="Critically refine or improve the text you receive. Be concise.",
        tools=[],
        model_id=model_id,
        api_key=api_key,
        logging_level=2,
    )

    synthesizer = IkaBaseAgent(
        name="synthesizer",
        description="Final synthesis",
        system_prompt="You are a synthesizer. Produce a short final answer.",
        prompt="Synthesize the information you receive into a clear, final answer in 2-3 sentences.",
        tools=[],
        model_id=model_id,
        api_key=api_key,
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


def build_async_workflow(api_key: str, model_id: str) -> IkaWorkflow:
    brainstorm = IkaBaseAgent(
        name="brainstorm",
        description="Brainstorms from multiple angles",
        system_prompt="You are a brainstormer. Reply in one short paragraph.",
        prompt="Analyze the topic you receive.",
        tools=[],
        model_id=model_id,
        api_key=api_key,
        logging_level=2,
    )

    decider = IkaBaseAgent(
        name="decider",
        description="Picks and summarizes",
        system_prompt="You are a decider. Produce a short conclusion.",
        prompt="From the analyses you receive, pick the best points and give a 2-3 sentence conclusion.",
        tools=[],
        model_id=model_id,
        api_key=api_key,
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
    api_key = os.getenv("API_KEY")
    model_id = os.getenv("MODEL_ID", "deepseek-chat")
    if not api_key:
        raise ValueError("API_KEY is not set")

    print("=" * 100)
    print("Running sync workflow:")
    workflow = build_sync_workflow(compress_hook=truncate_compress, api_key=api_key, model_id=model_id)
    results = workflow.run(initial_context="Explain the basics of machine learning and why it matters.")
    for name, res in results.items():
        summary = res.summary or ""
        childs = list(res.child_summaries.keys()) if res.child_summaries else []
        print(f"[{name}] summary: {summary}")
        if childs:
            print(f"  child_summaries: {childs}")
    r = results.get("synthesizer")
    print("Final (synthesizer):", (r.final or ""))


def run_async() -> None:
    api_key = os.getenv("API_KEY")
    model_id = os.getenv("MODEL_ID", "deepseek-chat")
    if not api_key:
        raise ValueError("API_KEY is not set")

    print("=" * 100)
    print("Running async workflow:")
    workflow = build_async_workflow(api_key=api_key, model_id=model_id)
    results = workflow.run(initial_context="The future of renewable energy.", use_async=True)
    for name, res in results.items():
        summary = res.summary or ""
        has_instance_0 = "[Instance 0]:" in summary
        has_instance_1 = "[Instance 1]:" in summary
        if has_instance_0 and has_instance_1:
            print(f"[{name}] summary: (contains Instance 0 and Instance 1)")
            print(f"  {summary}")
        else:
            print(f"[{name}] summary: {summary}")
    decider = results.get("decider")
    if decider:
        print("Final (decider):", (decider.final or ""))


def main() -> None:
    # run_sync()
    run_async()


if __name__ == "__main__":
    main()
