import os

from IkaCore import IkaBaseAgent, IkaWorkflow, WorkflowEdge, WorkflowNode


def main():
    api_key = os.getenv("API_KEY")
    model_id = os.getenv("MODEL_ID", "deepseek-reasoner")
    if not api_key:
        raise ValueError("API_KEY is not set")

    agent_a = IkaBaseAgent(
        name="agent_a",
        description="First agent in the pipeline",
        system_prompt="You are a concise assistant. Reply in one short paragraph.",
        prompt="Expand on the topic you receive. Produce a brief explanation.",
        tools=[],
        model_id=model_id,
        api_key=api_key,
    )

    agent_b = IkaBaseAgent(
        name="agent_b",
        description="Second agent in the pipeline",
        system_prompt="You are a concise assistant. Reply in one short paragraph.",
        prompt="Summarize the text you receive into 2 or 3 bullet points.",
        tools=[],
        model_id=model_id,
        api_key=api_key,
    )

    node_a = WorkflowNode(name="node_a", agent=agent_a)
    node_b = WorkflowNode(name="node_b", agent=agent_b)

    edges = [
        WorkflowEdge(source="node_a", target="node_b", edge_type="next"),
    ]

    workflow = IkaWorkflow(
        name="two_step",
        description="Linear workflow: A then B",
        nodes=[node_a, node_b],
        edges=edges,
        start_node="node_a",
    )

    results = workflow.run(initial_context="The number 42 and its cultural significance.")

    for name, res in results.items():
        print(f"[{name}] summary: {res.summary[:200]}...")
        print()

    print("Final from last node:", results["node_b"].final[:300], "...")


if __name__ == "__main__":
    main()
