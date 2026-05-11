import os

from IkaCore import IkaBaseAgent, IkaWorkflow, WorkflowNode, WorkflowEdge

API_KEY = os.getenv("API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "deepseek-reasoner")
if not API_KEY:
    raise ValueError("API_KEY is not set")

def main():
    agent_a = IkaBaseAgent(
        name="agent_a",
        description="First agent in the pipeline",
        system_prompt="You are a concise assistant. Reply in one short paragraph.",
        prompt="Expand on the topic you receive. Produce a brief explanation.",
        tools=[],
        model_id=MODEL_ID,
        api_key=API_KEY,
    )

    agent_b = IkaBaseAgent(
        name="agent_b",
        description="Second agent in the pipeline",
        system_prompt="You are a concise assistant. Reply in one short paragraph.",
        prompt="Summarize the text you receive into 2 or 3 bullet points.",
        tools=[],
        model_id=MODEL_ID,
        api_key=API_KEY,
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
