"""Opt-in upstream context that preserves the agent's own instructions."""

# pyright: strict

from typing import Any

from IkaModel.runtime_policy import current_runtime_options


def inject_context(agent: Any, context: str, label: str) -> None:
    previous = getattr(agent, "_workflow_prompt", None)
    if previous != agent.prompt or not hasattr(agent, "_workflow_base_input"):
        first = agent.message_history.get("first_input", {}).get("message", "")
        base = agent.prompt if previous is not None or agent.Stages else (first or agent.prompt)
        agent._workflow_base_input, agent._workflow_prompt = base, agent.prompt
    agent._workflow_context, agent._workflow_context_label = context, label
    agent.message_history["first_input"]["message"] = f"{label}\n{context}\n\n{agent._workflow_base_input}"


def stage_context(agent: Any, content: str) -> str:
    if not current_runtime_options().preserve_workflow_prompts:
        return content
    context = getattr(agent, "_workflow_context", "")
    label = getattr(agent, "_workflow_context_label", "Context from upstream workflow steps:")
    return f"{label}\n{context}\n\n{content}" if context else content
