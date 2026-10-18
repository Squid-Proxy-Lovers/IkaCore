# pyright: reportPrivateUsage=false
from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Dict, List, Optional, cast

from IkaCore.config import AgentTopology
from IkaCore.stages import IkaStage

"AgentLifecycle component; owns one execution responsibility."
if TYPE_CHECKING:
    from .agents import IkaBaseAgent


class AgentLifecycle:
    def __init__(self, agent: IkaBaseAgent):
        self.agent = agent

    def inject_workflow_context(self, context: str, label: str = "Context from upstream workflow steps:") -> None:
        agent = self.agent
        if not context:
            return
        from .workflow_prompt_context import inject_context

        inject_context(agent, context, label)

    def apply_workflow_stage_wiring(self, stage_wiring: Dict[int, Dict[str, List["IkaBaseAgent"]]]) -> None:
        agent = self.agent
        if not stage_wiring or not agent.Stages:
            return
        for stage_idx, wiring in stage_wiring.items():
            if 0 <= stage_idx < len(agent.Stages) and "subagents" in wiring:
                agent.Stages[stage_idx].subagents = wiring["subagents"]

    @staticmethod
    def _clone_stage_for_run(stage: IkaStage) -> IkaStage:
        cloned = copy(stage)
        cloned.tools = list(getattr(stage, "tools", []))
        cloned.subagents = list(getattr(stage, "subagents", []))
        cloned.allowed_back_to = list(getattr(stage, "allowed_back_to", []))
        return cloned

    def final_prompt(self, stage: IkaStage) -> str:
        agent = self.agent
        parts = [agent.start_prompt, stage.prompt, agent.end_prompt]
        if not any(parts):
            parts = [agent.description, stage.prompt]
        return "\n\n".join([part for part in parts if part])

    def _clone_constructor_kwargs(self, name: Optional[str], prompt: Optional[str]) -> dict:
        agent = self.agent
        from dataclasses import replace

        config = getattr(agent, "config")
        return {
            "name": name or agent.name,
            "description": agent.description,
            "prompt": agent.prompt if prompt is None else prompt,
            "model_id": agent.model_id,
            "api_key": agent.api_key,
            "tools": list(agent.tools),
            "stages": [agent.lifecycle._clone_stage_for_run(stage) for stage in agent.Stages],
            "provider": replace(config.provider, api_url=agent.api_url, max_tokens=agent.max_tokens,
                                temperature=agent.temperature, use_responses_api=agent.use_responses_api,
                                reasoning_effort=agent.reasoning_effort,
                                client=None if agent._client_owned else agent.client,
                                client_factory=config.provider.client_factory if agent._client_owned else None),
            "prompts": replace(config.prompts, system_prompt=agent.system_prompt,
                               start_prompt=agent.start_prompt, end_prompt=agent.end_prompt),
            "limits": replace(config.limits, maxsteps=agent.maxsteps, step_timeout=agent.step_timeout,
                              max_tool_rounds=agent.max_tool_rounds, max_tool_calls=agent.max_tool_calls,
                              rate_limit_per_min=agent.rate_limit_per_min,
                              per_tool_rate_limit=agent.per_tool_rate_limit),
            "policy": replace(config.policy, final_answer_checks=(), use_async=agent.use_async),
            "topology": AgentTopology(tuple(agent.subagents), agent.next_agent),
            "output": replace(config.output, cli=False, subscribers=(), logging_level=0, summarize_final=agent.summarize_final),
        }

    def _sync_clone_runtime_state(self, clone: "IkaBaseAgent", include_history: bool) -> None:
        agent = self.agent
        clone.logging_level = agent.logging_level
        clone.events.shutdown()
        clone.events = agent.events
        clone.logger = agent.logger
        clone.final_answer_checks = list(agent.final_answer_checks)
        clone.message_history = (
            deepcopy(agent.message_history) if include_history else agent._initial_message_history(clone.system_prompt)
        )
        clone._owns_events = False

    def _copy_clone_overrides(self, clone: "IkaBaseAgent") -> None:
        agent = self.agent
        for attribute_name in ("context_budget", "_base_prompt"):
            if hasattr(agent, attribute_name):
                setattr(clone, attribute_name, getattr(agent, attribute_name))
        if hasattr(agent, "_parent_hierarchy"):
            setattr(clone, "_parent_hierarchy", list(getattr(agent, "_parent_hierarchy", [])))
        for override_name in ("execution", "async_execution"):
            if override_name in getattr(agent, "__dict__", {}):
                setattr(clone, override_name, getattr(agent, override_name))
        session = getattr(agent, "_snapshot_runtime", None)
        if session is not None:
            clone.enable_snapshots(session.store.db_path, deepcopy(session.control))

    def clone_for_run(
        self, *, name: Optional[str] = None, prompt: Optional[str] = None, include_history: bool = False
    ) -> "IkaBaseAgent":
        agent = self.agent
        clone = cast("IkaBaseAgent", agent.__class__(**agent.lifecycle._clone_constructor_kwargs(name, prompt)))
        agent.lifecycle._sync_clone_runtime_state(clone, include_history)
        agent.lifecycle._copy_clone_overrides(clone)
        return clone
