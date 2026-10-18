# pyright: strict

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Coroutine, Optional, cast

from IkaCore.agent_helper_support import (
    RawFinalAnswerCheck,
    initial_message_history,
    validate_final_answer_checks_for_history,
    validate_required_fields,
)
from IkaCore.output_adapters import make_output
from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools
from IkaModel.model_metadata import get_api_url_for_model

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent


def _assign_identity_and_model_config(
    self: Any,
    name: str,
    description: str,
    prompt: str,
    system_prompt: Optional[str],
    start_prompt: Optional[str],
    end_prompt: Optional[str],
    tools: list[IkaTools],
    model_id: str,
    api_key: str,
    api_url: Optional[str],
    max_tokens: int,
    temperature: float,
    use_responses_api: bool,
    reasoning_effort: Optional[str],
) -> None:
    self.name = name
    self.description = description
    self.prompt = prompt
    self.system_prompt = system_prompt
    self.start_prompt = start_prompt
    self.end_prompt = end_prompt
    self.tools = tools
    self.model_id = model_id
    self.api_key = api_key
    self.use_responses_api = use_responses_api
    self.api_url = api_url if api_url else get_api_url_for_model(model_id, use_responses_api)
    self.max_tokens = max_tokens
    self.temperature = temperature
    self.reasoning_effort = reasoning_effort


def _assign_agent_topology(
    self: Any, stages: list[IkaStage], subagents: list["IkaBaseAgent"], next_agent: Optional["IkaBaseAgent"]
) -> None:
    self.Stages = stages
    self.subagents = subagents
    self.next_agent = next_agent
    if self.Stages and (self.subagents or self.next_agent):
        raise ValueError("Subagents or next_agent are not allowed when stages are defined.")
    if not self.Stages and self.subagents and self.next_agent:
        raise ValueError("Only one of subagents or next_agent may be set when no stages are provided.")


def _configure_agent_logger(self: Any, logging_level: int, logging_file: str, show_usage_level0: bool) -> None:
    self.logging_level = logging_level
    self.logging_file = logging_file
    self.events = make_output(self.config.output)
    self.logger = self.events
    if logging_level == 3:
        import logging

        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logging.getLogger("IkaModel.chat_interface.chat_interface").setLevel(logging.DEBUG)


def _assign_execution_limits(
    self: Any,
    maxsteps: int,
    step_timeout: int,
    max_tool_rounds: Optional[int],
    max_tool_calls: Optional[int],
    max_step_extensions: int,
    extend_steps_by: int,
    max_stage_extensions: int,
    extend_stage_steps_by: int,
) -> None:
    self.maxsteps = maxsteps
    self.step_timeout = step_timeout
    self.max_tool_rounds = max_tool_rounds if max_tool_rounds is not None else 5
    self.max_tool_calls = max_tool_calls if max_tool_calls is not None else maxsteps


def _assign_runtime_state(
    self: Any,
    rate_limit_per_min: Optional[float],
    per_tool_rate_limit: Optional[dict[str, float]],
    final_answer_check: Optional[list[RawFinalAnswerCheck]],
    summarize_final: bool,
    use_async: bool,
) -> None:
    self.message_history = initial_message_history(getattr(self, "system_prompt", None))
    self.rate_limit_per_min = rate_limit_per_min
    self.per_tool_rate_limit = per_tool_rate_limit or {}
    self._last_api_call_ts = 0.0
    self._resume_state = None
    self._last_interrupt = None
    self.summarize_final = summarize_final
    self.use_async = use_async
    self.final_answer_checks = validate_final_answer_checks_for_history(self.message_history, final_answer_check)
    self._tool_call_counts = {}
    self._total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "input_cached_tokens": 0}
    self._total_cost = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
    self.client = None


def shutdown_agent(self: Any) -> None:
    client = getattr(self, "client", None)
    if client is not None:
        self.client = None
        close = getattr(client, "close", None)
        if callable(close):
            close()
        else:
            aclose = getattr(client, "aclose", None)
            if callable(aclose):
                import asyncio

                asyncio.run(cast(Coroutine[Any, Any, Any], aclose()))
    logger = getattr(self, "logger", None)
    if logger:
        logger.shutdown()


def initialize_agent(agent: Any) -> None:
    c = agent.config
    from .agent_state import AgentRunState

    agent.state = AgentRunState()
    validate_required_fields(
        {"name": c.name, "description": c.description, "prompt": c.prompt, "model_id": c.model_id, "api_key": c.api_key}
    )
    _assign_identity_and_model_config(
        agent,
        c.name,
        c.description,
        c.prompt,
        c.prompts.system_prompt,
        c.prompts.start_prompt,
        c.prompts.end_prompt,
        list(c.tools),
        c.model_id,
        c.api_key,
        c.provider.api_url,
        c.provider.max_tokens,
        c.provider.temperature,
        c.provider.use_responses_api,
        c.provider.reasoning_effort,
    )
    _assign_agent_topology(agent, list(c.stages), list(c.topology.subagents), c.topology.next_agent)
    _configure_agent_logger(agent, c.output.logging_level, c.output.logging_file, c.output.show_usage_level0)
    _assign_execution_limits(
        agent,
        c.limits.maxsteps,
        c.limits.step_timeout,
        c.limits.max_tool_rounds,
        c.limits.max_tool_calls,
        c.policy.max_step_extensions,
        c.policy.extend_steps_by,
        c.policy.max_stage_extensions,
        c.policy.extend_stage_steps_by,
    )
    _assign_runtime_state(
        agent,
        c.limits.rate_limit_per_min,
        dict(c.limits.per_tool_rate_limit),
        list(c.policy.final_answer_checks) or None,
        c.output.summarize_final,
        c.policy.use_async,
    )
