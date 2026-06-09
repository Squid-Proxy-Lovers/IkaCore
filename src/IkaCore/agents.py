from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from IkaCore.agent_execution import AgentExecutionMixin
from IkaCore.agent_helper_support import RawFinalAnswerCheck
from IkaCore.agent_helpers import AgentHelpersMixin
from IkaCore.agent_initialization import AgentInitializationMixin
from IkaCore.agent_memory import AgentMemoryMixin
from IkaCore.agent_tools import AgentToolsMixin
from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools


@dataclass
class _AgentConstructorConfig:
    name: str
    description: str
    prompt: str
    system_prompt: Optional[str]
    start_prompt: Optional[str]
    end_prompt: Optional[str]
    tools: Optional[list[IkaTools]]
    model_id: str
    api_key: str
    api_url: Optional[str]
    max_tokens: int
    temperature: float
    use_responses_api: bool
    checkpoint: bool
    checkpoint_db_path: str
    Stages: Optional[list[IkaStage]]
    subagents: Optional[list[IkaBaseAgent]]
    next_agent: Optional["IkaBaseAgent"]
    maxsteps: int
    step_timeout: int
    rate_limit_per_min: Optional[float]
    per_tool_rate_limit: Optional[dict[str, float]]
    memory: bool
    memory_access: Optional[dict[str, bool]]
    final_answer_check: Optional[list[RawFinalAnswerCheck]]
    logging_level: int
    logging_file: str
    show_usage_level0: bool
    summarize_final: bool
    use_async: bool
    max_tool_rounds: Optional[int]
    max_tool_calls: Optional[int]
    max_step_extensions: int
    extend_steps_by: int
    max_stage_extensions: int
    extend_stage_steps_by: int
    reasoning_effort: Optional[str]


def _initialize_base_agent(agent: "IkaBaseAgent", config: _AgentConstructorConfig) -> None:
    tools, stages, subagents = agent._normalize_init_collections(config.tools, config.Stages, config.subagents)

    agent._validate_required_fields(
        {
            "name": config.name,
            "description": config.description,
            "prompt": config.prompt,
            "model_id": config.model_id,
            "api_key": config.api_key,
        }
    )
    agent._assign_identity_and_model_config(
        config.name,
        config.description,
        config.prompt,
        config.system_prompt,
        config.start_prompt,
        config.end_prompt,
        tools,
        config.model_id,
        config.api_key,
        config.api_url,
        config.max_tokens,
        config.temperature,
        config.use_responses_api,
        config.reasoning_effort,
    )
    agent._assign_agent_topology(stages, subagents, config.next_agent)
    agent._configure_agent_logger(config.logging_level, config.logging_file, config.show_usage_level0)
    agent._assign_execution_limits(
        config.maxsteps,
        config.step_timeout,
        config.max_tool_rounds,
        config.max_tool_calls,
        config.max_step_extensions,
        config.extend_steps_by,
        config.max_stage_extensions,
        config.extend_stage_steps_by,
    )
    agent._assign_runtime_state(
        config.checkpoint,
        config.checkpoint_db_path,
        config.rate_limit_per_min,
        config.per_tool_rate_limit,
        config.memory,
        config.memory_access,
        config.final_answer_check,
        config.summarize_final,
        config.use_async,
    )


class IkaBaseAgent(AgentMemoryMixin, AgentToolsMixin, AgentExecutionMixin, AgentHelpersMixin, AgentInitializationMixin):
    def __init__(
        self,
        name: str, description: str, prompt: str,
        system_prompt: Optional[str] = None, start_prompt: Optional[str] = None, end_prompt: Optional[str] = None,
        tools: Optional[list[IkaTools]] = None,
        model_id: str = "", api_key: str = "", api_url: Optional[str] = None,
        max_tokens: int = 50000, temperature: float = 0.0, use_responses_api: bool = True,
        checkpoint: bool = False, checkpoint_db_path: str = "checkpoints.db",
        Stages: Optional[list[IkaStage]] = None,
        subagents: Optional[list[IkaBaseAgent]] = None,
        next_agent: Optional["IkaBaseAgent"] = None,
        maxsteps: int = 100, step_timeout: int = 900,
        rate_limit_per_min: Optional[float] = None, per_tool_rate_limit: Optional[dict[str, float]] = None,
        memory: bool = False, memory_access: Optional[dict[str, bool]] = None,
        final_answer_check: Optional[list[RawFinalAnswerCheck]] = None,
        logging_level: int = 0, logging_file: str = "logs.txt", show_usage_level0: bool = True,
        summarize_final: bool = False, use_async: bool = False,
        max_tool_rounds: Optional[int] = None, max_tool_calls: Optional[int] = None,
        max_step_extensions: int = 2, extend_steps_by: int = 3,
        max_stage_extensions: int = 2, extend_stage_steps_by: int = 3,
        reasoning_effort: Optional[str] = None,
    ):
        _initialize_base_agent(
            self,
            _AgentConstructorConfig(
                name=name,
                description=description,
                prompt=prompt,
                system_prompt=system_prompt,
                start_prompt=start_prompt,
                end_prompt=end_prompt,
                tools=tools,
                model_id=model_id,
                api_key=api_key,
                api_url=api_url,
                max_tokens=max_tokens,
                temperature=temperature,
                use_responses_api=use_responses_api,
                checkpoint=checkpoint,
                checkpoint_db_path=checkpoint_db_path,
                Stages=Stages,
                subagents=subagents,
                next_agent=next_agent,
                maxsteps=maxsteps,
                step_timeout=step_timeout,
                rate_limit_per_min=rate_limit_per_min,
                per_tool_rate_limit=per_tool_rate_limit,
                memory=memory,
                memory_access=memory_access,
                final_answer_check=final_answer_check,
                logging_level=logging_level,
                logging_file=logging_file,
                show_usage_level0=show_usage_level0,
                summarize_final=summarize_final,
                use_async=use_async,
                max_tool_rounds=max_tool_rounds,
                max_tool_calls=max_tool_calls,
                max_step_extensions=max_step_extensions,
                extend_steps_by=extend_steps_by,
                max_stage_extensions=max_stage_extensions,
                extend_stage_steps_by=extend_stage_steps_by,
                reasoning_effort=reasoning_effort,
            ),
        )
