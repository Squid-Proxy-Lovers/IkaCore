"""Immutable agent configuration. Mutable counters and transcripts live in AgentRunState."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agents import IkaBaseAgent
    from .stages import IkaStage
    from .tools import IkaTools


@dataclass(frozen=True)
class ProviderConfig:
    api_url: str | None = None
    max_tokens: int = 50000
    temperature: float = 0.0
    use_responses_api: bool = True
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str | None = None
    start_prompt: str | None = None
    end_prompt: str | None = None


@dataclass(frozen=True)
class ExecutionLimits:
    maxsteps: int = 100
    step_timeout: int = 900
    max_tool_rounds: int = 5
    max_tool_calls: int | None = None
    rate_limit_per_min: float | None = None
    per_tool_rate_limit: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_tool_rate_limit", MappingProxyType(dict(self.per_tool_rate_limit)))


@dataclass(frozen=True)
class ExecutionPolicy:
    """Checks and bounded automatic continuations, with the established v2 defaults."""
    final_answer_checks: tuple[Callable[[dict[str, Any]], object], ...] = ()
    max_step_extensions: int = 2
    extend_steps_by: int = 3
    max_stage_extensions: int = 2
    extend_stage_steps_by: int = 3
    validation_retries: int = 3
    validation_extra_steps: int = 10
    use_async: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_answer_checks", tuple(self.final_answer_checks))


@dataclass(frozen=True)
class AgentTopology:
    subagents: tuple[IkaBaseAgent, ...] = ()
    next_agent: IkaBaseAgent | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subagents", tuple(self.subagents))


@dataclass(frozen=True)
class OutputConfig:
    logging_level: int = 0
    logging_file: str = "logs.txt"
    show_usage_level0: bool = True
    summarize_final: bool = False


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = False
    access: Mapping[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    name: str
    description: str
    prompt: str
    model_id: str
    api_key: str = field(repr=False)
    tools: tuple[IkaTools, ...] = ()
    stages: tuple[IkaStage, ...] = ()
    provider: ProviderConfig = ProviderConfig()
    prompts: PromptConfig = PromptConfig()
    limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    policy: ExecutionPolicy = ExecutionPolicy()
    topology: AgentTopology = AgentTopology()
    output: OutputConfig = OutputConfig()
    memory: MemoryConfig = field(default_factory=MemoryConfig)
