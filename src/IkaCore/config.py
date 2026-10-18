"""Immutable agent configuration. Mutable counters and transcripts live in AgentRunState."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .results import AgentResult

if TYPE_CHECKING:
    from .agents import IkaBaseAgent
    from .events import EventSubscriber
    from .stages import IkaStage
    from .tools import IkaTools


@dataclass(frozen=True)
class ProviderConfig:
    api_url: str | None = None
    max_tokens: int = 50000
    temperature: float = 0.0
    use_responses_api: bool = True
    reasoning_effort: str | None = None
    thinking_budget_tokens: int | None = None
    client: Any = field(default=None, repr=False, compare=False)
    client_factory: Callable[[], Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.client is not None and self.client_factory is not None:
            raise ValueError("Choose a borrowed client or an owned client_factory")

    @classmethod
    def bounded(cls, **overrides: Any) -> ProviderConfig:
        return cls(**{"max_tokens": 1024, **overrides})


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
    context_budget: int | None = None
    per_tool_rate_limit: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_budget is not None and self.context_budget <= 0:
            raise ValueError("context_budget must be a positive token count")
        object.__setattr__(self, "per_tool_rate_limit", MappingProxyType(dict(self.per_tool_rate_limit)))


    @classmethod
    def bounded(cls, **overrides: Any) -> ExecutionLimits:
        return cls(**{"maxsteps": 6, "step_timeout": 30, "max_tool_rounds": 3, "max_tool_calls": 12, **overrides})


@dataclass(frozen=True)
class ExecutionPolicy:
    """Checks and bounded automatic continuations, with the established v2 defaults."""
    final_answer_checks: tuple[Callable[[AgentResult], bool], ...] = ()
    max_step_extensions: int = 2
    extend_steps_by: int = 3
    max_stage_extensions: int = 2
    extend_stage_steps_by: int = 3
    validation_retries: int = 3
    validation_extra_steps: int = 10
    fail_closed: bool = False
    use_async: bool = False

    def __post_init__(self) -> None:
        if self.validation_retries < 0 or self.validation_extra_steps < 0:
            raise ValueError("validation retry and step allowances must be nonnegative")
        object.__setattr__(self, "final_answer_checks", tuple(self.final_answer_checks))


    @classmethod
    def bounded(cls, **overrides: Any) -> ExecutionPolicy:
        return cls(**{"max_step_extensions": 0, "max_stage_extensions": 0,
                      "validation_retries": 1, "validation_extra_steps": 2, **overrides})


@dataclass(frozen=True)
class AgentTopology:
    subagents: tuple[IkaBaseAgent, ...] = ()
    next_agent: IkaBaseAgent | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subagents", tuple(self.subagents))


@dataclass(frozen=True)
class OutputConfig:
    cli: bool = True
    subscribers: tuple[EventSubscriber, ...] = ()
    logging_level: int = 0
    logging_file: str = "logs.txt"
    show_usage_level0: bool = True
    summarize_final: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "subscribers", tuple(self.subscribers))


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
