"""Immutable agent configuration. Mutable counters and transcripts live in AgentRunState."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .events import EventSubscriber
from .results import AgentResult
from .stages import IkaStage
from .tools import IkaTools

if TYPE_CHECKING:
    from .agents import IkaBaseAgent


def _integer(name: str, value: object, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name: str, value: object, *, positive: bool = False, maximum: float | None = None) -> None:
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value)
            or (value <= 0 if positive else value < 0) or (maximum is not None and value > maximum)):
        domain = "> 0" if positive else ">= 0"
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a finite number {domain}{upper}")


def _string(name: str, value: object, *, optional: bool = False) -> None:
    if not isinstance(value, str) and not (optional and value is None):
        raise ValueError(f"{name} must be a string" + (" or None" if optional else ""))


def _boolean(name: str, value: object) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")


def _sequence(name: str, value: object) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")


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
        _string("api_url", self.api_url, optional=True)
        _integer("max_tokens", self.max_tokens, 1)
        _number("temperature", self.temperature, maximum=1)
        _boolean("use_responses_api", self.use_responses_api)
        _string("reasoning_effort", self.reasoning_effort, optional=True)
        if self.thinking_budget_tokens is not None:
            _integer("thinking_budget_tokens", self.thinking_budget_tokens, 1024)
            if self.thinking_budget_tokens >= self.max_tokens:
                raise ValueError("thinking_budget_tokens must be less than max_tokens")
        if self.client_factory is not None and not callable(self.client_factory):
            raise ValueError("client_factory must be callable or None")
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

    def __post_init__(self) -> None:
        for name in ("system_prompt", "start_prompt", "end_prompt"):
            _string(name, getattr(self, name), optional=True)


@dataclass(frozen=True)
class ExecutionLimits:
    maxsteps: int = 100
    step_timeout: float = 900
    max_tool_rounds: int = 5
    max_tool_calls: int | None = None
    rate_limit_per_min: float | None = None
    context_budget: int | None = None
    per_tool_rate_limit: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("maxsteps", "max_tool_rounds"):
            _integer(name, getattr(self, name))
        _number("step_timeout", self.step_timeout, positive=True)
        if self.max_tool_calls is not None:
            _integer("max_tool_calls", self.max_tool_calls)
        if self.context_budget is not None:
            _integer("context_budget", self.context_budget, 1)
        if self.rate_limit_per_min is not None:
            _number("rate_limit_per_min", self.rate_limit_per_min, positive=True)
        if not isinstance(self.per_tool_rate_limit, Mapping):
            raise ValueError("per_tool_rate_limit must be a mapping of tool names to positive rates")
        for name, rate in self.per_tool_rate_limit.items():
            _string("per_tool_rate_limit key", name)
            _number(f"per_tool_rate_limit[{name!r}]", rate, positive=True)
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
    async_requests: bool = False

    def __post_init__(self) -> None:
        for name in ("max_step_extensions", "extend_steps_by", "max_stage_extensions",
                     "extend_stage_steps_by", "validation_retries", "validation_extra_steps"):
            _integer(name, getattr(self, name))
        _boolean("fail_closed", self.fail_closed)
        _boolean("async_requests", self.async_requests)
        _sequence("final_answer_checks", self.final_answer_checks)
        if any(not callable(check) for check in self.final_answer_checks):
            raise ValueError("final_answer_checks must contain callables")
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
        _sequence("subagents", self.subagents)
        if self.subagents or self.next_agent is not None:
            from .agents import IkaBaseAgent
            if any(not isinstance(agent, IkaBaseAgent) for agent in self.subagents):
                raise ValueError("subagents must contain IkaBaseAgent instances")
            if self.next_agent is not None and not isinstance(self.next_agent, IkaBaseAgent):
                raise ValueError("next_agent must be an IkaBaseAgent or None")
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
        for name in ("cli", "show_usage_level0", "summarize_final"):
            _boolean(name, getattr(self, name))
        _integer("logging_level", self.logging_level)
        if self.logging_level > 3:
            raise ValueError("logging_level must be an integer from 0 to 3")
        _string("logging_file", self.logging_file)
        _sequence("subscribers", self.subscribers)
        if any(not callable(subscriber) for subscriber in self.subscribers):
            raise ValueError("subscribers must contain callables")
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


    def __post_init__(self) -> None:
        for name in ("name", "description", "prompt", "model_id", "api_key"):
            _string(name, getattr(self, name))
        for name, kind in (("provider", ProviderConfig), ("prompts", PromptConfig),
                           ("limits", ExecutionLimits), ("policy", ExecutionPolicy),
                           ("topology", AgentTopology), ("output", OutputConfig)):
            if not isinstance(getattr(self, name), kind):
                raise ValueError(f"{name} must be a {kind.__name__}")
        for name, kind in (("tools", IkaTools), ("stages", IkaStage)):
            values = getattr(self, name)
            _sequence(name, values)
            if any(not isinstance(value, kind) for value in values):
                raise ValueError(f"{name} must contain {kind.__name__} instances")
            object.__setattr__(self, name, tuple(values))
