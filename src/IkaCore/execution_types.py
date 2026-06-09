# pyright: strict

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

StageTarget = Union[int, str]


@dataclass(frozen=True)
class StageExecutionResult:
    next_stage_index: int
    last_content: str
    agent_end_called: bool
    end_text: Optional[str]
    used_steps: int

    def __iter__(self) -> Iterator[object]:
        yield self.next_stage_index
        yield self.last_content
        yield self.agent_end_called
        yield self.end_text
        yield self.used_steps


@dataclass
class AgentChatTurn:
    response: dict[str, Any]
    agent_end_exception: bool
    last_content: str
    content_before_tools: str
    tool_calls: list[dict[str, Any]]
    executed_tool_calls: list[dict[str, Any]]


@dataclass
class StageRuntimeContext:
    stage: Any
    system_prompt: str
    content_prompt: str
    current_hierarchy: list[str]
    barebone_model: Any
    messages: list[dict[str, Any]]
    tool_executors: dict[str, Any]
    step_limit: int


@dataclass
class StagedExecutionState:
    stage_idx: int
    remaining_steps: int
    last_content: str
    resume_stage_input: Optional[str] = None
    resumed_stage_index: Optional[int] = None
    agent_end_text: Optional[str] = None


@dataclass
class SimpleRuntimeContext:
    system_prompt: str
    current_hierarchy: list[str]
    barebone_model: Any
    messages: list[dict[str, Any]]
    tool_executors: dict[str, Any]
    cli: Any
