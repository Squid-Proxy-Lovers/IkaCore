# pyright: strict

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, cast

USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens", "input_cached_tokens")


def _coerce_token_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


@dataclass
class UsageInfo:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cached_tokens: int = 0

    @classmethod
    def from_mapping(cls, usage: Optional[dict[str, Any]]) -> "UsageInfo":
        usage = usage or {}
        return cls(
            input_tokens=_coerce_token_count(usage.get("input_tokens")),
            output_tokens=_coerce_token_count(usage.get("output_tokens")),
            total_tokens=_coerce_token_count(usage.get("total_tokens")),
            input_cached_tokens=_coerce_token_count(usage.get("input_cached_tokens")),
        )

    def add_mapping(self, usage: Optional[dict[str, Any]]) -> None:
        usage = usage or {}
        self.input_tokens += _coerce_token_count(usage.get("input_tokens"))
        self.output_tokens += _coerce_token_count(usage.get("output_tokens"))
        self.total_tokens += _coerce_token_count(usage.get("total_tokens"))
        self.input_cached_tokens += _coerce_token_count(usage.get("input_cached_tokens"))

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cached_tokens": self.input_cached_tokens,
        }


@dataclass
class ChatLoopState:
    provider: str
    content: str
    reasoning_content: Optional[str]
    tool_calls: list[dict[str, Any]]
    tokens: int
    content_before_tools: str
    usage: UsageInfo
    executed_tool_calls: list[dict[str, Any]] = field(default_factory=lambda: [])
    rounds: int = 0
    total_tool_calls_in_cycle: int = 0
    hijacked: bool = False

    def add_usage(self, usage: Optional[dict[str, Any]]) -> None:
        self.usage.add_mapping(usage)


@dataclass
class ToolRuntimeState:
    call_counts: dict[str, int] = field(default_factory=lambda: {})
    recent_tool_calls: list[tuple[str, str]] = field(default_factory=lambda: [])
    current_step: int = 0
    total_tool_calls_in_cycle: int = 0

    @classmethod
    def from_model(cls, barebone_model: Any) -> "ToolRuntimeState":
        raw_recent_tool_calls = getattr(barebone_model, "_recent_tool_calls", None)
        recent_tool_calls: list[tuple[str, str]] = []
        if isinstance(raw_recent_tool_calls, list):
            for item in cast(list[object], raw_recent_tool_calls):
                if not isinstance(item, tuple):
                    continue
                values = cast(tuple[object, ...], item)
                if len(values) != 2:
                    continue
                tool_name, signature = values
                if isinstance(tool_name, str) and isinstance(signature, str):
                    recent_tool_calls.append((tool_name, signature))
        return cls(
            call_counts=dict(getattr(barebone_model, "_tool_call_counts", None) or {}),
            recent_tool_calls=recent_tool_calls,
            current_step=int(getattr(barebone_model, "_current_step", 0) or 0),
        )

    def apply_updated_counts(self, updated_counts: dict[str, int]) -> None:
        self.call_counts.update(updated_counts)

    def record_executed(self, executed_tool_calls: list[dict[str, Any]]) -> None:
        num_tools = len(executed_tool_calls)
        self.current_step += num_tools
        self.total_tool_calls_in_cycle += num_tools

    def sync_to_model(self, barebone_model: Any) -> None:
        barebone_model._tool_call_counts = dict(self.call_counts)
        barebone_model._recent_tool_calls = list(self.recent_tool_calls)
        barebone_model._current_step = self.current_step


@dataclass
class ChatResponsePayload:
    content: str
    reasoning_content: Optional[str]
    tool_calls: list[dict[str, Any]]
    executed_tool_calls: list[dict[str, Any]]
    content_before_tools: str
    message_history: dict[str, Any]
    usage: UsageInfo
    cost: Optional[dict[str, Any]]
    hijacked: bool
    interrupted: bool = False
    interrupt_data: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": self.tool_calls,
            "executed_tool_calls": self.executed_tool_calls,
            "content_before_tools": self.content_before_tools,
            "message_history": self.message_history,
            "usage": self.usage.to_dict(),
            "cost": self.cost,
            "hijacked": self.hijacked,
            "interrupted": self.interrupted,
            "interrupt_data": self.interrupt_data,
        }
