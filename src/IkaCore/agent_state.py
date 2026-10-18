"""Live per-agent state; configuration is held separately in frozen AgentConfig."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRunState:
    message_history: dict[str, Any] = field(default_factory=dict)
    maxsteps: int = 100
    prompt: str = ""
    turn_message: str | None = None
    compaction_count: int = 0
    validation_attempts: int = 0
    validation_maxsteps: int | None = None
    _resume_state: dict[str, Any] | None = None
    _last_interrupt: dict[str, Any] | None = None
    _tool_call_counts: dict[str, int] = field(default_factory=dict)
    _total_usage: dict[str, int] = field(default_factory=dict)
    _total_cost: dict[str, float] = field(default_factory=dict)
    _last_api_call_ts: float = 0.0
