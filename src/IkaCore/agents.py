from __future__ import annotations

from typing import Any

from IkaCore.agent_execution import AgentExecutionMixin
from IkaCore.agent_helpers import AgentHelpersMixin
from IkaCore.agent_initialization import initialize_agent
from IkaCore.agent_memory import AgentMemoryMixin
from IkaCore.agent_snapshots import AgentSnapshotReplayMixin
from IkaCore.agent_state import AgentRunState
from IkaCore.agent_tools import AgentToolsMixin
from IkaCore.config import (
    AgentConfig,
    AgentTopology,
    ExecutionLimits,
    ExecutionPolicy,
    MemoryConfig,
    OutputConfig,
    PromptConfig,
    ProviderConfig,
)
from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools


class IkaBaseAgent(AgentMemoryMixin, AgentToolsMixin, AgentExecutionMixin, AgentHelpersMixin,
                   AgentSnapshotReplayMixin):
    state: AgentRunState

    def __init__(
        self, name: str, description: str, prompt: str, *,
        model_id: str = "", api_key: str = "", tools: list[IkaTools] | None = None,
        Stages: list[IkaStage] | None = None,
        provider: ProviderConfig | None = None, prompts: PromptConfig | None = None,
        limits: ExecutionLimits | None = None, policy: ExecutionPolicy | None = None,
        topology: AgentTopology | None = None, output: OutputConfig | None = None,
        memory: MemoryConfig | None = None,
    ):
        self.config = AgentConfig(name, description, prompt, model_id, api_key,
                                  tuple(tools or ()), tuple(Stages or ()), provider or ProviderConfig(), prompts or PromptConfig(),
                                  limits or ExecutionLimits(), policy or ExecutionPolicy(), topology or AgentTopology(),
                                  output or OutputConfig(), memory or MemoryConfig())
        initialize_agent(self)

    def shutdown(self) -> None:
        from IkaCore.agent_initialization import shutdown_agent
        shutdown_agent(self)

    @property
    def message_history(self) -> dict[str, Any]:
        return self.state.message_history

    @message_history.setter
    def message_history(self, value: dict[str, Any]) -> None:
        self.state.message_history = value

    @property
    def maxsteps(self) -> int:
        return self.state.maxsteps

    @maxsteps.setter
    def maxsteps(self, value: int) -> None:
        self.state.maxsteps = value

    @property
    def prompt(self) -> str:
        return self.state.prompt

    @prompt.setter
    def prompt(self, value: str) -> None:
        self.state.prompt = value

    @property
    def _resume_state(self) -> dict[str, Any] | None:
        return self.state._resume_state

    @_resume_state.setter
    def _resume_state(self, value: dict[str, Any] | None) -> None:
        self.state._resume_state = value

    @property
    def _last_interrupt(self) -> dict[str, Any] | None:
        return self.state._last_interrupt

    @_last_interrupt.setter
    def _last_interrupt(self, value: dict[str, Any] | None) -> None:
        self.state._last_interrupt = value

    @property
    def _tool_call_counts(self) -> dict[str, int]:
        return self.state._tool_call_counts

    @_tool_call_counts.setter
    def _tool_call_counts(self, value: dict[str, int]) -> None:
        self.state._tool_call_counts = value

    @property
    def _total_usage(self) -> dict[str, int]:
        return self.state._total_usage

    @_total_usage.setter
    def _total_usage(self, value: dict[str, int]) -> None:
        self.state._total_usage = value

    @property
    def _total_cost(self) -> dict[str, float]:
        return self.state._total_cost

    @_total_cost.setter
    def _total_cost(self, value: dict[str, float]) -> None:
        self.state._total_cost = value

    @property
    def _last_api_call_ts(self) -> float:
        return self.state._last_api_call_ts

    @_last_api_call_ts.setter
    def _last_api_call_ts(self, value: float) -> None:
        self.state._last_api_call_ts = value
