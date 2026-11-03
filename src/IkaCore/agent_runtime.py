"""Compatibility facade for agent execution runtime mixins."""

# pyright: strict

from __future__ import annotations

from .agent_runtime_dispatch import (
    ExecutionDispatchRuntimeMixin,
    InterruptOutputMixin,
    SimpleExecutionDispatchMixin,
    StagedExecutionDispatchMixin,
    StagedExecutionStateMixin,
    StagedResumeMixin,
    ValidatedSimpleExecutionMixin,
)
from .agent_runtime_foundation import (
    AgentChatTurnMixin,
    RuntimePreparationMixin,
    SimpleInstructionMixin,
    SimpleRunExtensionMixin,
    SimpleRuntimePreparationMixin,
    SimpleRuntimeSupportMixin,
    SimpleStepExtensionMixin,
    StageExtensionMixin,
    StageInterruptMixin,
    StageRuntimeBuildMixin,
    StageRuntimePreparationMixin,
    StageRuntimeSupportMixin,
)
from .agent_runtime_simple_execution import (
    SimpleAgentEndMixin,
    SimpleExecutionRuntimeMixin,
    SimpleIterationMixin,
    SimpleLoopGuardMixin,
    SimpleProgressMixin,
)
from .agent_runtime_stage_execution import StageControlMixin, StageExecutionRuntimeMixin, StageExecutionStepMixin


class AgentExecutionMixin(ExecutionDispatchRuntimeMixin):
    """Compatibility aggregate for the execution runtime mixins."""

    pass


__all__ = [
    "AgentChatTurnMixin",
    "AgentExecutionMixin",
    "ExecutionDispatchRuntimeMixin",
    "InterruptOutputMixin",
    "RuntimePreparationMixin",
    "SimpleAgentEndMixin",
    "SimpleExecutionDispatchMixin",
    "SimpleExecutionRuntimeMixin",
    "SimpleInstructionMixin",
    "SimpleIterationMixin",
    "SimpleLoopGuardMixin",
    "SimpleProgressMixin",
    "SimpleRunExtensionMixin",
    "SimpleRuntimePreparationMixin",
    "SimpleRuntimeSupportMixin",
    "SimpleStepExtensionMixin",
    "StagedExecutionDispatchMixin",
    "StagedExecutionStateMixin",
    "StagedResumeMixin",
    "StageControlMixin",
    "StageExecutionRuntimeMixin",
    "StageExecutionStepMixin",
    "StageExtensionMixin",
    "StageInterruptMixin",
    "StageRuntimeBuildMixin",
    "StageRuntimePreparationMixin",
    "StageRuntimeSupportMixin",
    "ValidatedSimpleExecutionMixin",
]
