"""Public agent execution mixin compatibility facade."""

# pyright: strict

from __future__ import annotations

from typing import Any, Dict, Optional

from IkaCore.agent_runtime import AgentExecutionMixin as _AgentExecutionRuntimeMixin
from IkaCore.execution_types import StageExecutionResult


class AgentExecutionMixin(_AgentExecutionRuntimeMixin):
    def execute_stage(
        self,
        stage_index: int,
        remaining_steps: int,
        resume_input: Optional[str] = None,
    ) -> StageExecutionResult:
        return super().execute_stage(stage_index, remaining_steps, resume_input=resume_input)

    def run_simple(self) -> tuple[str, str]:
        return super().run_simple()

    def execution(self, checkpoint_uid: Optional[str] = None, resume_input: Optional[str] = None) -> Dict[str, Any]:
        return super().execution(checkpoint_uid=checkpoint_uid, resume_input=resume_input)


__all__ = ["AgentExecutionMixin"]
