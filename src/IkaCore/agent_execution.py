"""Agent execution with optional snapshot recording."""

# pyright: strict

from __future__ import annotations

from typing import Any, Dict, Optional

from IkaCore.agent_runtime_dispatch import ExecutionDispatchRuntimeMixin
from IkaCore.agent_snapshot_execution import execute_with_snapshots


class AgentExecutionMixin(ExecutionDispatchRuntimeMixin):


    def execution(self, resume_input: Optional[str] = None) -> Dict[str, Any]:
        return execute_with_snapshots(self, super().execution, resume_input)


__all__ = ["AgentExecutionMixin"]
