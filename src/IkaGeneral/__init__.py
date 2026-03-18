"""IkaGeneral: Continuous General Agent Execution System.

Provides a higher-level execution environment on top of IkaCore with:
- Context packs for managing information across execution
- Recursive sub-agents with configurable models
- Dynamic tool creation at runtime
- Parallel branching for exploration
- Checkpoint/reset for execution state management
"""

from .models import (
    CcpContextConfig,
    ContextPack,
    ExecutionCheckpoint,
    SubAgentDefinition,
    DynamicToolDefinition,
)
from .execution_env import IkaExecutionEnvironment

__all__ = [
    "IkaExecutionEnvironment",
    "CcpContextConfig",
    "ContextPack",
    "ExecutionCheckpoint",
    "SubAgentDefinition",
    "DynamicToolDefinition",
]
