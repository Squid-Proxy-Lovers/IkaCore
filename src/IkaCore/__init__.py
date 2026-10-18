# pyright: strict

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent as IkaBaseAgent
    from IkaCore.config import AgentConfig as AgentConfig
    from IkaCore.config import AgentTopology as AgentTopology
    from IkaCore.config import ExecutionLimits as ExecutionLimits
    from IkaCore.config import ExecutionPolicy as ExecutionPolicy
    from IkaCore.config import MemoryConfig as MemoryConfig
    from IkaCore.config import OutputConfig as OutputConfig
    from IkaCore.config import PromptConfig as PromptConfig
    from IkaCore.config import ProviderConfig as ProviderConfig
    from IkaCore.runtime_control import RuntimeControl as RuntimeControl
    from IkaCore.snapshot_store import SnapshotStore as SnapshotStore
    from IkaCore.stages import IkaStage as IkaStage
    from IkaCore.tools import IkaTools as IkaTools
    from IkaCore.workflow import IkaWorkflow as IkaWorkflow
    from IkaCore.workflow import WorkflowEdge as WorkflowEdge
    from IkaCore.workflow import WorkflowNode as WorkflowNode
    from IkaCore.workflow import WorkflowResult as WorkflowResult

try:
    __version__ = version("ikacore")
except PackageNotFoundError:
    __version__ = "2.0.0"

__all__ = [
    "AgentConfig",
    "ProviderConfig",
    "PromptConfig",
    "ExecutionLimits",
    "ExecutionPolicy",
    "AgentTopology",
    "OutputConfig",
    "MemoryConfig",

    "__version__",
    "IkaBaseAgent",
    "IkaStage",
    "IkaTools",
    "IkaWorkflow",
    "WorkflowEdge",
    "WorkflowNode",
    "WorkflowResult",
    "SnapshotStore",
    "RuntimeControl",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentConfig": ("IkaCore.config", "AgentConfig"),
    "ProviderConfig": ("IkaCore.config", "ProviderConfig"),
    "PromptConfig": ("IkaCore.config", "PromptConfig"),
    "ExecutionLimits": ("IkaCore.config", "ExecutionLimits"),
    "ExecutionPolicy": ("IkaCore.config", "ExecutionPolicy"),
    "AgentTopology": ("IkaCore.config", "AgentTopology"),
    "OutputConfig": ("IkaCore.config", "OutputConfig"),
    "MemoryConfig": ("IkaCore.config", "MemoryConfig"),

    "IkaBaseAgent": ("IkaCore.agents", "IkaBaseAgent"),
    "IkaStage": ("IkaCore.stages", "IkaStage"),
    "IkaTools": ("IkaCore.tools", "IkaTools"),
    "IkaWorkflow": ("IkaCore.workflow", "IkaWorkflow"),
    "WorkflowEdge": ("IkaCore.workflow", "WorkflowEdge"),
    "WorkflowNode": ("IkaCore.workflow", "WorkflowNode"),
    "WorkflowResult": ("IkaCore.workflow", "WorkflowResult"),
    "SnapshotStore": ("IkaCore.snapshot_store", "SnapshotStore"),
    "RuntimeControl": ("IkaCore.runtime_control", "RuntimeControl"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
