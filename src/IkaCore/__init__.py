# pyright: strict

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent as IkaBaseAgent
    from IkaCore.stages import IkaStage as IkaStage
    from IkaCore.tools import IkaTools as IkaTools
    from IkaCore.workflow import IkaWorkflow as IkaWorkflow
    from IkaCore.workflow import WorkflowEdge as WorkflowEdge
    from IkaCore.workflow import WorkflowNode as WorkflowNode
    from IkaCore.workflow import WorkflowResult as WorkflowResult

__all__ = [
    "IkaBaseAgent",
    "IkaStage",
    "IkaTools",
    "IkaWorkflow",
    "WorkflowEdge",
    "WorkflowNode",
    "WorkflowResult",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "IkaBaseAgent": ("IkaCore.agents", "IkaBaseAgent"),
    "IkaStage": ("IkaCore.stages", "IkaStage"),
    "IkaTools": ("IkaCore.tools", "IkaTools"),
    "IkaWorkflow": ("IkaCore.workflow", "IkaWorkflow"),
    "WorkflowEdge": ("IkaCore.workflow", "WorkflowEdge"),
    "WorkflowNode": ("IkaCore.workflow", "WorkflowNode"),
    "WorkflowResult": ("IkaCore.workflow", "WorkflowResult"),
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
