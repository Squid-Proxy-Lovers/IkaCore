from importlib import import_module
from typing import Any

_EXPORTS = {
    "IkaBaseAgent": ("IkaCore.agents", "IkaBaseAgent"),
    "IkaStage": ("IkaCore.stages", "IkaStage"),
    "IkaTools": ("IkaCore.tools", "IkaTools"),
    "IkaWorkflow": ("IkaCore.workflow", "IkaWorkflow"),
    "WorkflowEdge": ("IkaCore.workflow", "WorkflowEdge"),
    "WorkflowNode": ("IkaCore.workflow", "WorkflowNode"),
    "WorkflowResult": ("IkaCore.workflow", "WorkflowResult"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
