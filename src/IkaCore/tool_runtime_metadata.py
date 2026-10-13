"""Attach opt-in execution metadata without mutating shared user functions."""

# pyright: strict

import inspect
from functools import wraps
from typing import Any

from .tools import ToolExecutor


def runtime_executor(tool: Any, function: ToolExecutor) -> ToolExecutor:
    if not any(hasattr(tool, name) for name in ("timeout", "replay_policy", "side_effect_type")):
        return function
    if inspect.iscoroutinefunction(function):
        @wraps(function)
        async def async_call(args: dict[str, Any]) -> Any:
            return await function(args)
        wrapper = async_call
    else:
        @wraps(function)
        def call(args: dict[str, Any]) -> Any:
            return function(args)
        wrapper = call
    if hasattr(tool, "timeout"):
        setattr(wrapper, "__tool_timeout__", tool.timeout)
    setattr(wrapper, "__ika_runtime_metadata__", {
        "replay_policy": getattr(tool, "replay_policy", "deny"),
        "side_effect_type": getattr(tool, "side_effect_type", "unknown"),
    })
    return wrapper
