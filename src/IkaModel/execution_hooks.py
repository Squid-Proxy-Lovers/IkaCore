"""Context-scoped hooks for explicitly enabled snapshot runtimes."""

# pyright: strict

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional, cast

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ExecutionHooks:
    boundary: Callable[[str, JsonDict], Optional[str]]
    context: Callable[[], JsonDict]
    tool_started: Callable[[str, JsonDict, JsonDict], str]
    tool_finished: Callable[[str, object, Optional[BaseException]], None]


_hooks: ContextVar[Optional[ExecutionHooks]] = ContextVar("ika_execution_hooks", default=None)
_frame: ContextVar[Optional[str]] = ContextVar("ika_tool_frame", default=None)


def hooks_enabled() -> bool:
    return _hooks.get() is not None


def current_execution_context() -> JsonDict:
    hooks = _hooks.get()
    return hooks.context() if hooks is not None else {}


def current_tool_frame() -> Optional[str]:
    return _frame.get()


def emit_boundary(label: str, payload: Optional[JsonDict] = None) -> Optional[str]:
    hooks = _hooks.get()
    return hooks.boundary(label, payload or {}) if hooks is not None else None


@contextmanager
def execution_hooks(hooks: ExecutionHooks) -> Generator[None, None, None]:
    token = _hooks.set(hooks)
    try:
        yield
    finally:
        _hooks.reset(token)


def _consume_generator(generator: Generator[Any, Any, Any]) -> object:
    try:
        while True:
            item = next(generator)
            if not isinstance(item, dict) or "__ika_checkpoint__" not in item:
                raise ValueError("generator tools must yield explicit checkpoint records")
            raw_checkpoint = cast(JsonDict, item)["__ika_checkpoint__"]
            if not isinstance(raw_checkpoint, dict):
                raise ValueError("tool checkpoint must be an object")
            checkpoint = cast(JsonDict, raw_checkpoint)
            emit_boundary("tool:" + str(checkpoint.get("label", "checkpoint")), checkpoint.get("payload") or {})
    except StopIteration as completed:
        return completed.value
    finally:
        generator.close()


def invoke_tool(function: Callable[[JsonDict], Any], args: JsonDict, name: str, metadata: JsonDict) -> object:
    hooks = _hooks.get()
    if hooks is None:
        return function(args)
    frame_id = hooks.tool_started(name, args, metadata)
    token = _frame.set(frame_id)
    try:
        emit_boundary("pre_tool", {"tool_name": name, "arguments": args})
        result = function(args)
        if inspect.isgenerator(result):
            result = _consume_generator(result)
        hooks.tool_finished(frame_id, result, None)
        emit_boundary("post_tool", {"tool_name": name, "result": result})
        return result
    except BaseException as error:
        hooks.tool_finished(frame_id, None, error)
        raise
    finally:
        _frame.reset(token)


async def invoke_async_tool(function: Callable[[JsonDict], Awaitable[Any]], args: JsonDict, name: str,
                            metadata: JsonDict) -> object:
    hooks = _hooks.get()
    if hooks is None:
        return await function(args)
    frame_id = hooks.tool_started(name, args, metadata)
    token = _frame.set(frame_id)
    try:
        emit_boundary("pre_tool", {"tool_name": name, "arguments": args})
        result = await function(args)
        hooks.tool_finished(frame_id, result, None)
        emit_boundary("post_tool", {"tool_name": name, "result": result})
        return result
    except BaseException as error:
        hooks.tool_finished(frame_id, None, error)
        raise
    finally:
        _frame.reset(token)
