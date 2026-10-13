import asyncio
import json
import threading

import pytest

from IkaCore import IkaTools
from IkaCore.tool_runtime_metadata import runtime_executor
from IkaModel.chat_interface.tool_execution_async import async_execute_tool
from IkaModel.chat_interface.tool_execution_sync import execute_tool
from IkaModel.execution_hooks import ExecutionHooks, execution_hooks, invoke_async_tool, invoke_tool


def test_tool_metadata_wrapper_is_private_to_one_tool_and_default_function_is_unchanged():
    def function(args):
        return 'done'
    original = IkaTools('default', 'test', {'type': 'object'}, execute_function=function)
    limited = IkaTools('limited', 'test', {'type': 'object'}, execute_function=function, timeout=0.01, replay_policy='allow')
    assert runtime_executor(original, function) is function
    wrapped = runtime_executor(limited, function)
    assert wrapped is not function and wrapped({}) == 'done'
    assert wrapped.__tool_timeout__ == 0.01
    assert wrapped.__ika_runtime_metadata__['replay_policy'] == 'allow'
    assert not hasattr(function, '__tool_timeout__')


def test_sync_tool_timeout_returns_at_configured_limit_and_worker_is_released():
    release = threading.Event()
    entered = threading.Event()
    def function(args):
        entered.set()
        release.wait(1)
        return 'late'
    tool = IkaTools('limited', 'test', {'type': 'object'}, execute_function=function, timeout=0.03)
    try:
        result = json.loads(execute_tool('limited', {}, {'limited': runtime_executor(tool, function)}, timeout=10))
        assert entered.is_set() and '0.03s' in result['error']
    finally:
        release.set()


def test_async_tool_timeout_cancels_coroutine_and_reports_its_configured_limit():
    cancelled = []
    async def function(args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    tool = IkaTools('limited', 'test', {'type': 'object'}, execute_function=function, timeout=0.01)
    wrapped = runtime_executor(tool, function)
    result = json.loads(asyncio.run(async_execute_tool('limited', {}, {'limited': wrapped}, timeout=10)))
    assert '0.01s' in result['error'] and cancelled == [True]


def test_async_snapshot_hooks_record_completion_and_failure_without_swallowing():
    events = []
    hooks = ExecutionHooks(lambda label, payload: events.append(label), lambda: {}, lambda *args: 'tool-frame',
                           lambda frame, result, error: events.append((result, type(error) if error else None)))
    async def success(args):
        return 'done'
    async def failure(args):
        raise ValueError('expected')
    async def run():
        with execution_hooks(hooks):
            assert await invoke_async_tool(success, {}, 'success', {}) == 'done'
            with pytest.raises(ValueError):
                await invoke_async_tool(failure, {}, 'failure', {})
    asyncio.run(run())
    assert events == ['pre_tool', ('done', None), 'post_tool', 'pre_tool', (None, ValueError)]


@pytest.mark.parametrize('yielded', ['ordinary', {'__ika_checkpoint__': 'invalid'}])
def test_generator_tools_require_explicit_object_checkpoint_records(yielded):
    def function(args):
        yield yielded
    hooks = ExecutionHooks(lambda *args: None, lambda: {}, lambda *args: 'frame', lambda *args: None)
    with execution_hooks(hooks), pytest.raises(ValueError):
        invoke_tool(function, {}, 'generator', {})
