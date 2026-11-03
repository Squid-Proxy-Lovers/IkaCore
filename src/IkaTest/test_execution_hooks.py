"""Execution and tool identities remain scoped across nested agents."""

import pytest

from IkaModel.execution_hooks import ExecutionHooks, current_tool_frame, execution_hooks, invoke_tool


@pytest.mark.parametrize('fail', [False, True])
def test_nested_hook_scope_clears_and_restores_tool_identity(fail):
    def hooks(frame):
        return ExecutionHooks(lambda *args: None, lambda: {}, lambda *args: frame, lambda *args: None)
    def outer(args):
        assert current_tool_frame() == 'outer'
        try:
            with execution_hooks(hooks('inner')):
                assert current_tool_frame() is None
                assert invoke_tool(lambda args: current_tool_frame(), {}, 'inner', {}) == 'inner'
                assert current_tool_frame() is None
                if fail:
                    raise ValueError('child failure')
        except ValueError:
            pass
        assert current_tool_frame() == 'outer'
    with execution_hooks(hooks('outer')):
        invoke_tool(outer, {}, 'outer', {})
        assert current_tool_frame() is None
    assert current_tool_frame() is None
