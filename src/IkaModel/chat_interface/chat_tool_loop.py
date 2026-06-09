"""Compatibility facade for chat tool-loop helpers."""

from __future__ import annotations

from .chat_tool_common import (
    ToolExecutionBatch,
    _append_tool_loop_messages,
    _apply_tool_execution_result,
    _tool_loop_control_completed,
    _tool_loop_interrupt_response,
    _tool_loop_post_batch_result,
)
from .chat_tool_loop_async import (
    _async_advance_tool_round,
    _async_continue_tool_loop_after_batch,
    _async_execute_tool_batch,
    _async_force_complete_if_max_tools,
    _async_repeated_tool_guard,
    _async_repeated_tool_response_for_state,
    _async_tool_loop,
    _async_tool_loop_iteration,
    _flush_async_remaining_tools,
)
from .chat_tool_loop_sync import (
    _flush_sync_remaining_tools,
    _sync_advance_tool_round,
    _sync_continue_tool_loop_after_batch,
    _sync_execute_tool_batch,
    _sync_force_complete_if_max_tools,
    _sync_repeated_tool_guard,
    _sync_repeated_tool_response_for_state,
    _sync_tool_loop,
    _sync_tool_loop_iteration,
)

__all__ = [
    "ToolExecutionBatch",
    "_append_tool_loop_messages",
    "_apply_tool_execution_result",
    "_async_advance_tool_round",
    "_async_continue_tool_loop_after_batch",
    "_async_execute_tool_batch",
    "_async_force_complete_if_max_tools",
    "_async_repeated_tool_guard",
    "_async_repeated_tool_response_for_state",
    "_async_tool_loop",
    "_async_tool_loop_iteration",
    "_flush_async_remaining_tools",
    "_flush_sync_remaining_tools",
    "_sync_advance_tool_round",
    "_sync_continue_tool_loop_after_batch",
    "_sync_execute_tool_batch",
    "_sync_force_complete_if_max_tools",
    "_sync_repeated_tool_guard",
    "_sync_repeated_tool_response_for_state",
    "_sync_tool_loop",
    "_sync_tool_loop_iteration",
    "_tool_loop_control_completed",
    "_tool_loop_interrupt_response",
    "_tool_loop_post_batch_result",
]
