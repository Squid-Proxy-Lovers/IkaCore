"""Shared helpers for sync and async chat tool loops."""

# pyright: strict
# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from typing import Any, Optional

from IkaCore.agent_runtime_payloads import JsonDict

from ..base import BareBoneModel
from .chat_response import (
    _append_tool_messages,
    _handle_control_tool_completion,
    _interrupt_response_if_needed,
    _remember_tool_calls,
    _repeated_tool_warning,
)
from .types import ChatLoopState, ToolRuntimeState

MessageList = list[JsonDict]
ToolExecutionBatch = tuple[MessageList, list[str], dict[str, int], list[JsonDict], Optional[JsonDict]]


def _apply_tool_execution_result(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    tool_runtime: ToolRuntimeState,
    all_executed_tool_call_list: list[JsonDict],
    updated_counts: dict[str, int],
    executed_tool_call_list: list[JsonDict],
    tool_results: list[str],
    remember_signatures: bool,
) -> None:
    tool_runtime.apply_updated_counts(updated_counts)
    all_executed_tool_call_list.extend(executed_tool_call_list)
    tool_runtime.record_executed(executed_tool_call_list)
    if remember_signatures:
        _remember_tool_calls(executed_tool_call_list, tool_runtime.recent_tool_calls)
    tool_runtime.sync_to_model(barebone_model)
    if logger:
        logger.log_tool_results(executed_tool_call_list, tool_results)


def _tool_loop_interrupt_response(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    messages: MessageList,
    message_history: JsonDict,
    state: ChatLoopState,
    batch: ToolExecutionBatch,
    hitl_stage_name: Optional[str] = None,
) -> Optional[JsonDict]:
    tool_messages, tool_results, _, executed_tool_call_list, interrupt_data = batch
    return _interrupt_response_if_needed(
        barebone_model,
        logger,
        interrupt_data,
        state.provider,
        messages,
        message_history,
        state.content,
        state.reasoning_content,
        executed_tool_call_list,
        state.executed_tool_calls,
        tool_messages,
        tool_results,
        state.tokens,
        state.content_before_tools,
        state.usage,
        state.hijacked,
        hitl_stage_name=hitl_stage_name,
    )


def _tool_loop_control_completed(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    message_history: JsonDict,
    state: ChatLoopState,
    batch: ToolExecutionBatch,
) -> bool:
    return _handle_control_tool_completion(
        barebone_model,
        logger,
        state.content,
        state.reasoning_content,
        batch[3],
        state.executed_tool_calls,
        state.content_before_tools,
        message_history,
        state.usage,
        state.hijacked,
        state.tokens,
    )


def _append_tool_loop_messages(
    messages: MessageList,
    message_history: JsonDict,
    state: ChatLoopState,
    batch: ToolExecutionBatch,
    repeated_tool_names: Optional[list[str]] = None,
) -> None:
    tool_messages, tool_results, _, executed_tool_call_list, _ = batch
    _append_tool_messages(
        state.provider,
        messages,
        message_history,
        state.content,
        state.reasoning_content,
        executed_tool_call_list,
        tool_messages,
        tool_results,
        state.tokens,
        _repeated_tool_warning(repeated_tool_names or []),
    )


def _tool_loop_post_batch_result(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    messages: MessageList,
    message_history: JsonDict,
    state: ChatLoopState,
    batch: ToolExecutionBatch,
    trace_label: str,
    hitl_stage_name: Optional[str] = None,
) -> Optional[tuple[Optional[JsonDict], bool]]:
    interrupt_response = _tool_loop_interrupt_response(
        barebone_model,
        logger,
        messages,
        message_history,
        state,
        batch,
        hitl_stage_name=hitl_stage_name,
    )
    if interrupt_response:
        return interrupt_response, False

    import logging

    logging.getLogger(__name__).debug(f"[TRACE] {trace_label}: Checking agent_end")
    if _tool_loop_control_completed(barebone_model, logger, message_history, state, batch):
        state.tool_calls = []
        return None, False
    return None
