"""Synchronous chat tool-loop coordination."""

# pyright: strict
# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from IkaCore.agent_runtime_payloads import JsonDict

from ..base import BareBoneModel
from ..summarization import run_summarization
from .chat_request import _request_provider_round
from .chat_response import (
    UsageInput,
    _build_repeated_tool_hijack_response,
    _build_tool_metadata,
    _find_repeated_tool_calls,
    _force_complete_with_summary,
)
from .chat_tool_common import (
    ToolExecutionBatch,
    _append_tool_loop_messages,
    _apply_tool_execution_result,
    _tool_loop_control_completed,
    _tool_loop_interrupt_response,
    _tool_loop_post_batch_result,
)
from .response_interface import execute_tool_calls
from .tool_execution_sync import ToolExecutorMap
from .types import ChatLoopState, ToolRuntimeState

LOG = logging.getLogger(__name__)


def _sync_repeated_tool_guard(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    tool_calls: list[JsonDict],
    tool_runtime: ToolRuntimeState,
    message_history: JsonDict,
    total_usage: UsageInput,
    all_executed_tool_call_list: list[JsonDict],
    content_before_tools: str,
    client: Optional[httpx.Client],
) -> tuple[Optional[JsonDict], list[str]]:
    repeated_events = _find_repeated_tool_calls(tool_calls, tool_runtime.recent_tool_calls)
    repeated_tool_names = [name for name, _ in repeated_events]
    for tool_name, recent_count in repeated_events:
        LOG.error(
            f"Tool '{tool_name}' has been called {recent_count} times with identical arguments. "
            "Agent may be stuck in a loop."
        )
        if recent_count >= 5:
            LOG.error(
                f"Tool '{tool_name}' called {recent_count} times identically. "
                "Forcing agent termination with summarization."
            )
            force_answer = run_summarization(
                barebone_model,
                message_history,
                prompt_kind="force_answer",
                write_to_history=False,
                use_same_model=True,
                client=client,
            )
            return (
                _build_repeated_tool_hijack_response(
                    barebone_model,
                    logger,
                    tool_name,
                    recent_count,
                    force_answer,
                    content_before_tools,
                    message_history,
                    total_usage,
                    all_executed_tool_call_list,
                ),
                repeated_tool_names,
            )
    return None, repeated_tool_names


def _sync_execute_tool_batch(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
    tool_executors: ToolExecutorMap,
    timeout: float,
    remember_signatures: bool,
) -> ToolExecutionBatch:
    tool_metadata = _build_tool_metadata(barebone_model)
    agent_hierarchy = getattr(barebone_model, "agent_hierarchy", None)
    batch = execute_tool_calls(
        state.tool_calls,
        tool_executors,
        state.provider,
        timeout,
        tool_metadata,
        agent_hierarchy,
        tool_runtime.current_step,
        tool_runtime.call_counts,
    )
    _apply_tool_execution_result(
        barebone_model,
        logger,
        tool_runtime,
        state.executed_tool_calls,
        batch[2],
        batch[3],
        batch[1],
        remember_signatures=remember_signatures,
    )
    return batch


def _sync_force_complete_if_max_tools(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    message_history: JsonDict,
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
    max_tool_calls: Optional[int],
) -> None:
    if not (max_tool_calls and tool_runtime.current_step >= max_tool_calls):
        return
    LOG.debug("[TRACE] chat: Max tool calls reached")
    LOG.warning(f"Max tool calls ({max_tool_calls}) reached. Force-completing agent locally.")
    _force_complete_with_summary(
        barebone_model=barebone_model,
        message_history=message_history,
        total_usage=state.usage,
        logger=logger,
        all_executed_tool_call_list=state.executed_tool_calls,
        content_before_tools=state.content_before_tools,
        content=state.content,
        reason=f"Maximum tool call limit ({max_tool_calls}) reached at step {tool_runtime.current_step}.",
        hijacked=True,
    )


def _sync_advance_tool_round(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    timeout: float,
    client: Optional[httpx.Client],
    max_tool_rounds: int,
    state: ChatLoopState,
) -> bool:
    state.rounds += 1
    if state.rounds >= max_tool_rounds:
        LOG.debug("[TRACE] chat: Max tool rounds reached")
        state.tool_calls = []
        return False

    LOG.debug("[TRACE] chat: Sending follow-up API request")
    state.content, state.reasoning_content, state.tool_calls, state.tokens, usage_info = _request_provider_round(
        state.provider,
        barebone_model,
        messages,
        message_history,
        timeout,
        client,
        usage_token_fallback=0,
    )
    if usage_info:
        state.add_usage(usage_info)
    return True


def _sync_repeated_tool_response_for_state(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
    message_history: JsonDict,
    client: Optional[httpx.Client],
) -> tuple[Optional[JsonDict], list[str]]:
    return _sync_repeated_tool_guard(
        barebone_model,
        logger,
        state.tool_calls,
        tool_runtime,
        message_history,
        state.usage,
        state.executed_tool_calls,
        state.content_before_tools,
        client,
    )


def _sync_continue_tool_loop_after_batch(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    logger: Optional[Any],
    timeout: float,
    client: Optional[httpx.Client],
    max_tool_rounds: int,
    max_tool_calls: Optional[int],
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
    batch: ToolExecutionBatch,
    repeated_tool_names: list[str],
) -> bool:
    _sync_force_complete_if_max_tools(
        barebone_model,
        logger,
        message_history,
        state,
        tool_runtime,
        max_tool_calls,
    )
    LOG.debug("[TRACE] chat: Appending tool messages")
    _append_tool_loop_messages(messages, message_history, state, batch, repeated_tool_names)
    return _sync_advance_tool_round(
        barebone_model,
        messages,
        message_history,
        timeout,
        client,
        max_tool_rounds,
        state,
    )


def _sync_tool_loop_iteration(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    tool_executors: ToolExecutorMap,
    logger: Optional[Any],
    timeout: float,
    client: Optional[httpx.Client],
    max_tool_rounds: int,
    max_tool_calls: Optional[int],
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
) -> tuple[Optional[JsonDict], bool]:
    repeated_response, repeated_tool_names = _sync_repeated_tool_response_for_state(
        barebone_model, logger, state, tool_runtime, message_history, client
    )
    if repeated_response:
        return repeated_response, False

    batch = _sync_execute_tool_batch(
        barebone_model,
        logger,
        state,
        tool_runtime,
        tool_executors,
        timeout,
        remember_signatures=True,
    )

    post_batch_result = _tool_loop_post_batch_result(
        barebone_model,
        logger,
        messages,
        message_history,
        state,
        batch,
        "chat",
        hitl_stage_name=batch[4].get("stage_name") if batch[4] else None,
    )
    if post_batch_result is not None:
        return post_batch_result

    should_continue = _sync_continue_tool_loop_after_batch(
        barebone_model,
        messages,
        message_history,
        logger,
        timeout,
        client,
        max_tool_rounds,
        max_tool_calls,
        state,
        tool_runtime,
        batch,
        repeated_tool_names,
    )
    return None, should_continue


def _sync_tool_loop(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    tool_executors: ToolExecutorMap,
    logger: Optional[Any],
    timeout: float,
    client: Optional[httpx.Client],
    max_tool_rounds: int,
    max_tool_calls: Optional[int],
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
) -> Optional[JsonDict]:
    while state.tool_calls and tool_executors and state.rounds < max_tool_rounds:
        loop_response, should_continue = _sync_tool_loop_iteration(
            barebone_model,
            messages,
            message_history,
            tool_executors,
            logger,
            timeout,
            client,
            max_tool_rounds,
            max_tool_calls,
            state,
            tool_runtime,
        )
        if loop_response:
            return loop_response
        if not should_continue:
            break

    return None


def _flush_sync_remaining_tools(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    tool_executors: ToolExecutorMap,
    logger: Optional[Any],
    timeout: float,
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
) -> Optional[JsonDict]:
    if not (state.tool_calls and tool_executors):
        return None

    batch = _sync_execute_tool_batch(
        barebone_model,
        logger,
        state,
        tool_runtime,
        tool_executors,
        timeout,
        remember_signatures=False,
    )

    interrupt_response = _tool_loop_interrupt_response(
        barebone_model,
        logger,
        messages,
        message_history,
        state,
        batch,
    )
    if interrupt_response:
        return interrupt_response

    stage_ended = _tool_loop_control_completed(
        barebone_model,
        logger,
        message_history,
        state,
        batch,
    )
    if not stage_ended:
        _append_tool_loop_messages(
            messages,
            message_history,
            state,
            batch,
        )

    state.tool_calls = []
    return None
