"""Asynchronous chat tool-loop coordination."""

# pyright: strict
# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from IkaCore.agent_runtime_payloads import JsonDict

from ..base import BareBoneModel
from ..summarization import async_summarise_message_history
from .chat_request import _request_provider_round_async
from .chat_response import (
    _SUMMARY_FALLBACK_EXCEPTIONS,
    UsageInput,
    _async_force_complete_with_summary,
    _build_repeated_tool_hijack_response,
    _build_tool_metadata,
    _find_repeated_tool_calls,
)
from .chat_tool_common import (
    ToolExecutionBatch,
    _append_tool_loop_messages,
    _apply_tool_execution_result,
    _tool_loop_control_completed,
    _tool_loop_interrupt_response,
    _tool_loop_post_batch_result,
)
from .response_interface import async_execute_tool_calls
from .tool_execution_sync import ToolExecutorMap
from .types import ChatLoopState, ToolRuntimeState

LOG = logging.getLogger(__name__)


async def _async_repeated_tool_guard(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    tool_calls: list[JsonDict],
    tool_runtime: ToolRuntimeState,
    message_history: JsonDict,
    total_usage: UsageInput,
    all_executed_tool_call_list: list[JsonDict],
    content_before_tools: str,
    client: httpx.AsyncClient,
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
            try:
                force_answer = await async_summarise_message_history(
                    barebone_model,
                    message_history,
                    client=client,
                    use_same_model=True,
                    prompt_kind="force_answer",
                    write_to_history=False,
                )
            except _SUMMARY_FALLBACK_EXCEPTIONS as e:
                LOG.error(f"Failed to generate force_answer summary: {e}")
                force_answer = ""
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


async def _async_execute_tool_batch(
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
    batch = await async_execute_tool_calls(
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


async def _async_force_complete_if_max_tools(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    message_history: JsonDict,
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
    max_tool_calls: Optional[int],
    client: httpx.AsyncClient,
) -> None:
    if not (max_tool_calls and tool_runtime.current_step >= max_tool_calls):
        return
    LOG.debug("[TRACE] async_chat: Max tool calls reached")
    LOG.warning(f"Max tool calls ({max_tool_calls}) reached. Force-completing agent locally.")
    await _async_force_complete_with_summary(
        barebone_model=barebone_model,
        message_history=message_history,
        total_usage=state.usage,
        logger=logger,
        all_executed_tool_call_list=state.executed_tool_calls,
        content_before_tools=state.content_before_tools,
        content=state.content,
        reason=f"Maximum tool call limit ({max_tool_calls}) reached at step {tool_runtime.current_step}.",
        client=client,
        hijacked=True,
    )


async def _async_advance_tool_round(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    timeout: float,
    client: httpx.AsyncClient,
    max_tool_rounds: int,
    state: ChatLoopState,
) -> bool:
    state.rounds += 1
    if state.rounds >= max_tool_rounds:
        LOG.debug("[TRACE] async_chat: Max tool rounds reached")
        state.tool_calls = []
        return False

    LOG.debug("[TRACE] async_chat: Sending follow-up API request")
    state.content, state.reasoning_content, state.tool_calls, state.tokens, usage_info = await _request_provider_round_async(
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


async def _async_repeated_tool_response_for_state(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
    message_history: JsonDict,
    client: httpx.AsyncClient,
) -> tuple[Optional[JsonDict], list[str]]:
    return await _async_repeated_tool_guard(
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


async def _async_continue_tool_loop_after_batch(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    logger: Optional[Any],
    timeout: float,
    client: httpx.AsyncClient,
    max_tool_rounds: int,
    max_tool_calls: Optional[int],
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
    batch: ToolExecutionBatch,
    repeated_tool_names: list[str],
) -> bool:
    await _async_force_complete_if_max_tools(
        barebone_model,
        logger,
        message_history,
        state,
        tool_runtime,
        max_tool_calls,
        client,
    )
    LOG.debug("[TRACE] async_chat: Appending tool messages")
    _append_tool_loop_messages(messages, message_history, state, batch, repeated_tool_names)
    return await _async_advance_tool_round(
        barebone_model,
        messages,
        message_history,
        timeout,
        client,
        max_tool_rounds,
        state,
    )


async def _async_tool_loop_iteration(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    tool_executors: ToolExecutorMap,
    logger: Optional[Any],
    timeout: float,
    client: httpx.AsyncClient,
    max_tool_rounds: int,
    max_tool_calls: Optional[int],
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
) -> tuple[Optional[JsonDict], bool]:
    repeated_response, repeated_tool_names = await _async_repeated_tool_response_for_state(
        barebone_model, logger, state, tool_runtime, message_history, client
    )
    if repeated_response:
        return repeated_response, False

    batch = await _async_execute_tool_batch(
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
        "async_chat",
    )
    if post_batch_result is not None:
        return post_batch_result

    should_continue = await _async_continue_tool_loop_after_batch(
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


async def _async_tool_loop(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    tool_executors: ToolExecutorMap,
    logger: Optional[Any],
    timeout: float,
    client: httpx.AsyncClient,
    max_tool_rounds: int,
    max_tool_calls: Optional[int],
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
) -> Optional[JsonDict]:
    while state.tool_calls and tool_executors and state.rounds < max_tool_rounds:
        loop_response, should_continue = await _async_tool_loop_iteration(
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


async def _flush_async_remaining_tools(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    tool_executors: ToolExecutorMap,
    logger: Optional[Any],
    timeout: float,
    client: httpx.AsyncClient,
    state: ChatLoopState,
    tool_runtime: ToolRuntimeState,
) -> Optional[JsonDict]:
    if not (state.tool_calls and tool_executors):
        return None

    batch = await _async_execute_tool_batch(
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
