"""Chat runtime orchestration facade."""

# pyright: strict
# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from typing import Any, Optional

import httpx

from IkaCore.agent_runtime_payloads import JsonDict

from ..base import BareBoneModel
from ..request_interface import api_request_retry, async_api_request_retry
from ..request_interface import get_provider as _get_provider
from ..summarization import (
    async_summarise_message_history,
    run_summarization,
    summarise_message_history,
)
from ..summarization import (
    create_summary_payload as _create_summary_payload,
)
from ..summarization import (
    get_summary_model as _get_summary_model,
)
from . import chat_request as _request
from . import chat_response as _response
from . import chat_tool_loop as _tool_loop
from . import chat_tool_loop_async as _tool_loop_async
from . import chat_tool_loop_sync as _tool_loop_sync
from .chat_request import (
    _ensure_first_input,
    _prepare_chat_inputs,
    _provider_for_model,
    _request_provider_round,
    _request_provider_round_async,
    _summarize_async_if_near_budget,
    _summarize_sync_if_near_budget,
)
from .chat_response import (
    _finalize_chat_response,
    _new_chat_state,
)
from .chat_tool_loop import (
    _async_tool_loop,
    _flush_async_remaining_tools,
    _flush_sync_remaining_tools,
    _sync_tool_loop,
)
from .tool_execution_sync import ToolExecutorMap
from .types import ChatLoopState, ToolRuntimeState

create_summary_payload = _create_summary_payload
get_provider = _get_provider
get_summary_model = _get_summary_model
get_total_tokens = _request.get_total_tokens
init_message_history = _request.init_message_history


def _sync_dependency_modules() -> None:
    _request.api_request_retry = api_request_retry
    _request.async_api_request_retry = async_api_request_retry
    _request.summarise_message_history = summarise_message_history
    _request.async_summarise_message_history = async_summarise_message_history
    _response.run_summarization = run_summarization
    _response.async_summarise_message_history = async_summarise_message_history
    _tool_loop_sync.run_summarization = run_summarization
    _tool_loop_async.async_summarise_message_history = async_summarise_message_history


def __getattr__(name: str) -> Any:
    for module in (_request, _response, _tool_loop):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _initial_sync_chat_state(
    provider: str,
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    timeout: float,
    client: Optional[httpx.Client],
) -> tuple[ChatLoopState, ToolRuntimeState]:
    content, reasoning_content, tool_calls, tokens, usage_info = _request_provider_round(
        provider,
        barebone_model,
        messages,
        message_history,
        timeout,
        client,
    )
    state = _new_chat_state(provider, content, reasoning_content, tool_calls, tokens, usage_info)
    tool_runtime = ToolRuntimeState.from_model(barebone_model)
    tool_runtime.sync_to_model(barebone_model)
    return state, tool_runtime


def _finalize_state_response(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    message_history: JsonDict,
    state: ChatLoopState,
) -> JsonDict:
    return _finalize_chat_response(
        barebone_model,
        logger,
        state.content,
        state.reasoning_content,
        state.tool_calls,
        state.executed_tool_calls,
        state.content_before_tools,
        message_history,
        state.usage,
        state.hijacked,
        state.tokens,
    )


def chat(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: Optional[JsonDict] = None,
    tool_executors: Optional[ToolExecutorMap] = None,
    logger: Optional[Any] = None,
    timeout: float = 900.0,
    client: Optional[httpx.Client] = None,
    max_tool_rounds: int = 5,
    max_tool_calls: Optional[int] = None,
    current_stage_index: Optional[int] = None,
    total_stages: int = 0,
) -> JsonDict:
    _sync_dependency_modules()
    message_history, tool_executors = _prepare_chat_inputs(
        barebone_model, messages, message_history, tool_executors, logger
    )
    _summarize_sync_if_near_budget(barebone_model, message_history, client)
    _ensure_first_input(message_history, messages)

    provider = _provider_for_model(barebone_model)
    state, tool_runtime = _initial_sync_chat_state(
        provider,
        barebone_model,
        messages,
        message_history,
        timeout,
        client,
    )
    loop_response = _sync_tool_loop(
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

    flush_response = _flush_sync_remaining_tools(
        barebone_model,
        messages,
        message_history,
        tool_executors,
        logger,
        timeout,
        state,
        tool_runtime,
    )
    if flush_response:
        return flush_response

    return _finalize_state_response(barebone_model, logger, message_history, state)


async def _initial_async_chat_state(
    provider: str,
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    timeout: float,
    client: httpx.AsyncClient,
) -> tuple[ChatLoopState, ToolRuntimeState]:
    content, reasoning_content, tool_calls, tokens, usage_info = await _request_provider_round_async(
        provider,
        barebone_model,
        messages,
        message_history,
        timeout,
        client,
    )
    state = _new_chat_state(provider, content, reasoning_content, tool_calls, tokens, usage_info)
    tool_runtime = ToolRuntimeState.from_model(barebone_model)
    tool_runtime.sync_to_model(barebone_model)
    return state, tool_runtime


async def _async_chat_with_client(
    provider: str,
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    tool_executors: ToolExecutorMap,
    logger: Optional[Any],
    timeout: float,
    client: httpx.AsyncClient,
    max_tool_rounds: int,
    max_tool_calls: Optional[int],
) -> JsonDict:
    state, tool_runtime = await _initial_async_chat_state(
        provider,
        barebone_model,
        messages,
        message_history,
        timeout,
        client,
    )
    loop_response = await _async_tool_loop(
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

    flush_response = await _flush_async_remaining_tools(
        barebone_model,
        messages,
        message_history,
        tool_executors,
        logger,
        timeout,
        client,
        state,
        tool_runtime,
    )
    if flush_response:
        return flush_response

    return _finalize_state_response(barebone_model, logger, message_history, state)


async def async_chat(
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: Optional[JsonDict] = None,
    tool_executors: Optional[ToolExecutorMap] = None,
    logger: Optional[Any] = None,
    timeout: float = 900.0,
    client: Optional[httpx.AsyncClient] = None,
    max_tool_rounds: int = 5,
    max_tool_calls: Optional[int] = None,
    current_stage_index: Optional[int] = None,
    total_stages: int = 0,
) -> JsonDict:
    _sync_dependency_modules()
    message_history, tool_executors = _prepare_chat_inputs(
        barebone_model, messages, message_history, tool_executors, logger
    )

    await _summarize_async_if_near_budget(barebone_model, message_history, client)
    _ensure_first_input(message_history, messages)
    provider = _provider_for_model(barebone_model)

    should_close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)

    try:
        return await _async_chat_with_client(
            provider,
            barebone_model,
            messages,
            message_history,
            tool_executors,
            logger,
            timeout,
            client,
            max_tool_rounds,
            max_tool_calls,
        )
    finally:
        if should_close_client:
            await client.aclose()


__all__ = [
    "api_request_retry",
    "async_api_request_retry",
    "async_chat",
    "async_summarise_message_history",
    "chat",
    "create_summary_payload",
    "get_provider",
    "get_summary_model",
    "get_total_tokens",
    "init_message_history",
    "run_summarization",
    "summarise_message_history",
]
