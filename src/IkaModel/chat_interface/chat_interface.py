"""Public chat interface compatibility facade."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import httpx

from ..request_interface import api_request_retry, async_api_request_retry, get_provider
from ..summarization import (
    async_summarise_message_history,
    create_summary_payload,
    get_summary_model,
    run_summarization,
    summarise_message_history,
)
from . import chat_runtime as _runtime
from .response_interface import (
    async_execute_tool,
    async_execute_tool_calls,
    execute_tool_calls,
    extract_usage,
    format_gemini_results,
)

init_message_history = _runtime.init_message_history
get_total_tokens = _runtime.get_total_tokens


def _sync_runtime_dependencies() -> None:
    _runtime.api_request_retry = api_request_retry
    _runtime.async_api_request_retry = async_api_request_retry
    _runtime.run_summarization = run_summarization
    _runtime.summarise_message_history = summarise_message_history
    _runtime.async_summarise_message_history = async_summarise_message_history


def chat(
    barebone_model: Any,
    messages: list[dict],
    message_history: Optional[dict] = None,
    tool_executors: Optional[Dict[str, Callable]] = None,
    logger: Optional[Any] = None,
    timeout: float = 900.0,
    client: Optional[httpx.Client] = None,
    max_tool_rounds: int = 5,
    max_tool_calls: Optional[int] = None,
    current_stage_index: Optional[int] = None,
    total_stages: int = 0,
) -> Dict[str, Any]:
    _sync_runtime_dependencies()
    return _runtime.chat(
        barebone_model,
        messages,
        message_history=message_history,
        tool_executors=tool_executors,
        logger=logger,
        timeout=timeout,
        client=client,
        max_tool_rounds=max_tool_rounds,
        max_tool_calls=max_tool_calls,
        current_stage_index=current_stage_index,
        total_stages=total_stages,
    )


async def async_chat(
    barebone_model: Any,
    messages: list[dict],
    message_history: Optional[dict] = None,
    tool_executors: Optional[Dict[str, Callable]] = None,
    logger: Optional[Any] = None,
    timeout: float = 900.0,
    client: Optional[httpx.AsyncClient] = None,
    max_tool_rounds: int = 5,
    max_tool_calls: Optional[int] = None,
    current_stage_index: Optional[int] = None,
    total_stages: int = 0,
) -> Dict[str, Any]:
    _sync_runtime_dependencies()
    return await _runtime.async_chat(
        barebone_model,
        messages,
        message_history=message_history,
        tool_executors=tool_executors,
        logger=logger,
        timeout=timeout,
        client=client,
        max_tool_rounds=max_tool_rounds,
        max_tool_calls=max_tool_calls,
        current_stage_index=current_stage_index,
        total_stages=total_stages,
    )


__all__ = [
    "async_api_request_retry",
    "async_chat",
    "async_execute_tool",
    "async_execute_tool_calls",
    "async_summarise_message_history",
    "chat",
    "create_summary_payload",
    "execute_tool_calls",
    "extract_usage",
    "format_gemini_results",
    "get_provider",
    "get_summary_model",
    "get_total_tokens",
    "init_message_history",
    "run_summarization",
    "summarise_message_history",
]
