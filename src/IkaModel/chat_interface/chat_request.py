"""Provider request and context-window helpers for chat runtime orchestration."""

# pyright: strict
# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional, cast

import httpx

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value
from IkaCore.cli_output import OutputType, get_cli_output
from IkaCore.tools import ToolExecutor

from ..base import BareBoneModel
from ..chat_helpers_common import build_provider_request, parse_provider_response
from ..request_interface import (
    IkaAPIError,
    _is_context_length_error,
    api_request_retry,
    async_api_request_retry,
    get_max_tokens,
    get_provider,
)
from ..runtime_errors import IkaContextWindowError, IkaProviderPayloadError, IkaProviderResponseError
from ..summarization import async_summarise_message_history, summarise_message_history
from .chat_response import _SUMMARY_FALLBACK_EXCEPTIONS, _dump_api_round
from .response_interface import extract_usage

LOG = logging.getLogger(__name__)

ToolExecutorMap = dict[str, ToolExecutor]
ProviderRound = tuple[str, Optional[str], list[JsonDict], int, Optional[JsonDict]]
ProviderPayload = tuple[str, dict[str, str], JsonDict]
ProviderPayloadBuilder = Callable[[], ProviderPayload]


def init_message_history() -> JsonDict:
    return {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _history_entry_tokens(entry: JsonDict) -> int:
    tokens = _token_count(entry.get("tokens"))
    return tokens if tokens > 0 else _estimate_tokens(string_value(entry.get("message")))


def get_total_tokens(message_history: JsonDict) -> int:
    total = 0
    for key in ("system", "first_input", "summary"):
        total += _history_entry_tokens(history_section(message_history, key))
    for msg in history_section(message_history, "messages").values():
        total += _history_entry_tokens(json_dict(msg))
    return total


def _prepare_chat_inputs(
    barebone_model: BareBoneModel,
    messages: object,
    message_history: Optional[JsonDict],
    tool_executors: object,
    logger: Optional[Any],
) -> tuple[JsonDict, ToolExecutorMap]:
    if not barebone_model:
        raise ValueError("barebone_model is required")
    if not messages:
        raise ValueError("messages is required and cannot be empty")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    resolved_messages = cast(list[JsonDict], messages)
    if not hasattr(barebone_model, 'model_id') or not barebone_model.model_id:
        raise ValueError("barebone_model.model_id is required")
    if not hasattr(barebone_model, 'api_key') or not barebone_model.api_key:
        raise ValueError("barebone_model.api_key is required")
    if not hasattr(barebone_model, 'api_url') or not barebone_model.api_url:
        raise ValueError("barebone_model.api_url is required")
    if tool_executors is not None and not isinstance(tool_executors, dict):
        raise ValueError("tool_executors must be a dictionary if provided")

    resolved_history = message_history or init_message_history()
    resolved_tools = cast(ToolExecutorMap, tool_executors) if tool_executors else {}
    if logger:
        logger.log_input(resolved_messages)
    return resolved_history, resolved_tools


def _ensure_first_input(message_history: JsonDict, messages: list[JsonDict]) -> None:
    first_input = history_section(message_history, "first_input")
    if not first_input.get("message") and messages:
        first_input["message"] = string_value(messages[0].get("content"), str(messages[0]))
        first_input["tokens"] = 0


def _provider_for_model(barebone_model: BareBoneModel) -> str:
    use_responses_api = getattr(barebone_model, "use_responses_api", True)
    return get_provider(barebone_model.model_id, barebone_model.api_url, use_responses_api)


def _summarize_sync_if_near_budget(
    barebone_model: BareBoneModel,
    message_history: JsonDict,
    client: Optional[httpx.Client],
) -> None:
    token_count = get_total_tokens(message_history)
    max_tokens = getattr(barebone_model, 'context_budget', None) or get_max_tokens(barebone_model.model_id)
    if token_count > max_tokens * 0.8:
        summarise_message_history(barebone_model, message_history, client=client)


async def _summarize_async_if_near_budget(
    barebone_model: BareBoneModel,
    message_history: JsonDict,
    client: Optional[httpx.AsyncClient],
) -> None:
    token_count = get_total_tokens(message_history)
    max_tokens = getattr(barebone_model, 'context_budget', None) or get_max_tokens(barebone_model.model_id)
    if token_count > max_tokens * 0.8:
        LOG.info(f"Token count ({token_count}) approaching limit ({max_tokens}). Summarizing history...")
        await async_summarise_message_history(barebone_model, message_history, client)


def _parse_provider_round(
    provider: str,
    data: JsonDict,
    model_id: str,
    usage_token_fallback: Optional[int] = None,
) -> ProviderRound:
    try:
        content, reasoning_content, tool_calls, tokens = parse_provider_response(provider, data, model_id)
        usage_info = extract_usage(provider, data)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise IkaProviderResponseError(
            f"Failed to parse {provider} response for model {model_id}: {e}"
        ) from e
    if usage_info:
        fallback = tokens if usage_token_fallback is None else usage_token_fallback
        tokens = _token_count(usage_info.get("total_tokens", fallback))
    return content, reasoning_content, tool_calls, tokens, usage_info


def _response_json_or_raise(response: httpx.Response, provider: str) -> JsonDict:
    try:
        data: object = response.json()
    except (TypeError, ValueError) as e:
        raise IkaProviderResponseError(f"{provider} response was not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise IkaProviderResponseError(f"{provider} response JSON must be an object, got {type(data).__name__}")
    return cast(JsonDict, data)


def _dump_response_round(
    barebone_model: BareBoneModel,
    payload: JsonDict,
    response: httpx.Response,
    error: str,
) -> None:
    try:
        _dump_api_round(barebone_model, payload, json_dict(response.json()))
    except ValueError:
        _dump_api_round(barebone_model, payload, None, error=error)


def _request_provider_round(
    provider: str,
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    timeout: float,
    client: Optional[httpx.Client],
    usage_token_fallback: Optional[int] = None,
) -> ProviderRound:
    def _build() -> ProviderPayload:
        return build_provider_request(provider, barebone_model, messages, message_history)

    response = _api_request_with_context_fallback(_build, barebone_model, message_history, timeout, client)
    response.raise_for_status()
    data = _response_json_or_raise(response, provider)
    return _parse_provider_round(provider, data, barebone_model.model_id, usage_token_fallback)


async def _request_provider_round_async(
    provider: str,
    barebone_model: BareBoneModel,
    messages: list[JsonDict],
    message_history: JsonDict,
    timeout: float,
    client: httpx.AsyncClient,
    usage_token_fallback: Optional[int] = None,
) -> ProviderRound:
    def _build() -> ProviderPayload:
        return build_provider_request(provider, barebone_model, messages, message_history)

    response = await _api_request_with_context_fallback_async(_build, barebone_model, message_history, timeout, client)
    response.raise_for_status()
    data = _response_json_or_raise(response, provider)
    return _parse_provider_round(provider, data, barebone_model.model_id, usage_token_fallback)


def _api_request_with_context_fallback(
    build_payload_fn: ProviderPayloadBuilder,
    barebone_model: BareBoneModel,
    message_history: JsonDict,
    timeout: float = 900.0,
    client: Optional[httpx.Client] = None,
) -> httpx.Response:
    try:
        api_url, headers, payload = build_payload_fn()
    except (TypeError, ValueError, KeyError) as e:
        raise IkaProviderPayloadError(f"Failed to build provider request payload: {e}") from e

    try:
        resp = api_request_retry(api_url, headers, payload, timeout=timeout, client=client)
        _dump_response_round(barebone_model, payload, resp, error="response not json")
        return resp
    except (IkaAPIError, httpx.HTTPError) as e:
        if not _is_context_length_error(e):
            _dump_api_round(barebone_model, payload, None, error=repr(e))
            raise
        LOG.warning("Context length exceeded. Forcing summarization and retrying.")
        get_cli_output().emit(OutputType.AGENT_RESPONSE, "Context limit exceeded. Summarized history and retrying.", ["API"], step=0)
        try:
            summarise_message_history(barebone_model, message_history, client=client)
        except _SUMMARY_FALLBACK_EXCEPTIONS as summary_error:
            raise IkaContextWindowError(
                f"Failed to summarize context before retrying provider request: {summary_error}"
            ) from summary_error
        try:
            api_url, headers, payload = build_payload_fn()
        except (TypeError, ValueError, KeyError) as payload_error:
            raise IkaContextWindowError(
                f"Failed to rebuild provider request after context summarization: {payload_error}"
            ) from payload_error
        resp = api_request_retry(api_url, headers, payload, timeout=timeout, client=client)
        _dump_response_round(barebone_model, payload, resp, error="response not json (post-summarization)")
        return resp
    except (TypeError, ValueError) as e:
        raise IkaContextWindowError(f"Failed to build or retry provider request: {e}") from e


async def _api_request_with_context_fallback_async(
    build_payload_fn: ProviderPayloadBuilder,
    barebone_model: BareBoneModel,
    message_history: JsonDict,
    timeout: float = 900.0,
    client: Optional[httpx.AsyncClient] = None
) -> httpx.Response:
    try:
        api_url, headers, payload = build_payload_fn()
    except (TypeError, ValueError, KeyError) as e:
        raise IkaProviderPayloadError(f"Failed to build provider request payload: {e}") from e

    try:
        resp = await async_api_request_retry(api_url, headers, payload, timeout=timeout, client=client)
        _dump_response_round(barebone_model, payload, resp, error="response not json")
        return resp
    except (IkaAPIError, httpx.HTTPError) as e:
        if not _is_context_length_error(e):
            _dump_api_round(barebone_model, payload, None, error=repr(e))
            raise
        LOG.warning("Context length exceeded. Forcing summarization and retrying.")
        get_cli_output().emit(OutputType.AGENT_RESPONSE, "Context limit exceeded. Summarized history and retrying.", ["API"], step=0)
        try:
            await async_summarise_message_history(barebone_model, message_history, client=client)
        except _SUMMARY_FALLBACK_EXCEPTIONS as summary_error:
            raise IkaContextWindowError(
                f"Failed to summarize context before retrying provider request: {summary_error}"
            ) from summary_error
        try:
            api_url, headers, payload = build_payload_fn()
        except (TypeError, ValueError, KeyError) as payload_error:
            raise IkaContextWindowError(
                f"Failed to rebuild provider request after context summarization: {payload_error}"
            ) from payload_error
        resp = await async_api_request_retry(api_url, headers, payload, timeout=timeout, client=client)
        _dump_response_round(barebone_model, payload, resp, error="response not json (post-summarization)")
        return resp
    except (TypeError, ValueError) as e:
        raise IkaContextWindowError(f"Failed to build or retry provider request: {e}") from e
