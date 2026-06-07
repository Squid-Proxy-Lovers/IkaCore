"""Response construction helpers for chat runtime orchestration."""

# pyright: strict
# pyright: reportUnusedFunction=false

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Optional, cast

import httpx

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_list, string_value

from ..base import AgentEndException, AgentTool, BareBoneModel
from ..chat_helpers_common import append_provider_tool_messages
from ..request_interface import IkaAPIError
from ..runtime_errors import IkaDebugDumpError
from ..summarization import async_summarise_message_history, run_summarization
from .response_interface import format_gemini_results
from .types import ChatLoopState, ChatResponsePayload, UsageInfo

LOG = logging.getLogger(__name__)
_SUMMARY_FALLBACK_EXCEPTIONS = (IkaAPIError, RuntimeError, ValueError, TypeError, KeyError, httpx.HTTPError)
UsageInput = UsageInfo | JsonDict | None
ToolCall = JsonDict
ToolCallList = list[ToolCall]
MessageList = list[JsonDict]
ToolMetadata = dict[str, JsonDict]


def _tool_function_payload(tool_call: JsonDict) -> JsonDict:
    return json_dict(tool_call.get("function"))


def _tool_name_from_call(tool_call: JsonDict) -> str:
    function_payload = _tool_function_payload(tool_call)
    return string_value(function_payload.get("name") or tool_call.get("name"))


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def _dump_api_round(
    barebone_model: BareBoneModel,
    payload: JsonDict,
    response_data: Optional[JsonDict],
    error: Optional[str] = None,
) -> None:
    out_dir = os.environ.get("IKA_DUMP_REQUESTS")
    if not out_dir:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        agent = getattr(barebone_model, "agent_name", "agent") or "agent"
        agent_safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(agent))[:60]
        # Hash the conversation portion so duplicate prompts collide on disk.
        # Different providers use different field names: chat completions uses
        # "messages", the OpenAI Responses API uses "input", Anthropic uses
        # "messages" with its own shape. Hash whichever is present.
        convo: object = payload.get("messages") or payload.get("input") or []
        msg_repr = json.dumps(convo, sort_keys=True, default=str)
        sig = hashlib.sha256(msg_repr.encode("utf-8", "replace")).hexdigest()[:12]
        ts = f"{time.time():.3f}"
        fname = f"{ts}__{agent_safe}__{sig}.json"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "agent_name": agent,
                    "model_id": getattr(barebone_model, "model_id", None),
                    "payload": payload,
                    "response": response_data,
                    "error": error,
                    "messages_sha256_12": sig,
                },
                f,
                indent=2,
                default=str,
            )
    except (OSError, TypeError, ValueError) as e:  # debug aid only — never fail the request
        LOG.debug("%s", IkaDebugDumpError(f"_dump_api_round failed: {e}"))


def _record_model_message(
    message_history: JsonDict,
    content: str,
    tokens: int,
    reasoning_content: Optional[str] = None,
) -> None:
    msg_id = str(uuid.uuid4())
    history_entry: JsonDict = {"message": content, "tokens": tokens}
    if reasoning_content:
        history_entry["reasoning_content"] = reasoning_content
    history_section(message_history, "messages")[msg_id] = history_entry


def _build_tool_metadata(barebone_model: BareBoneModel) -> ToolMetadata:
    tool_metadata: ToolMetadata = {}
    raw_agent_tools = getattr(barebone_model, "agent_tools", None)
    if not isinstance(raw_agent_tools, list):
        return tool_metadata

    for agent_tool in cast(list[AgentTool], raw_agent_tools):
        params: JsonDict = {}
        tool_args = agent_tool.args
        raw_properties: object = tool_args.properties
        if isinstance(raw_properties, dict):
            properties_input = cast(dict[str, object], raw_properties)
            properties = {
                key: cast(JsonDict, value)
                for key, value in properties_input.items()
                if key != "__required__" and isinstance(value, dict)
            }
            params = {"properties": properties, "required": string_list(properties_input.get("__required__"))}
        tool_metadata[agent_tool.name] = {
            "parallel": agent_tool.parallel,
            "limit_calls": agent_tool.limit_calls,
            "parameters": params,
        }
    return tool_metadata


def _clear_forced_tool_choice(barebone_model: BareBoneModel) -> None:
    if hasattr(barebone_model, "forced_tool_name"):
        barebone_model.forced_tool_name = None


def _build_synthetic_agent_end(summary_text: str) -> JsonDict:
    return {
        "type": "function",
        "function": {
            "name": "agent_end",
            "arguments": json.dumps({"input": summary_text}),
        },
    }


def _fallback_force_completion_text(
    reason: str,
    content_before_tools: str,
    content: str,
) -> str:
    for candidate in (content_before_tools, content):
        if candidate and candidate.strip() not in {"", "{}", "."}:
            return candidate.strip()
    return reason


def _usage_info(usage: UsageInput) -> UsageInfo:
    if isinstance(usage, UsageInfo):
        return usage
    return UsageInfo.from_mapping(usage)


def _compute_cost(logger: Optional[Any], barebone_model: BareBoneModel, usage: UsageInput) -> Optional[JsonDict]:
    if not logger:
        return None
    cost = logger.compute_cost(barebone_model.model_id, _usage_info(usage).to_dict())
    return cast(JsonDict, cost) if isinstance(cost, dict) else None


def _force_complete_with_summary(
    barebone_model: BareBoneModel,
    message_history: JsonDict,
    total_usage: UsageInput,
    logger: Optional[Any],
    all_executed_tool_call_list: ToolCallList,
    content_before_tools: str,
    content: str,
    reason: str,
    hijacked: bool = True,
) -> None:
    try:
        summary_text = run_summarization(
            barebone_model,
            message_history,
            prompt_kind="force_answer",
            write_to_history=False,
            use_same_model=True,
        )
    except _SUMMARY_FALLBACK_EXCEPTIONS as e:
        LOG.error(f"Failed to generate force completion summary: {e}")
        summary_text = ""

    final_text = (summary_text or "").strip()
    if not final_text:
        final_text = _fallback_force_completion_text(reason, content_before_tools, content)

    _clear_forced_tool_choice(barebone_model)
    _record_model_message(message_history, final_text, _estimate_tokens(final_text))
    synthetic_end = _build_synthetic_agent_end(final_text)
    executed_tool_calls = list(all_executed_tool_call_list) + [synthetic_end]
    cost_info = _compute_cost(logger, barebone_model, total_usage)
    if logger:
        logger.log_output(final_text, _usage_info(total_usage).to_dict(), cost_info, message_history)
    response_payload = _build_chat_response(
        final_text,
        None,
        [],
        executed_tool_calls,
        final_text,
        message_history,
        total_usage,
        cost_info,
        hijacked,
    )
    raise AgentEndException(response_payload)


async def _async_force_complete_with_summary(
    barebone_model: BareBoneModel,
    message_history: JsonDict,
    total_usage: UsageInput,
    logger: Optional[Any],
    all_executed_tool_call_list: ToolCallList,
    content_before_tools: str,
    content: str,
    reason: str,
    client: Optional[httpx.AsyncClient] = None,
    hijacked: bool = True,
) -> None:
    try:
        summary_text = await async_summarise_message_history(
            barebone_model,
            message_history,
            client=client,
            use_same_model=True,
            prompt_kind="force_answer",
            write_to_history=False,
        )
    except _SUMMARY_FALLBACK_EXCEPTIONS as e:
        LOG.error(f"Failed to generate async force completion summary: {e}")
        summary_text = ""

    final_text = (summary_text or "").strip()
    if not final_text:
        final_text = _fallback_force_completion_text(reason, content_before_tools, content)

    _clear_forced_tool_choice(barebone_model)
    _record_model_message(message_history, final_text, _estimate_tokens(final_text))
    synthetic_end = _build_synthetic_agent_end(final_text)
    executed_tool_calls = list(all_executed_tool_call_list) + [synthetic_end]
    cost_info = _compute_cost(logger, barebone_model, total_usage)
    if logger:
        logger.log_output(final_text, _usage_info(total_usage).to_dict(), cost_info, message_history)
    response_payload = _build_chat_response(
        final_text,
        None,
        [],
        executed_tool_calls,
        final_text,
        message_history,
        total_usage,
        cost_info,
        hijacked,
    )
    raise AgentEndException(response_payload)


def _build_chat_response(
    content: str,
    reasoning_content: Optional[str],
    tool_calls: ToolCallList,
    executed_tool_calls: ToolCallList,
    content_before_tools: str,
    message_history: JsonDict,
    usage_info: UsageInput,
    cost_info: Optional[JsonDict],
    hijacked: bool,
    interrupted: bool = False,
    interrupt_data: Optional[JsonDict] = None,
) -> JsonDict:
    return ChatResponsePayload(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        executed_tool_calls=executed_tool_calls,
        content_before_tools=content_before_tools,
        message_history=message_history,
        usage=_usage_info(usage_info),
        cost=cost_info,
        hijacked=hijacked,
        interrupted=interrupted,
        interrupt_data=interrupt_data,
    ).to_dict()


def _tool_signature_from_call(tool_call: JsonDict) -> tuple[str, str]:
    fn = _tool_function_payload(tool_call)
    tool_name = _tool_name_from_call(tool_call)
    args_raw: object = fn.get("arguments") or "{}"
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    except (TypeError, ValueError, json.JSONDecodeError):
        args = {}
    return tool_name, json.dumps(args, sort_keys=True)


def _find_repeated_tool_calls(tool_calls: ToolCallList, recent_tool_calls: list[tuple[str, str]]) -> list[tuple[str, int]]:
    repeated: list[tuple[str, int]] = []
    for tool_call in tool_calls:
        signature = _tool_signature_from_call(tool_call)
        recent_count = recent_tool_calls.count(signature)
        if recent_count >= 3:
            repeated.append((signature[0], recent_count))
    return repeated


def _remember_tool_calls(executed_tool_call_list: ToolCallList, recent_tool_calls: list[tuple[str, str]]) -> None:
    for tool_call in executed_tool_call_list:
        recent_tool_calls.append(_tool_signature_from_call(tool_call))
        if len(recent_tool_calls) > 10:
            recent_tool_calls.pop(0)


def _control_tool_flags(executed_tool_call_list: ToolCallList) -> tuple[bool, bool]:
    names = {_tool_name_from_call(tc) for tc in executed_tool_call_list}
    return "agent_end" in names, "stage_end" in names


def _repeated_tool_warning(repeated_tool_names: list[str]) -> str:
    if not repeated_tool_names:
        return ""
    seen = list(dict.fromkeys(repeated_tool_names))
    return (
        "System note: You have already called the following tool(s) multiple times with the same arguments: "
        + ", ".join(seen)
        + ". Do not repeat these calls. Proceed to the next step (e.g. use submit_discovery or other tools, then agent_end when done)."
    )


def _build_repeated_tool_hijack_response(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    tool_name: str,
    recent_count: int,
    force_answer: str,
    content_before_tools: str,
    message_history: JsonDict,
    total_usage: UsageInput,
    executed_tool_calls: ToolCallList,
) -> JsonDict:
    if not force_answer or not force_answer.strip():
        force_answer = (
            f"Agent stuck in loop after {recent_count} identical calls to '{tool_name}'. "
            f"Partial results:\n{content_before_tools}"
        )

    force_content = (
        f"CRITICAL: Agent stuck in loop. Tool '{tool_name}' called {recent_count} times with identical arguments. "
        "Auto-terminated and generated final response.\n\n"
        f"{force_answer}"
    )
    cost_info = _compute_cost(logger, barebone_model, total_usage)
    return _build_chat_response(
        force_content,
        None,
        [],
        executed_tool_calls,
        content_before_tools,
        message_history,
        total_usage,
        cost_info,
        True,
    )


def _build_interrupt_response(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    interrupt_content: str,
    reasoning_content: Optional[str],
    executed_tool_calls: ToolCallList,
    content_before_tools: str,
    message_history: JsonDict,
    total_usage: UsageInput,
    hijacked: bool,
    interrupt_data: JsonDict,
) -> JsonDict:
    cost_info = _compute_cost(logger, barebone_model, total_usage)
    if logger:
        logger.log_output(interrupt_content, _usage_info(total_usage).to_dict(), cost_info, message_history)
    return _build_chat_response(
        interrupt_content,
        reasoning_content,
        [],
        executed_tool_calls,
        content_before_tools,
        message_history,
        total_usage,
        cost_info,
        hijacked,
        interrupted=True,
        interrupt_data=interrupt_data,
    )


def _raise_agent_end_response(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    content: str,
    reasoning_content: Optional[str],
    executed_tool_calls: ToolCallList,
    content_before_tools: str,
    message_history: JsonDict,
    total_usage: UsageInput,
    hijacked: bool,
    tokens: int,
) -> None:
    _clear_forced_tool_choice(barebone_model)
    cost_info = _compute_cost(logger, barebone_model, total_usage)
    if logger:
        logger.log_output(content, _usage_info(total_usage).to_dict(), cost_info, message_history)
    _record_model_message(message_history, content, tokens, reasoning_content)
    response_payload = _build_chat_response(
        content,
        reasoning_content,
        [],
        executed_tool_calls,
        content_before_tools,
        message_history,
        total_usage,
        cost_info,
        hijacked,
    )
    raise AgentEndException(response_payload)


def _append_tool_messages(
    provider: str,
    messages: MessageList,
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: ToolCallList,
    tool_messages: MessageList,
    tool_results: list[str],
    tokens: int,
    warning_message: str = "",
) -> None:
    append_provider_tool_messages(
        provider,
        messages,
        message_history,
        content,
        reasoning_content,
        executed_tool_call_list,
        tool_messages,
        tool_results,
        tokens,
        warning_message,
        format_gemini_results,
    )


def _interrupt_response_if_needed(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    interrupt_data: Optional[JsonDict],
    provider: str,
    messages: MessageList,
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: ToolCallList,
    all_executed_tool_call_list: ToolCallList,
    tool_messages: MessageList,
    tool_results: list[str],
    tokens: int,
    content_before_tools: str,
    total_usage: UsageInput,
    hijacked: bool,
    hitl_stage_name: Optional[str] = None,
) -> Optional[JsonDict]:
    if not interrupt_data:
        return None

    interrupt_content = string_value(interrupt_data.get("question") or content or "Awaiting human input")
    if hitl_stage_name and logger:
        logger.log_hitl_prompt(hitl_stage_name)
    _append_tool_messages(
        provider,
        messages,
        message_history,
        content,
        reasoning_content,
        executed_tool_call_list,
        tool_messages,
        tool_results,
        tokens,
    )
    return _build_interrupt_response(
        barebone_model,
        logger,
        interrupt_content,
        reasoning_content,
        all_executed_tool_call_list,
        content_before_tools,
        message_history,
        total_usage,
        hijacked,
        interrupt_data,
    )


def _handle_control_tool_completion(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: ToolCallList,
    all_executed_tool_call_list: ToolCallList,
    content_before_tools: str,
    message_history: JsonDict,
    total_usage: UsageInput,
    hijacked: bool,
    tokens: int,
) -> bool:
    agent_end_called, stage_end_called = _control_tool_flags(executed_tool_call_list)
    if agent_end_called:
        _raise_agent_end_response(
            barebone_model,
            logger,
            content,
            reasoning_content,
            all_executed_tool_call_list,
            content_before_tools,
            message_history,
            total_usage,
            hijacked,
            tokens,
        )

    if stage_end_called:
        _clear_forced_tool_choice(barebone_model)
        return True

    return False


def _finalize_chat_response(
    barebone_model: BareBoneModel,
    logger: Optional[Any],
    content: str,
    reasoning_content: Optional[str],
    tool_calls: ToolCallList,
    all_executed_tool_call_list: ToolCallList,
    content_before_tools: str,
    message_history: JsonDict,
    total_usage: UsageInput,
    hijacked: bool,
    tokens: int,
) -> JsonDict:
    cost_info = _compute_cost(logger, barebone_model, total_usage)
    usage = _usage_info(total_usage)
    if logger:
        logger.log_output(content, usage.to_dict(), cost_info, message_history)
    _record_model_message(message_history, content, tokens, reasoning_content)
    return _build_chat_response(
        content,
        reasoning_content,
        tool_calls,
        all_executed_tool_call_list,
        content_before_tools,
        message_history,
        usage,
        cost_info,
        hijacked,
    )


def _new_chat_state(
    provider: str,
    content: str,
    reasoning_content: Optional[str],
    tool_calls: ToolCallList,
    tokens: int,
    usage_info: Optional[JsonDict],
) -> ChatLoopState:
    return ChatLoopState(
        provider=provider,
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        tokens=tokens,
        content_before_tools=content,
        usage=UsageInfo.from_mapping(usage_info),
    )
