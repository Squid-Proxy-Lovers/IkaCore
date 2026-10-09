import copy
import json
import logging
import re
import uuid
from typing import Any, Dict, Optional, List, Callable

import httpx

from ..base import BareBoneModel, AgentEndException
from IkaCore.cli_output import get_cli_output, OutputType
from ..request_interface import (
    get_provider,
    get_max_tokens,
    api_request_retry,
    async_api_request_retry,
    _is_context_length_error,
)
from ..summarization import (
    summarise_message_history,
    async_summarise_message_history,
    run_summarization,  # noqa: F401 re-export for callers
    get_summary_model,  # noqa: F401 re-export for callers
    create_summary_payload,  # noqa: F401 re-export for callers
    get_context_usage,  # noqa: F401 re-export for callers
)
from .response_interface import (
    extract_usage,
    execute_tool_calls,
    async_execute_tool_calls,
    async_execute_tool,  # noqa: F401 re-export for callers
    format_gemini_results,
)
from ..chat_helpers_common import (
    build_provider_request,
    parse_provider_response,
    append_provider_tool_messages,
)

LOG = logging.getLogger(__name__)
_TERMINAL_CONTROL_TOOL_NAMES = {"agent_end", "end_execution"}
_OVERFLOW_FAST_PATH_RATIO = 2.0
_COMPACTION_TAIL_MAX_MESSAGES = 4
_CONTEXT_PACK_MESSAGE_PREFIXES = ("[Context Pack:", "[Context Pack Loaded:")
_CONTEXT_ERROR_TOKEN_RE = re.compile(
    r"maximum context length is\s*([\d,]+)\s*tokens.*?"
    r"requested\s*([\d,]+)\s*tokens\s*"
    r"\(([\d,]+)\s*in the messages,\s*([\d,]+)\s*in the completion\)",
    re.IGNORECASE | re.DOTALL,
)


def _get_registered_tool_names(barebone_model: BareBoneModel) -> set[str]:
    agent_tools = getattr(barebone_model, "agent_tools", None) or []
    return {
        getattr(tool, "name", "")
        for tool in agent_tools
        if getattr(tool, "name", "")
    }


def _get_terminal_control_tool_names(barebone_model: BareBoneModel) -> set[str]:
    registered = _get_registered_tool_names(barebone_model)
    names = registered & _TERMINAL_CONTROL_TOOL_NAMES
    return names or {"agent_end"}


def _get_primary_final_tool_name(barebone_model: BareBoneModel) -> str:
    terminal_names = _get_terminal_control_tool_names(barebone_model)
    if "end_execution" in terminal_names and "agent_end" not in terminal_names:
        return "end_execution"
    if "agent_end" in terminal_names:
        return "agent_end"
    return "end_execution" if "end_execution" in terminal_names else "agent_end"


def _record_model_message(message_history: dict, content: str, tokens: int, reasoning_content: Optional[str] = None) -> None:
    msg_id = str(uuid.uuid4())
    history_entry = {"message": content, "tokens": tokens}
    if reasoning_content:
        history_entry["reasoning_content"] = reasoning_content
    message_history["messages"][msg_id] = history_entry


def _build_chat_response(
    content: str,
    reasoning_content: Optional[str],
    tool_calls: List[dict],
    executed_tool_calls: List[dict],
    content_before_tools: str,
    message_history: dict,
    usage_info: Optional[dict],
    cost_info: Optional[dict],
    hijacked: bool,
) -> Dict[str, Any]:
    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
        "executed_tool_calls": executed_tool_calls,
        "content_before_tools": content_before_tools,
        "message_history": message_history,
        "usage": usage_info,
        "cost": cost_info,
        "hijacked": hijacked,
        "compaction_count": message_history.get("compaction_count", 0),
    }


def init_message_history() -> dict:
    return {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
        "compaction_count": 0,
        "_context_warning_issued": None,  # tracks last warning level emitted: "warning" or "critical"
    }


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content)
    return str(message)


def _is_context_pack_message(message: Any) -> bool:
    return _message_content(message).startswith(_CONTEXT_PACK_MESSAGE_PREFIXES)


def _sanitize_recent_tail(messages: List[dict]) -> List[dict]:
    sanitized: List[dict] = []
    tool_context_open = False

    for message in messages:
        role = message.get("role")
        if role == "tool":
            if tool_context_open:
                sanitized.append(message)
            continue

        sanitized.append(message)
        tool_context_open = bool(message.get("tool_calls"))

    return sanitized


def _rebuild_messages_after_compaction(message_history: dict, messages: Optional[list]) -> None:
    if messages is None:
        return

    snapshot = [copy.deepcopy(message) for message in messages]
    first_input = (message_history.get("first_input", {}) or {}).get("message", "")

    preserved_context_messages: List[dict] = []
    seen_context_contents: set[str] = set()
    for message in snapshot:
        content = _message_content(message)
        if not content or not _is_context_pack_message(message):
            continue
        if content in seen_context_contents:
            continue
        if isinstance(message, dict):
            preserved_context_messages.append(message)
            seen_context_contents.add(content)

    preserved_tail: List[dict] = []
    seen_tail_contents: set[str] = set()
    for message in reversed(snapshot):
        if not isinstance(message, dict):
            continue
        content = _message_content(message)
        if not content or content == first_input or _is_context_pack_message(message):
            continue
        if content in seen_tail_contents:
            continue
        preserved_tail.append(message)
        seen_tail_contents.add(content)
        if len(preserved_tail) >= _COMPACTION_TAIL_MAX_MESSAGES:
            break
    preserved_tail.reverse()
    preserved_tail = _sanitize_recent_tail(preserved_tail)

    rebuilt_messages: List[dict] = []
    if first_input:
        rebuilt_messages.append({"role": "user", "content": first_input})
    rebuilt_messages.extend(preserved_context_messages)
    rebuilt_messages.extend(preserved_tail)

    messages.clear()
    messages.extend(rebuilt_messages)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def _estimate_payload_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return _estimate_tokens(value)
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return _estimate_tokens(str(value))
    if isinstance(value, list):
        return sum(_estimate_payload_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(_estimate_payload_tokens(item) for item in value.values())
    return _estimate_tokens(str(value))


def _extract_requested_completion_tokens(payload: dict) -> int:
    for key in ("max_completion_tokens", "max_tokens"):
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            return value
    generation_config = payload.get("generationConfig")
    if isinstance(generation_config, dict):
        value = generation_config.get("maxOutputTokens")
        if isinstance(value, int) and value > 0:
            return value
    return 0


def _parse_context_error_token_stats(error: Exception) -> dict:
    match = _CONTEXT_ERROR_TOKEN_RE.search(str(error))
    if not match:
        return {}

    max_tokens, requested_tokens, message_tokens, completion_tokens = (
        int(group.replace(",", ""))
        for group in match.groups()
    )
    return {
        "max_tokens": max_tokens,
        "requested_tokens": requested_tokens,
        "message_tokens": message_tokens,
        "completion_tokens": completion_tokens,
        "source": "provider",
    }


def _get_request_token_stats(
    barebone_model: BareBoneModel,
    payload: dict,
    error: Optional[Exception] = None,
) -> dict:
    parsed = _parse_context_error_token_stats(error) if error else {}
    max_tokens = (
        parsed.get("max_tokens")
        or getattr(barebone_model, "context_budget", None)
        or get_max_tokens(barebone_model.model_id)
    )
    completion_tokens = parsed.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = _extract_requested_completion_tokens(payload)
    message_tokens = parsed.get("message_tokens")
    if message_tokens is None:
        message_tokens = _estimate_payload_tokens(payload)
    requested_tokens = parsed.get("requested_tokens")
    if requested_tokens is None:
        requested_tokens = message_tokens + completion_tokens
    usage_ratio = (requested_tokens / max_tokens) if max_tokens else 0.0
    return {
        "max_tokens": max_tokens,
        "requested_tokens": requested_tokens,
        "message_tokens": message_tokens,
        "completion_tokens": completion_tokens,
        "usage_ratio": usage_ratio,
        "source": parsed.get("source", "estimated"),
    }


def _build_overflow_recovery_prompt(stats: dict, *, hard_limit: bool) -> str:
    ratio_pct = int(stats["usage_ratio"] * 100) if stats["max_tokens"] else 0
    strict_line = (
        "You must not request any broad raw listings, recursive dumps, or unbounded search results. "
        "Return exactly one narrow tool call with explicit bounds, or a short plan if a tool call is not yet safe."
        if hard_limit else
        "Remake the step with much tighter control. Use one small bounded tool call, page results, and prefer summaries over raw output."
    )
    return (
        "CRITICAL CONTEXT OVERFLOW RECOVERY.\n"
        f"Your previous request would have used approximately {stats['requested_tokens']} tokens total "
        f"({stats['message_tokens']} in messages, {stats['completion_tokens']} requested for completion) "
        f"against a maximum context of {stats['max_tokens']} tokens ({ratio_pct}% of limit).\n"
        "The oversized in-flight input has been deleted.\n"
        f"{strict_line}\n"
        "Required constraints:\n"
        "- Choose a much narrower scope.\n"
        "- Add explicit limits such as depth, count, page size, or file count.\n"
        "- Avoid returning large raw outputs when a summary or targeted slice is enough.\n"
        "- If you need more data, gather it over multiple small tool calls instead of one huge call."
    )


def _apply_overflow_remake_recovery(
    barebone_model: BareBoneModel,
    message_history: dict,
    messages: Optional[list],
    stats: dict,
    *,
    hard_limit: bool,
    fast_path: bool = False,
) -> None:
    if messages is None:
        return

    prompt = _build_overflow_recovery_prompt(stats, hard_limit=hard_limit)
    summarise_message_history(barebone_model, message_history)
    messages.clear()
    messages.append({"role": "user", "content": prompt})

    hierarchy = getattr(barebone_model, "agent_hierarchy", None) or []
    mode = "fast-path" if fast_path else "retry"
    strictness = "strict" if hard_limit else "controlled"
    get_cli_output().emit(
        OutputType.AGENT_RESPONSE,
        (
            f"Context overflow recovery ({mode}, {strictness}): dropped oversized in-flight input "
            f"and requested a smaller remake ({stats['requested_tokens']}/{stats['max_tokens']} tokens, "
            f"{stats['usage_ratio'] * 100:.0f}% of limit)."
        ),
        hierarchy,
        step=0,
    )


async def _apply_overflow_remake_recovery_async(
    barebone_model: BareBoneModel,
    message_history: dict,
    messages: Optional[list],
    stats: dict,
    *,
    hard_limit: bool,
    fast_path: bool = False,
    client: Optional[httpx.AsyncClient] = None,
) -> None:
    if messages is None:
        return

    prompt = _build_overflow_recovery_prompt(stats, hard_limit=hard_limit)
    await async_summarise_message_history(barebone_model, message_history, client=client)
    messages.clear()
    messages.append({"role": "user", "content": prompt})

    hierarchy = getattr(barebone_model, "agent_hierarchy", None) or []
    mode = "fast-path" if fast_path else "retry"
    strictness = "strict" if hard_limit else "controlled"
    get_cli_output().emit(
        OutputType.AGENT_RESPONSE,
        (
            f"Context overflow recovery ({mode}, {strictness}): dropped oversized in-flight input "
            f"and requested a smaller remake ({stats['requested_tokens']}/{stats['max_tokens']} tokens, "
            f"{stats['usage_ratio'] * 100:.0f}% of limit)."
        ),
        hierarchy,
        step=0,
    )


def _build_tool_result_guardrail_message(
    executed_tool_call_list: List[dict],
    tool_results: List[str],
) -> str:
    directives: List[str] = []

    for idx, tool_call in enumerate(executed_tool_call_list):
        raw_result = tool_results[idx] if idx < len(tool_results) else ""
        if not isinstance(raw_result, str):
            continue
        try:
            parsed = json.loads(raw_result)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue

        stop_info = parsed.get("stop_repeating_tool_call")
        if not isinstance(stop_info, dict):
            continue

        tool_name = stop_info.get("tool") or tool_call.get("function", {}).get("name") or tool_call.get("name", "")
        pack_name = stop_info.get("name", "")
        book_name = stop_info.get("book", "")
        next_action = parsed.get("next_required_action") or "Use a different tool."

        directives.append(
            "CRITICAL: STOP. "
            f"Do not call {tool_name} again for context pack '{pack_name}' in book '{book_name}' "
            "during this execution. "
            f"{next_action}"
        )

    return "\n".join(dict.fromkeys(directives))


def get_total_tokens(message_history: dict) -> int:
    total = 0
    for key in ("system", "first_input", "summary"):
        d = message_history.get(key) or {}
        t = d.get("tokens", 0) or 0
        total += t if t > 0 else _estimate_tokens(d.get("message", ""))
    for msg in (message_history.get("messages") or {}).values():
        t = msg.get("tokens", 0) or 0
        total += t if t > 0 else _estimate_tokens(msg.get("message", ""))
    return total


def _check_context_and_compact(
    barebone_model: BareBoneModel,
    message_history: dict,
    messages: list,
    force: bool = False,
) -> bool:
    """Check context usage, emit warnings, and auto-compact if needed.

    Returns True if compaction was performed.
    """
    context_budget = getattr(barebone_model, 'context_budget', None)
    usage = get_context_usage(message_history, barebone_model.model_id, context_budget)
    ratio = usage["usage_ratio"]
    warning_level = usage["warning_level"]
    last_warning = message_history.get("_context_warning_issued")

    cli = get_cli_output()
    hierarchy = getattr(barebone_model, 'agent_hierarchy', None) or []

    # Emit context usage warnings (only escalate, don't repeat same level)
    if warning_level == "warning" and last_warning is None:
        LOG.info(
            "Context usage at %.0f%% (%d/%d tokens). Consider wrapping up or expect auto-compaction soon.",
            ratio * 100, usage["token_count"], usage["max_tokens"],
        )
        cli.emit(
            OutputType.AGENT_RESPONSE,
            f"Context usage: {ratio*100:.0f}% ({usage['token_count']}/{usage['max_tokens']} tokens). Auto-compaction will trigger at 80%.",
            hierarchy,
            step=0,
        )
        message_history["_context_warning_issued"] = "warning"

    if warning_level == "critical" or force:
        if last_warning != "critical":
            LOG.info(
                "Context usage at %.0f%% (%d/%d tokens). Triggering auto-compaction.",
                ratio * 100, usage["token_count"], usage["max_tokens"],
            )
            cli.emit(
                OutputType.SUMMARIZATION,
                f"Auto-compacting conversation (context at {ratio*100:.0f}%, compaction #{message_history.get('compaction_count', 0) + 1})...",
                hierarchy,
                step=0,
            )
        message_history["_context_warning_issued"] = "critical"
        summarise_message_history(barebone_model, message_history)
        _rebuild_messages_after_compaction(message_history, messages)
        # Reset warning state after compaction so warnings can fire again
        message_history["_context_warning_issued"] = None
        return True

    return False


async def _async_check_context_and_compact(
    barebone_model: BareBoneModel,
    message_history: dict,
    messages: list,
    client=None,
    force: bool = False,
) -> bool:
    """Async version: check context usage, emit warnings, and auto-compact if needed."""
    context_budget = getattr(barebone_model, 'context_budget', None)
    usage = get_context_usage(message_history, barebone_model.model_id, context_budget)
    ratio = usage["usage_ratio"]
    warning_level = usage["warning_level"]
    last_warning = message_history.get("_context_warning_issued")

    cli = get_cli_output()
    hierarchy = getattr(barebone_model, 'agent_hierarchy', None) or []

    if warning_level == "warning" and last_warning is None:
        LOG.info(
            "Context usage at %.0f%% (%d/%d tokens). Consider wrapping up or expect auto-compaction soon.",
            ratio * 100, usage["token_count"], usage["max_tokens"],
        )
        cli.emit(
            OutputType.AGENT_RESPONSE,
            f"Context usage: {ratio*100:.0f}% ({usage['token_count']}/{usage['max_tokens']} tokens). Auto-compaction will trigger at 80%.",
            hierarchy,
            step=0,
        )
        message_history["_context_warning_issued"] = "warning"

    if warning_level == "critical" or force:
        if last_warning != "critical":
            LOG.info(
                "Context usage at %.0f%% (%d/%d tokens). Triggering auto-compaction.",
                ratio * 100, usage["token_count"], usage["max_tokens"],
            )
            cli.emit(
                OutputType.SUMMARIZATION,
                f"Auto-compacting conversation (context at {ratio*100:.0f}%, compaction #{message_history.get('compaction_count', 0) + 1})...",
                hierarchy,
                step=0,
            )
        message_history["_context_warning_issued"] = "critical"
        await async_summarise_message_history(barebone_model, message_history, client)
        _rebuild_messages_after_compaction(message_history, messages)
        message_history["_context_warning_issued"] = None
        return True

    return False


def _api_request_with_context_fallback(
    build_payload_fn: Callable[[], tuple[str, dict, dict]],
    barebone_model: BareBoneModel,
    message_history: dict,
    timeout: float = 900.0,
    messages: Optional[list] = None,
) -> httpx.Response:
    recovery_stage = 0
    while True:
        api_url, headers, payload = build_payload_fn()
        stats = _get_request_token_stats(barebone_model, payload)
        if (
            recovery_stage == 0
            and messages is not None
            and stats["usage_ratio"] >= _OVERFLOW_FAST_PATH_RATIO
        ):
            LOG.warning(
                "Request preflight estimated at %.0f%% of context budget. Triggering overflow remake fast path.",
                stats["usage_ratio"] * 100,
            )
            _apply_overflow_remake_recovery(
                barebone_model,
                message_history,
                messages,
                stats,
                hard_limit=False,
                fast_path=True,
            )
            recovery_stage = 2
            continue
        try:
            return api_request_retry(api_url, headers, payload, timeout=timeout)
        except Exception as e:
            if not _is_context_length_error(e):
                raise
            stats = _get_request_token_stats(barebone_model, payload, error=e)
            if recovery_stage == 0:
                LOG.warning("Context length exceeded. Forcing summarization and retrying.")
                get_cli_output().emit(OutputType.AGENT_RESPONSE, "Context limit exceeded. Summarized history and retrying.", ["API"], step=0)
                summarise_message_history(barebone_model, message_history)
                recovery_stage = 1
                continue
            if recovery_stage == 1 and messages is not None:
                LOG.warning("Context still exceeded after summarization. Rebuilding the current input with tighter control.")
                _apply_overflow_remake_recovery(
                    barebone_model,
                    message_history,
                    messages,
                    stats,
                    hard_limit=False,
                )
                recovery_stage = 2
                continue
            if recovery_stage == 2 and messages is not None:
                LOG.warning("Context still exceeded after controlled remake. Escalating to strict remake mode.")
                _apply_overflow_remake_recovery(
                    barebone_model,
                    message_history,
                    messages,
                    stats,
                    hard_limit=True,
                )
                recovery_stage = 3
                continue
            raise


async def _api_request_with_context_fallback_async(
    build_payload_fn: Callable[[], tuple[str, dict, dict]],
    barebone_model: BareBoneModel,
    message_history: dict,
    timeout: float = 900.0,
    client: Optional[httpx.AsyncClient] = None,
    messages: Optional[list] = None,
) -> httpx.Response:
    recovery_stage = 0
    while True:
        api_url, headers, payload = build_payload_fn()
        stats = _get_request_token_stats(barebone_model, payload)
        if (
            recovery_stage == 0
            and messages is not None
            and stats["usage_ratio"] >= _OVERFLOW_FAST_PATH_RATIO
        ):
            LOG.warning(
                "Request preflight estimated at %.0f%% of context budget. Triggering overflow remake fast path.",
                stats["usage_ratio"] * 100,
            )
            await _apply_overflow_remake_recovery_async(
                barebone_model,
                message_history,
                messages,
                stats,
                hard_limit=False,
                fast_path=True,
                client=client,
            )
            recovery_stage = 2
            continue
        try:
            return await async_api_request_retry(api_url, headers, payload, timeout=timeout, client=client)
        except Exception as e:
            if not _is_context_length_error(e):
                raise
            stats = _get_request_token_stats(barebone_model, payload, error=e)
            if recovery_stage == 0:
                LOG.warning("Context length exceeded. Forcing summarization and retrying.")
                get_cli_output().emit(OutputType.AGENT_RESPONSE, "Context limit exceeded. Summarized history and retrying.", ["API"], step=0)
                await async_summarise_message_history(barebone_model, message_history, client=client)
                recovery_stage = 1
                continue
            if recovery_stage == 1 and messages is not None:
                LOG.warning("Context still exceeded after summarization. Rebuilding the current input with tighter control.")
                await _apply_overflow_remake_recovery_async(
                    barebone_model,
                    message_history,
                    messages,
                    stats,
                    hard_limit=False,
                    client=client,
                )
                recovery_stage = 2
                continue
            if recovery_stage == 2 and messages is not None:
                LOG.warning("Context still exceeded after controlled remake. Escalating to strict remake mode.")
                await _apply_overflow_remake_recovery_async(
                    barebone_model,
                    message_history,
                    messages,
                    stats,
                    hard_limit=True,
                    client=client,
                )
                recovery_stage = 3
                continue
            raise

def chat(
    barebone_model: BareBoneModel,
    messages: list[dict],
    message_history: Optional[dict] = None,
    tool_executors: Optional[Dict[str, Callable]] = None,
    logger: Optional[Any] = None,
    timeout: float = 900.0,
    max_tool_rounds: int = 5,
    max_tool_calls: Optional[int] = None,
    current_stage_index: Optional[int] = None,
    total_stages: int = 0
) -> Dict[str, Any]:
    if not barebone_model:
        raise ValueError("barebone_model is required")
    if not messages:
        raise ValueError("messages is required and cannot be empty")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if not hasattr(barebone_model, 'model_id') or not barebone_model.model_id:
        raise ValueError("barebone_model.model_id is required")
    if not hasattr(barebone_model, 'api_key') or not barebone_model.api_key:
        raise ValueError("barebone_model.api_key is required")
    if not hasattr(barebone_model, 'api_url') or not barebone_model.api_url:
        raise ValueError("barebone_model.api_url is required")
    
    if tool_executors is not None and not isinstance(tool_executors, dict):
        raise ValueError("tool_executors must be a dictionary if provided")
    
    message_history = message_history or init_message_history()
    # Ensure compaction tracking fields exist (for histories created before this change)
    message_history.setdefault("compaction_count", 0)
    message_history.setdefault("_context_warning_issued", None)
    tool_executors = tool_executors or {}
    if logger:
        logger.log_input(messages)

    _check_context_and_compact(barebone_model, message_history, messages)

    if not message_history["first_input"]["message"] and messages:
        message_history["first_input"]["message"] = messages[0].get("content", str(messages[0]))
        message_history["first_input"]["tokens"] = 0
    
    use_responses_api = getattr(barebone_model, "use_responses_api", False)
    provider = get_provider(barebone_model.model_id, barebone_model.api_url, use_responses_api)

    def _build():
        return build_provider_request(provider, barebone_model, messages, message_history)
    
    response = _api_request_with_context_fallback(_build, barebone_model, message_history, timeout, messages)
    
    response.raise_for_status()
    data = response.json()
    
    content, reasoning_content, tool_calls, tokens = parse_provider_response(provider, data, barebone_model.model_id)
    usage_info = extract_usage(provider, data)
    if usage_info:
        tokens = usage_info.get("total_tokens", tokens)

    # Accumulate usage across all API rounds in this chat call
    total_usage = {
        "input_tokens": (usage_info or {}).get("input_tokens", 0) or 0,
        "output_tokens": (usage_info or {}).get("output_tokens", 0) or 0,
        "total_tokens": (usage_info or {}).get("total_tokens", 0) or 0,
        "input_cached_tokens": (usage_info or {}).get("input_cached_tokens", 0) or 0,
    }

    cost_info = logger.compute_cost(barebone_model.model_id, total_usage) if logger else None

    all_executed_tool_call_list: List[dict] = []
    content_before_tools = content
    tool_call_counts = getattr(barebone_model, '_tool_call_counts', None) or {}
    terminal_tool_names = _get_terminal_control_tool_names(barebone_model)
    final_tool_name = _get_primary_final_tool_name(barebone_model)
    rounds = 0
    total_tool_calls_in_cycle = 0
    recent_tool_calls = []
    hijacked = False

    if not hasattr(barebone_model, '_current_step'):
        barebone_model._current_step = 0
    
    agent_hierarchy = getattr(barebone_model, 'agent_hierarchy', None)

    while tool_calls and tool_executors and rounds < max_tool_rounds:
        repeated_tool_names: List[str] = []
        for tool_call in tool_calls:
            fn = tool_call.get("function", {})
            tool_name = fn.get("name") or tool_call.get("name", "")
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            
            signature = (tool_name, json.dumps(args, sort_keys=True))
            recent_count = recent_tool_calls.count(signature)

            if recent_count >= 3:
                repeated_tool_names.append(tool_name)
                LOG.error(f"Tool '{tool_name}' has been called {recent_count} times with identical arguments. Agent may be stuck in a loop.")

                # Force termination after 5 identical calls
                if recent_count >= 5:
                    LOG.error(f"Tool '{tool_name}' called {recent_count} times identically. Forcing agent termination with summarization.")

                    # Use summarization to create a force_answer response
                    from ..summarization import run_summarization
                    force_answer = run_summarization(
                        barebone_model,
                        message_history,
                        prompt_kind="force_answer",
                        write_to_history=False,
                        use_same_model=True,
                    )

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

                    cost_info = logger.compute_cost(barebone_model.model_id, total_usage) if logger else None
                    return {
                        "content": force_content,
                        "reasoning_content": None,
                        "tool_calls": [],
                        "executed_tool_calls": all_executed_tool_call_list,
                        "content_before_tools": content_before_tools,
                        "message_history": message_history,
                        "usage": total_usage,
                        "cost": cost_info,
                        "hijacked": True,
                        "compaction_count": message_history.get("compaction_count", 0),
                    }

        tool_metadata = {}
        if hasattr(barebone_model, 'agent_tools'):
            for agent_tool in barebone_model.agent_tools:
                _tool_params = {}
                if hasattr(agent_tool, 'args') and agent_tool.args:
                    _ta = agent_tool.args
                    if hasattr(_ta, 'properties') and _ta.properties:
                        _req = list(_ta.properties.get("__required__", []))
                        _props = {k: v for k, v in _ta.properties.items() if k != "__required__" and isinstance(v, dict)}
                        _tool_params = {"properties": _props, "required": _req}
                tool_metadata[agent_tool.name] = {
                    "parallel": agent_tool.parallel if hasattr(agent_tool, 'parallel') else True,
                    "limit_calls": agent_tool.limit_calls if hasattr(agent_tool, 'limit_calls') else 0,
                    "parameters": _tool_params,
                }
        step = getattr(barebone_model, '_current_step', 0)
        get_cli_output().emit(
            OutputType.AGENT_RESPONSE,
            f"tool_calls_from_llm: {tool_calls}",
            list(agent_hierarchy or []),
            step=step,
        )
        tool_messages, tool_results, updated_counts, executed_tool_call_list = execute_tool_calls(tool_calls, tool_executors, provider, timeout, tool_metadata, agent_hierarchy, step, tool_call_counts)
        if hasattr(barebone_model, '_tool_call_counts'):
            barebone_model._tool_call_counts.update(updated_counts)
        all_executed_tool_call_list.extend(executed_tool_call_list)
        
        num_tools_executed = len(executed_tool_call_list)
        total_tool_calls_in_cycle += num_tools_executed
        if hasattr(barebone_model, '_current_step'):
            barebone_model._current_step += num_tools_executed
        
        for tool_call in executed_tool_call_list:
            fn = tool_call.get("function", {})
            tool_name = fn.get("name") or tool_call.get("name", "")
            args = fn.get("arguments", "{}")
            signature = (tool_name, args)
            recent_tool_calls.append(signature)
            if len(recent_tool_calls) > 10:
                recent_tool_calls.pop(0)
        
        if logger:
            logger.log_tool_results(executed_tool_call_list, tool_results)
        
        LOG.debug("[TRACE] chat: Checking terminal control tools")
        agent_end_called = False
        stage_end_called = False
        for tc in executed_tool_call_list:
            name = tc.get("function", {}).get("name") or tc.get("name", "")
            if name in terminal_tool_names:
                agent_end_called = True
            elif name == "stage_end":
                stage_end_called = True

        if agent_end_called:
            if logger:
                cost_info = logger.compute_cost(barebone_model.model_id, total_usage)
                logger.log_output(content, total_usage, cost_info, message_history)
            _record_model_message(message_history, content, tokens, reasoning_content)
            response_payload = _build_chat_response(
                content,
                reasoning_content,
                [],
                all_executed_tool_call_list,
                content_before_tools,
                message_history,
                total_usage,
                cost_info,
                hijacked,
            )
            raise AgentEndException(response_payload)

        if stage_end_called:
            tool_calls = []
            break
        
        if max_tool_calls and barebone_model._current_step >= max_tool_calls:
            LOG.debug("[TRACE] chat: Max tool calls reached")
            is_last_stage = (total_stages > 0 and current_stage_index is not None 
                           and current_stage_index == total_stages - 1)
            has_stages = total_stages > 0
            
            if has_stages and not is_last_stage:
                control_tool = "stage_end"
                hijack_message = f"CRITICAL: Maximum tool call limit ({max_tool_calls}) reached at step {barebone_model._current_step}. You MUST call stage_end to advance to the next stage."
            else:
                control_tool = final_tool_name
                hijack_message = (
                    f"CRITICAL: Maximum tool call limit ({max_tool_calls}) reached at step {barebone_model._current_step}. "
                    f"You MUST call {final_tool_name} with your final answer."
                )
            
            msg_id = str(uuid.uuid4())
            message_history["messages"][msg_id] = {
                "message": hijack_message,
                "tokens": 0,
                "type": "system_directive",
            }
            
            LOG.warning(f"Max tool calls ({max_tool_calls}) reached. Injected {control_tool} directive.")
            hijacked = True
            tool_calls = []
            break

        repeated_warning_msg = ""
        if repeated_tool_names:
            seen = list(dict.fromkeys(repeated_tool_names))
            repeated_warning_msg = (
                "System note: You have already called the following tool(s) multiple times with the same arguments: "
                + ", ".join(seen)
                + f". Do not repeat these calls. Proceed to the next step (e.g. use submit_discovery or other tools, then {final_tool_name} when done)."
            )
        guardrail_warning_msg = _build_tool_result_guardrail_message(
            executed_tool_call_list,
            tool_results,
        )
        if guardrail_warning_msg:
            repeated_warning_msg = (
                f"{repeated_warning_msg}\n{guardrail_warning_msg}".strip()
                if repeated_warning_msg else guardrail_warning_msg
            )

        LOG.debug("[TRACE] chat: Appending tool messages")
        append_provider_tool_messages(
            provider, messages, message_history, content, reasoning_content,
            executed_tool_call_list, tool_messages, tool_results, tokens,
            repeated_warning_msg, format_gemini_results
        )

        # Mid-tool-round compaction check
        _check_context_and_compact(barebone_model, message_history, messages)

        rounds += 1
        if rounds >= max_tool_rounds:
            LOG.debug("[TRACE] chat: Max tool rounds reached")
            tool_calls = []
            break

        def _build_follow():
            LOG.debug("[TRACE] chat: Building provider request")
            return build_provider_request(provider, barebone_model, messages, message_history)
        
        LOG.debug("[TRACE] chat: Sending follow-up API request")
        response = _api_request_with_context_fallback(_build_follow, barebone_model, message_history, timeout, messages)
        LOG.debug("[TRACE] chat: API request returned")
        response.raise_for_status()
        data = response.json()
        
        content, reasoning_content, tool_calls, tokens = parse_provider_response(provider, data, barebone_model.model_id)
        usage_info = extract_usage(provider, data)
        if usage_info:
            tokens = usage_info.get("total_tokens", 0)
            # Accumulate into total_usage
            for key in ("input_tokens", "output_tokens", "total_tokens", "input_cached_tokens"):
                total_usage[key] = total_usage.get(key, 0) + ((usage_info.get(key, 0)) or 0)

    if tool_calls and tool_executors:
        tool_metadata = {}
        if hasattr(barebone_model, 'agent_tools'):
            for agent_tool in barebone_model.agent_tools:
                _tool_params = {}
                if hasattr(agent_tool, 'args') and agent_tool.args:
                    _ta = agent_tool.args
                    if hasattr(_ta, 'properties') and _ta.properties:
                        _req = list(_ta.properties.get("__required__", []))
                        _props = {k: v for k, v in _ta.properties.items() if k != "__required__" and isinstance(v, dict)}
                        _tool_params = {"properties": _props, "required": _req}
                tool_metadata[agent_tool.name] = {
                    "parallel": agent_tool.parallel if hasattr(agent_tool, 'parallel') else True,
                    "limit_calls": agent_tool.limit_calls if hasattr(agent_tool, 'limit_calls') else 0,
                    "parameters": _tool_params,
                }
        agent_hierarchy = getattr(barebone_model, 'agent_hierarchy', None)
        step = getattr(barebone_model, '_current_step', 0)
        get_cli_output().emit(
            OutputType.AGENT_RESPONSE,
            f"tool_calls_from_llm: {tool_calls}",
            list(agent_hierarchy or []),
            step=step,
        )
        tool_messages, tool_results, updated_counts, executed_tool_call_list = execute_tool_calls(tool_calls, tool_executors, provider, timeout, tool_metadata, agent_hierarchy, step, tool_call_counts)
        if hasattr(barebone_model, '_tool_call_counts'):
            barebone_model._tool_call_counts.update(updated_counts)
        all_executed_tool_call_list.extend(executed_tool_call_list)
        if logger:
            logger.log_tool_results(executed_tool_call_list, tool_results)
        
        agent_end_called = False
        stage_end_called = False
        for tc in executed_tool_call_list:
            name = tc.get("function", {}).get("name") or tc.get("name", "")
            if name in terminal_tool_names:
                agent_end_called = True
            elif name == "stage_end":
                stage_end_called = True

        if agent_end_called:
            if logger:
                cost_info = logger.compute_cost(barebone_model.model_id, total_usage)
                logger.log_output(content, total_usage, cost_info, message_history)
            _record_model_message(message_history, content, tokens, reasoning_content)
            response_payload = _build_chat_response(
                content,
                reasoning_content,
                [],
                all_executed_tool_call_list,
                content_before_tools,
                message_history,
                total_usage,
                cost_info,
                hijacked,
            )
            raise AgentEndException(response_payload)

        if not stage_end_called:
            append_provider_tool_messages(
                provider, messages, message_history, content, reasoning_content,
                executed_tool_call_list, tool_messages, tool_results, tokens,
                "", format_gemini_results
            )
        else:
            tool_calls = []

        tool_calls = []

    cost_info = logger.compute_cost(barebone_model.model_id, total_usage) if logger else None
    if logger:
        logger.log_output(content, total_usage, cost_info, message_history)

    _record_model_message(message_history, content, tokens, reasoning_content)

    return _build_chat_response(
        content,
        reasoning_content,
        tool_calls,
        all_executed_tool_call_list,
        content_before_tools,
        message_history,
        total_usage,
        cost_info,
        hijacked,
    )


async def async_chat(
    barebone_model: BareBoneModel,
    messages: list[dict],
    message_history: Optional[dict] = None,
    tool_executors: Optional[Dict[str, Callable]] = None,
    logger: Optional[Any] = None,
    timeout: float = 900.0,
    client: Optional[httpx.AsyncClient] = None,
    max_tool_rounds: int = 5,
    max_tool_calls: Optional[int] = None,
    current_stage_index: Optional[int] = None,
    total_stages: int = 0
) -> Dict[str, Any]:
    if not barebone_model:
        raise ValueError("barebone_model is required")
    if not messages:
        raise ValueError("messages is required and cannot be empty")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    if not hasattr(barebone_model, 'model_id') or not barebone_model.model_id:
        raise ValueError("barebone_model.model_id is required")
    if not hasattr(barebone_model, 'api_key') or not barebone_model.api_key:
        raise ValueError("barebone_model.api_key is required")
    if not hasattr(barebone_model, 'api_url') or not barebone_model.api_url:
        raise ValueError("barebone_model.api_url is required")

    if tool_executors is not None and not isinstance(tool_executors, dict):
        raise ValueError("tool_executors must be a dictionary if provided")

    message_history = message_history or init_message_history()
    message_history.setdefault("compaction_count", 0)
    message_history.setdefault("_context_warning_issued", None)
    tool_executors = tool_executors or {}
    if logger:
        logger.log_input(messages)

    await _async_check_context_and_compact(barebone_model, message_history, messages, client)

    if not message_history["first_input"]["message"] and messages:
        message_history["first_input"]["message"] = messages[0].get("content", str(messages[0]))
        message_history["first_input"]["tokens"] = 0

    use_responses_api = getattr(barebone_model, "use_responses_api", False)
    provider = get_provider(barebone_model.model_id, barebone_model.api_url, use_responses_api)

    # Create shared client if not provided
    should_close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)

    try:
        def _build():
            return build_provider_request(provider, barebone_model, messages, message_history)
        
        response = await _api_request_with_context_fallback_async(_build, barebone_model, message_history, timeout, client, messages)
        response.raise_for_status()
        data = response.json()

        content, reasoning_content, tool_calls, tokens = parse_provider_response(provider, data, barebone_model.model_id)
        usage_info = extract_usage(provider, data)
        if usage_info:
            tokens = usage_info.get("total_tokens", tokens)

        # Accumulate usage across all API rounds in this chat call
        total_usage = {
            "input_tokens": (usage_info or {}).get("input_tokens", 0) or 0,
            "output_tokens": (usage_info or {}).get("output_tokens", 0) or 0,
            "total_tokens": (usage_info or {}).get("total_tokens", 0) or 0,
            "input_cached_tokens": (usage_info or {}).get("input_cached_tokens", 0) or 0,
        }

        cost_info = logger.compute_cost(barebone_model.model_id, total_usage) if logger else None

        all_executed_tool_call_list: List[dict] = []
        content_before_tools = content
        tool_call_counts = getattr(barebone_model, '_tool_call_counts', None) or {}
        terminal_tool_names = _get_terminal_control_tool_names(barebone_model)
        final_tool_name = _get_primary_final_tool_name(barebone_model)
        rounds = 0
        total_tool_calls_in_cycle = 0
        recent_tool_calls = []
        hijacked = False

        if not hasattr(barebone_model, '_current_step'):
            barebone_model._current_step = 0
        
        agent_hierarchy = getattr(barebone_model, 'agent_hierarchy', None)

        while tool_calls and tool_executors and rounds < max_tool_rounds:
            repeated_tool_names_async: List[str] = []
            for tool_call in tool_calls:
                fn = tool_call.get("function", {})
                tool_name = fn.get("name") or tool_call.get("name", "")
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {}
                
                signature = (tool_name, json.dumps(args, sort_keys=True))
                recent_count = recent_tool_calls.count(signature)

                if recent_count >= 3:
                    repeated_tool_names_async.append(tool_name)
                    LOG.error(f"Tool '{tool_name}' has been called {recent_count} times with identical arguments. Agent may be stuck in a loop.")

                    # Force termination after 5 identical calls
                    if recent_count >= 5:
                        LOG.error(f"Tool '{tool_name}' called {recent_count} times identically. Forcing agent termination with summarization.")

                        # Use summarization to create a force_answer response
                        from ..summarization import async_summarise_message_history
                        try:
                            force_answer = await async_summarise_message_history(
                                barebone_model,
                                message_history,
                                client=client,
                                use_same_model=True,
                                prompt_kind="force_answer",
                                write_to_history=False,
                            )
                        except Exception as e:
                            LOG.error(f"Failed to generate force_answer summary: {e}")
                            force_answer = ""

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

                        cost_info = logger.compute_cost(barebone_model.model_id, total_usage) if logger else None
                        return {
                            "content": force_content,
                            "reasoning_content": None,
                            "tool_calls": [],
                            "executed_tool_calls": all_executed_tool_call_list,
                            "content_before_tools": content_before_tools,
                            "message_history": message_history,
                            "usage": total_usage,
                            "cost": cost_info,
                            "hijacked": True,
                            "compaction_count": message_history.get("compaction_count", 0),
                        }

            repeated_warning_msg_async = ""
            if repeated_tool_names_async:
                seen_async = list(dict.fromkeys(repeated_tool_names_async))
                repeated_warning_msg_async = (
                    "System note: You have already called the following tool(s) multiple times with the same arguments: "
                    + ", ".join(seen_async)
                    + f". Do not repeat these calls. Proceed to the next step (e.g. use submit_discovery or other tools, then {final_tool_name} when done)."
                )

            tool_metadata = {}
            if hasattr(barebone_model, 'agent_tools'):
                for agent_tool in barebone_model.agent_tools:
                    tool_metadata[agent_tool.name] = {
                        "parallel": agent_tool.parallel if hasattr(agent_tool, 'parallel') else True,
                        "limit_calls": agent_tool.limit_calls if hasattr(agent_tool, 'limit_calls') else 0
                    }
            step = getattr(barebone_model, '_current_step', 0)
            get_cli_output().emit(
                OutputType.AGENT_RESPONSE,
                f"tool_calls_from_llm: {tool_calls}",
                list(agent_hierarchy or []),
                step=step,
            )
            tool_messages, tool_results, updated_counts, executed_tool_call_list = await async_execute_tool_calls(tool_calls, tool_executors, provider, timeout, tool_metadata, agent_hierarchy, step, tool_call_counts)
            if hasattr(barebone_model, '_tool_call_counts'):
                barebone_model._tool_call_counts.update(updated_counts)
            all_executed_tool_call_list.extend(executed_tool_call_list)
            
            num_tools_executed = len(executed_tool_call_list)
            total_tool_calls_in_cycle += num_tools_executed
            if hasattr(barebone_model, '_current_step'):
                barebone_model._current_step += num_tools_executed
            
            for tool_call in executed_tool_call_list:
                fn = tool_call.get("function", {})
                tool_name = fn.get("name") or tool_call.get("name", "")
                args = fn.get("arguments", "{}")
                signature = (tool_name, args)
                recent_tool_calls.append(signature)
                if len(recent_tool_calls) > 10:
                    recent_tool_calls.pop(0)
            
            if logger:
                logger.log_tool_results(executed_tool_call_list, tool_results)
            
            LOG.debug("[TRACE] async_chat: Checking terminal control tools")
            agent_end_called = False
            stage_end_called = False
            for tc in executed_tool_call_list:
                name = tc.get("function", {}).get("name") or tc.get("name", "")
                if name in terminal_tool_names:
                    agent_end_called = True
                elif name == "stage_end":
                    stage_end_called = True

            if agent_end_called:
                if logger:
                    cost_info = logger.compute_cost(barebone_model.model_id, total_usage)
                    logger.log_output(content, total_usage, cost_info, message_history)
                _record_model_message(message_history, content, tokens, reasoning_content)
                response_payload = _build_chat_response(
                    content,
                    reasoning_content,
                    [],
                    all_executed_tool_call_list,
                    content_before_tools,
                    message_history,
                    total_usage,
                    cost_info,
                    hijacked,
                )
                raise AgentEndException(response_payload)

            if stage_end_called:
                tool_calls = []
                break
            
            if max_tool_calls and barebone_model._current_step >= max_tool_calls:
                LOG.debug("[TRACE] async_chat: Max tool calls reached")
                is_last_stage = (total_stages > 0 and current_stage_index is not None 
                               and current_stage_index == total_stages - 1)
                has_stages = total_stages > 0
                
                if has_stages and not is_last_stage:
                    control_tool = "stage_end"
                    hijack_message = f"CRITICAL: Maximum tool call limit ({max_tool_calls}) reached at step {barebone_model._current_step}. You MUST call stage_end to advance to the next stage."
                else:
                    control_tool = final_tool_name
                    hijack_message = (
                        f"CRITICAL: Maximum tool call limit ({max_tool_calls}) reached at step {barebone_model._current_step}. "
                        f"You MUST call {final_tool_name} with your final answer."
                    )
                
                msg_id = str(uuid.uuid4())
                message_history["messages"][msg_id] = {
                    "message": hijack_message,
                    "tokens": 0,
                    "type": "system_directive",
                }
                
                LOG.warning(f"Max tool calls ({max_tool_calls}) reached. Injected {control_tool} directive.")
                hijacked = True
                tool_calls = []
                break

            guardrail_warning_msg_async = _build_tool_result_guardrail_message(
                executed_tool_call_list,
                tool_results,
            )
            if guardrail_warning_msg_async:
                repeated_warning_msg_async = (
                    f"{repeated_warning_msg_async}\n{guardrail_warning_msg_async}".strip()
                    if repeated_warning_msg_async else guardrail_warning_msg_async
                )

            LOG.debug("[TRACE] async_chat: Appending tool messages")
            append_provider_tool_messages(
                provider, messages, message_history, content, reasoning_content,
                executed_tool_call_list, tool_messages, tool_results, tokens,
                repeated_warning_msg_async, format_gemini_results
            )

            # Mid-tool-round compaction check
            await _async_check_context_and_compact(barebone_model, message_history, messages, client)

            rounds += 1
            if rounds >= max_tool_rounds:
                LOG.debug("[TRACE] async_chat: Max tool rounds reached")
                tool_calls = []
                break

            def _build_follow():
                LOG.debug("[TRACE] async_chat: Building provider request")
                return build_provider_request(provider, barebone_model, messages, message_history)
            
            LOG.debug("[TRACE] async_chat: Sending follow-up API request")
            response = await _api_request_with_context_fallback_async(_build_follow, barebone_model, message_history, timeout, client, messages)
            LOG.debug("[TRACE] async_chat: API request returned")
            response.raise_for_status()
            data = response.json()
            
            content, reasoning_content, tool_calls, tokens = parse_provider_response(provider, data, barebone_model.model_id)
            usage_info = extract_usage(provider, data)
            if usage_info:
                tokens = usage_info.get("total_tokens", 0)
                # Accumulate into total_usage
                for key in ("input_tokens", "output_tokens", "total_tokens", "input_cached_tokens"):
                    total_usage[key] = total_usage.get(key, 0) + ((usage_info.get(key, 0)) or 0)

        if tool_calls and tool_executors:
            tool_metadata = {}
            if hasattr(barebone_model, 'agent_tools'):
                for agent_tool in barebone_model.agent_tools:
                    tool_metadata[agent_tool.name] = {
                        "parallel": agent_tool.parallel if hasattr(agent_tool, 'parallel') else True,
                        "limit_calls": agent_tool.limit_calls if hasattr(agent_tool, 'limit_calls') else 0
                    }
            agent_hierarchy = getattr(barebone_model, 'agent_hierarchy', None)
            step = getattr(barebone_model, '_current_step', 0)
            get_cli_output().emit(
                OutputType.AGENT_RESPONSE,
                f"tool_calls_from_llm: {tool_calls}",
                list(agent_hierarchy or []),
                step=step,
            )
            tool_messages, tool_results, updated_counts, executed_tool_call_list = await async_execute_tool_calls(tool_calls, tool_executors, provider, timeout, tool_metadata, agent_hierarchy, step, tool_call_counts)
            if hasattr(barebone_model, '_tool_call_counts'):
                barebone_model._tool_call_counts.update(updated_counts)
            all_executed_tool_call_list.extend(executed_tool_call_list)
            if logger:
                logger.log_tool_results(executed_tool_call_list, tool_results)
            
            agent_end_called = False
            stage_end_called = False
            for tc in executed_tool_call_list:
                name = tc.get("function", {}).get("name") or tc.get("name", "")
                if name in terminal_tool_names:
                    agent_end_called = True
                elif name == "stage_end":
                    stage_end_called = True

            if agent_end_called:
                if logger:
                    cost_info = logger.compute_cost(barebone_model.model_id, total_usage)
                    logger.log_output(content, total_usage, cost_info, message_history)
                _record_model_message(message_history, content, tokens, reasoning_content)
                response_payload = _build_chat_response(
                    content,
                    reasoning_content,
                    [],
                    all_executed_tool_call_list,
                    content_before_tools,
                    message_history,
                    total_usage,
                    cost_info,
                    hijacked,
                )
                raise AgentEndException(response_payload)

            if not stage_end_called:
                append_provider_tool_messages(
                    provider, messages, message_history, content, reasoning_content,
                    executed_tool_call_list, tool_messages, tool_results, tokens,
                    "", format_gemini_results
                )
            else:
                tool_calls = []

            tool_calls = []

        cost_info = logger.compute_cost(barebone_model.model_id, total_usage) if logger else None
        if logger:
            logger.log_output(content, total_usage, cost_info, message_history)

        _record_model_message(message_history, content, tokens, reasoning_content)

        return _build_chat_response(
            content,
            reasoning_content,
            tool_calls,
            all_executed_tool_call_list,
            content_before_tools,
            message_history,
            total_usage,
            cost_info,
            hijacked,
        )

    finally:
        if should_close_client:
            await client.aclose()
