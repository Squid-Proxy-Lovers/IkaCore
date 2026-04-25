import json
import logging
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
    }


def init_message_history() -> dict:
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


def _api_request_with_context_fallback(
    build_payload_fn: Callable[[], tuple[str, dict, dict]],
    barebone_model: BareBoneModel,
    message_history: dict,
    timeout: float = 900.0
) -> httpx.Response:
    api_url, headers, payload = build_payload_fn()
    try:
        return api_request_retry(api_url, headers, payload, timeout=timeout)
    except Exception as e:
        if not _is_context_length_error(e):
            raise
        LOG.warning("Context length exceeded. Forcing summarization and retrying.")
        get_cli_output().emit(OutputType.AGENT_RESPONSE, "Context limit exceeded. Summarized history and retrying.", ["API"], step=0)
        summarise_message_history(barebone_model, message_history)
        api_url, headers, payload = build_payload_fn()
        return api_request_retry(api_url, headers, payload, timeout=timeout)


async def _api_request_with_context_fallback_async(
    build_payload_fn: Callable[[], tuple[str, dict, dict]],
    barebone_model: BareBoneModel,
    message_history: dict,
    timeout: float = 900.0,
    client: Optional[httpx.AsyncClient] = None
) -> httpx.Response:
    api_url, headers, payload = build_payload_fn()
    try:
        return await async_api_request_retry(api_url, headers, payload, timeout=timeout, client=client)
    except Exception as e:
        if not _is_context_length_error(e):
            raise
        LOG.warning("Context length exceeded. Forcing summarization and retrying.")
        get_cli_output().emit(OutputType.AGENT_RESPONSE, "Context limit exceeded. Summarized history and retrying.", ["API"], step=0)
        await async_summarise_message_history(barebone_model, message_history, client=client)
        api_url, headers, payload = build_payload_fn()

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
    tool_executors = tool_executors or {}
    if logger:
        logger.log_input(messages)
    
    token_count = get_total_tokens(message_history)
    max_tokens = getattr(barebone_model, 'context_budget', None) or get_max_tokens(barebone_model.model_id)

    if token_count > max_tokens * 0.8:
        #LOG.info(f"Token count ({token_count}) approaching limit ({max_tokens}). Summarizing history...")
        summarise_message_history(barebone_model, message_history)
    
    if not message_history["first_input"]["message"] and messages:
        message_history["first_input"]["message"] = messages[0].get("content", str(messages[0]))
        message_history["first_input"]["tokens"] = 0
    
    use_responses_api = getattr(barebone_model, "use_responses_api", True)
    provider = get_provider(barebone_model.model_id, barebone_model.api_url, use_responses_api)

    def _build():
        return build_provider_request(provider, barebone_model, messages, message_history)
    
    response = _api_request_with_context_fallback(_build, barebone_model, message_history, timeout)
    
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
                    from .summarization import run_summarization
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
        print("tool_calls_from_llm:", tool_calls)
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
            args_raw = fn.get("arguments", "{}")
            # Normalize to match the signature format used by the repeat-detection
            # check above (line ~230). Storing the raw provider string here would
            # never match the sorted/no-whitespace dump used for lookup, so the
            # 5-strike force-terminate guard would never fire.
            try:
                args_parsed = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args_parsed = {}
            signature = (tool_name, json.dumps(args_parsed, sort_keys=True))
            recent_tool_calls.append(signature)
            if len(recent_tool_calls) > 10:
                recent_tool_calls.pop(0)
        
        if logger:
            logger.log_tool_results(executed_tool_call_list, tool_results)
        
        LOG.debug("[TRACE] chat: Checking agent_end")
        agent_end_called = False
        stage_end_called = False
        for tc in executed_tool_call_list:
            name = tc.get("function", {}).get("name") or tc.get("name", "")
            if name == "agent_end":
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
                control_tool = "agent_end"
                hijack_message = f"CRITICAL: Maximum tool call limit ({max_tool_calls}) reached at step {barebone_model._current_step}. You MUST call agent_end with your final answer."
            
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
                + ", ".join(seen) + ". Do not repeat these calls. Proceed to the next step (e.g. use submit_discovery or other tools, then agent_end when done)."
            )

        LOG.debug("[TRACE] chat: Appending tool messages")
        append_provider_tool_messages(
            provider, messages, message_history, content, reasoning_content,
            executed_tool_call_list, tool_messages, tool_results, tokens,
            repeated_warning_msg, format_gemini_results
        )

        rounds += 1
        if rounds >= max_tool_rounds:
            LOG.debug("[TRACE] chat: Max tool rounds reached")
            tool_calls = []
            break

        def _build_follow():
            LOG.debug("[TRACE] chat: Building provider request")
            return build_provider_request(provider, barebone_model, messages, message_history)
        
        LOG.debug("[TRACE] chat: Sending follow-up API request")
        response = _api_request_with_context_fallback(_build_follow, barebone_model, message_history, timeout)
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
        print("tool_calls_from_llm:", tool_calls)
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
            if name == "agent_end":
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
    tool_executors = tool_executors or {}
    if logger:
        logger.log_input(messages)

    token_count = get_total_tokens(message_history)
    max_tokens = getattr(barebone_model, 'context_budget', None) or get_max_tokens(barebone_model.model_id)

    if token_count > max_tokens * 0.8:
        LOG.info(f"Token count ({token_count}) approaching limit ({max_tokens}). Summarizing history...")
        await async_summarise_message_history(barebone_model, message_history, client)

    if not message_history["first_input"]["message"] and messages:
        message_history["first_input"]["message"] = messages[0].get("content", str(messages[0]))
        message_history["first_input"]["tokens"] = 0

    use_responses_api = getattr(barebone_model, "use_responses_api", True)
    provider = get_provider(barebone_model.model_id, barebone_model.api_url, use_responses_api)

    # Create shared client if not provided
    should_close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)

    try:
        def _build():
            return build_provider_request(provider, barebone_model, messages, message_history)
        
        response = await _api_request_with_context_fallback_async(_build, barebone_model, message_history, timeout, client)
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
                        from .summarization import async_summarise_message_history
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
                        }

            repeated_warning_msg_async = ""
            if repeated_tool_names_async:
                seen_async = list(dict.fromkeys(repeated_tool_names_async))
                repeated_warning_msg_async = (
                    "System note: You have already called the following tool(s) multiple times with the same arguments: "
                    + ", ".join(seen_async) + ". Do not repeat these calls. Proceed to the next step (e.g. use submit_discovery or other tools, then agent_end when done)."
                )

            tool_metadata = {}
            if hasattr(barebone_model, 'agent_tools'):
                for agent_tool in barebone_model.agent_tools:
                    tool_metadata[agent_tool.name] = {
                        "parallel": agent_tool.parallel if hasattr(agent_tool, 'parallel') else True,
                        "limit_calls": agent_tool.limit_calls if hasattr(agent_tool, 'limit_calls') else 0
                    }
            step = getattr(barebone_model, '_current_step', 0)
            print("tool_calls_from_llm:", tool_calls)
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
                args_raw = fn.get("arguments", "{}")
                # Normalize so this matches the lookup signature built above.
                try:
                    args_parsed = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args_parsed = {}
                signature = (tool_name, json.dumps(args_parsed, sort_keys=True))
                recent_tool_calls.append(signature)
                if len(recent_tool_calls) > 10:
                    recent_tool_calls.pop(0)
            
            if logger:
                logger.log_tool_results(executed_tool_call_list, tool_results)
            
            LOG.debug("[TRACE] async_chat: Checking agent_end")
            agent_end_called = False
            stage_end_called = False
            for tc in executed_tool_call_list:
                name = tc.get("function", {}).get("name") or tc.get("name", "")
                if name == "agent_end":
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
                    control_tool = "agent_end"
                    hijack_message = f"CRITICAL: Maximum tool call limit ({max_tool_calls}) reached at step {barebone_model._current_step}. You MUST call agent_end with your final answer."
                
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

            LOG.debug("[TRACE] async_chat: Appending tool messages")
            append_provider_tool_messages(
                provider, messages, message_history, content, reasoning_content,
                executed_tool_call_list, tool_messages, tool_results, tokens,
                repeated_warning_msg_async, format_gemini_results
            )

            rounds += 1
            if rounds >= max_tool_rounds:
                LOG.debug("[TRACE] async_chat: Max tool rounds reached")
                tool_calls = []
                break

            def _build_follow():
                LOG.debug("[TRACE] async_chat: Building provider request")
                return build_provider_request(provider, barebone_model, messages, message_history)
            
            LOG.debug("[TRACE] async_chat: Sending follow-up API request")
            response = await _api_request_with_context_fallback_async(_build_follow, barebone_model, message_history, timeout, client)
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
            print("tool_calls_from_llm:", tool_calls)
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
                if name == "agent_end":
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
