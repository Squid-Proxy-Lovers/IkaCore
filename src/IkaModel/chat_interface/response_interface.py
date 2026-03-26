import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from copy import deepcopy
from typing import Any, Dict, List, Optional, Callable

from IkaCore.cli_output import get_cli_output

LOG = logging.getLogger(__name__)

_TOOL_EXECUTOR_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool-exec")
_TOOL_EXECUTOR_LOCK = threading.Lock()


def _resolve_tool_timeout(
    tool_name: str,
    tool_executors: Dict[str, Callable],
    default_timeout: float,
) -> float:
    executor_fn = tool_executors.get(tool_name)
    override = getattr(executor_fn, "__tool_timeout__", None) if executor_fn else None
    try:
        return float(override) if override is not None else float(default_timeout)
    except (TypeError, ValueError):
        LOG.warning(
            "Invalid timeout override %r for tool '%s'; using default timeout %ss",
            override,
            tool_name,
            default_timeout,
        )
        return float(default_timeout)


def _ensure_tool_executor_pool() -> ThreadPoolExecutor:
    """Ensure the global tool executor is alive before scheduling work."""
    global _TOOL_EXECUTOR_POOL
    with _TOOL_EXECUTOR_LOCK:
        is_shutdown = getattr(_TOOL_EXECUTOR_POOL, "_shutdown", False)
        if is_shutdown:
            _TOOL_EXECUTOR_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool-exec")
    return _TOOL_EXECUTOR_POOL


def _reset_tool_executor_pool() -> ThreadPoolExecutor:
    """Force-create a fresh executor pool after a scheduling race."""
    global _TOOL_EXECUTOR_POOL
    with _TOOL_EXECUTOR_LOCK:
        _TOOL_EXECUTOR_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool-exec")
    return _TOOL_EXECUTOR_POOL

def extract_usage(provider: str, data: dict) -> Dict[str, Any]:
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "input_cached_tokens": 0}
    raw_usage = data.get("usage", {}) or {}

    if provider in ["deepseek", "openai", "openrouter"]:
        usage["input_tokens"] = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
        usage["output_tokens"] = raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0))
        usage["total_tokens"] = raw_usage.get("total_tokens", usage["input_tokens"] + usage["output_tokens"])
    elif provider == "openai_responses":
        usage["input_tokens"] = raw_usage.get("input_tokens", 0)
        usage["output_tokens"] = raw_usage.get("output_tokens", 0)
        usage["total_tokens"] = raw_usage.get("total_tokens", usage["input_tokens"] + usage["output_tokens"])
    elif provider == "anthropic":
        usage["input_tokens"] = raw_usage.get("input_tokens", 0)
        usage["output_tokens"] = raw_usage.get("output_tokens", 0)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    elif provider == "gemini":
        meta = data.get("usageMetadata", {}) or raw_usage
        usage["input_tokens"] = meta.get("promptTokenCount", 0)
        usage["output_tokens"] = meta.get("candidatesTokenCount", meta.get("totalTokenCount", 0))
        usage["total_tokens"] = meta.get("totalTokenCount", usage["input_tokens"] + usage["output_tokens"])
        usage["input_cached_tokens"] = meta.get("cachedContentTokenCount", 0)
    else:
        usage["total_tokens"] = raw_usage.get("total_tokens", 0)

    return usage


def validate_tool_args(tool_name: str, tool_args: dict, max_size: int = 10000, max_keys: int = 50) -> dict:
    if tool_args is None:
        raise ValueError(f"Tool '{tool_name}' requires arguments, but got None")

    if not isinstance(tool_args, dict):
        raise ValueError(f"Tool '{tool_name}' arguments must be a JSON object, got {type(tool_args).__name__}")

    try:
        serialized = json.dumps(tool_args)
    except TypeError as e:
        raise ValueError(f"Tool '{tool_name}' arguments must be JSON-serializable: {e}")

    coerced_args: dict = {}
    for key, value in tool_args.items():
        if isinstance(value, str):
            v = value.strip()
            if v.lower() in ("true", "false"):
                coerced_args[key] = v.lower() == "true"
                continue
            try:
                coerced_args[key] = int(v)
                continue
            except ValueError:
                pass
            try:
                coerced_args[key] = float(v)
                continue
            except ValueError:
                pass
            coerced_args[key] = value
        else:
            coerced_args[key] = value

    if tool_name == "agent_end" and "input" in coerced_args and (coerced_args["input"] is None or coerced_args["input"] == ""):
        raise ValueError(f"Tool '{tool_name}' requires a non-empty 'input' argument")

    return coerced_args


def execute_tool(tool_name: str, tool_args: dict, tool_executors: Dict[str, Callable], timeout: float = 900.0, agent_hierarchy: Optional[List[str]] = None, step: int = 0) -> str:
    LOG.debug(f"[TOOL START] Executing tool '{tool_name}'")
    cli = get_cli_output()
    hierarchy = list(agent_hierarchy or []) + [tool_name]
    effective_timeout = _resolve_tool_timeout(tool_name, tool_executors, timeout)

    if tool_name not in tool_executors:
        error_msg = f"Tool '{tool_name}' not found in tool executors"
        LOG.error(error_msg)
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        return json.dumps({"error": error_msg})

    try:
        validated_args = validate_tool_args(tool_name, tool_args)
    except ValueError as e:
        error_msg = f"Validation error for tool '{tool_name}': {str(e)}"
        LOG.warning(error_msg)
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        return json.dumps({"error": error_msg})

    executor_fn = tool_executors[tool_name]
    cli.tool_call(tool_name, tool_args, hierarchy, step)

    future = None
    try:
        executor_pool = _ensure_tool_executor_pool()
        try:
            future = executor_pool.submit(executor_fn, validated_args)
        except RuntimeError:
            # Pool can be shut down by lifecycle races; recreate once and retry.
            executor_pool = _reset_tool_executor_pool()
            future = executor_pool.submit(executor_fn, validated_args)
        result = future.result(timeout=effective_timeout)

        result_str = result if isinstance(result, str) else json.dumps(result)
        cli.tool_result(tool_name, result_str, hierarchy, step)

        LOG.debug(f"[TOOL END] Finished tool '{tool_name}'")
        if isinstance(result, str):
            return result
        return json.dumps(result)
    except FutureTimeoutError:
        if future is not None:
            future.cancel()
        timeout_msg = f"Tool '{tool_name}' execution timed out after {effective_timeout}s"
        cli.tool_result(tool_name, timeout_msg, hierarchy, step, is_timeout=True)
        LOG.warning(timeout_msg)
        return json.dumps({"error": timeout_msg})
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        LOG.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


def format_openai_responses_results(tool_calls: List[dict], tool_results: List[str]) -> List[dict]:
    """Format tool results for the Responses API (role='tool' with tool_call_id).

    The openai_responses payload builder converts these to function_call_output
    items automatically when it processes the messages list.
    """
    tool_messages = []
    for i, tool_call in enumerate(tool_calls):
        tool_call_id = tool_call.get("id", f"call_{i}")
        tool_messages.append({
            "role": "tool",
            "content": tool_results[i] if i < len(tool_results) else json.dumps({"error": "No result"}),
            "tool_call_id": tool_call_id
        })
    return tool_messages


def format_openai_results(tool_calls: List[dict], tool_results: List[str]) -> List[dict]:
    tool_messages = []
    for i, tool_call in enumerate(tool_calls):
        tool_call_id = tool_call.get("id", f"call_{i}")
        tool_messages.append({
            "role": "tool",
            "content": tool_results[i] if i < len(tool_results) else json.dumps({"error": "No result"}),
            "tool_call_id": tool_call_id
        })
    return tool_messages


def format_anthropic_results(tool_calls: List[dict], tool_results: List[str]) -> List[dict]:
    tool_messages = []
    for i, tool_call in enumerate(tool_calls):
        tool_call_id = tool_call.get("id", f"call_{i}")
        tool_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_results[i] if i < len(tool_results) else json.dumps({"error": "No result"})
                }
            ]
        })
    return tool_messages


def format_gemini_results(tool_calls: List[dict], tool_results: List[str]) -> List[dict]:
    function_responses = []
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name", "")
        try:
            result_data = json.loads(tool_results[i]) if i < len(tool_results) else {"error": "No result"}
        except Exception:
            result_data = {"result": tool_results[i]} if i < len(tool_results) else {"error": "No result"}
        if not isinstance(result_data, dict):
            result_data = {"result": result_data}
        function_responses.append({
            "functionResponse": {
                "name": tool_name,
                "response": result_data
            }
        })
    return function_responses


def _build_empty_args_error(tool_name: str, tool_metadata: dict, tool_executors: dict) -> Optional[str]:
    """If a tool has required params, return an error message with schema hints.
    Returns None if the tool has no required params (empty args are OK)."""
    _schema = tool_metadata.get(tool_name, {}).get("parameters") or {}
    _required = _schema.get("required", [])
    _props = _schema.get("properties", {})
    # Fallback: check __tool_schema__ on executor
    if not _required and not _props:
        _exec = tool_executors.get(tool_name)
        if _exec and hasattr(_exec, "__tool_schema__"):
            _schema = _exec.__tool_schema__
            _required = _schema.get("required", [])
            _props = _schema.get("properties", {})
    if not _required and not _props:
        return None  # No schema info — let the tool handle it
    param_hints = []
    for pname, pdef in _props.items():
        if pname == "__required__":
            continue
        req_marker = " (REQUIRED)" if pname in _required else ""
        ptype = pdef.get("type", "string") if isinstance(pdef, dict) else "string"
        pdesc = pdef.get("description", "") if isinstance(pdef, dict) else ""
        param_hints.append(f'  "{pname}": <{ptype}>{req_marker} — {pdesc}')
    schema_hint = "\n".join(param_hints)
    return json.dumps({
        "error": f"Tool '{tool_name}' was called with empty arguments {{}}. "
                 f"You MUST provide the required parameters. Expected schema:\n{schema_hint}"
    })


def _get_tool_name(tool_call: dict) -> str:
    fn = tool_call.get("function", {})
    return fn.get("name") or tool_call.get("name", "")


def _format_results_for_provider(provider: str, tool_call_order: List[dict], tool_results: List[str]) -> List[dict]:
    if provider in ("deepseek", "openai", "openrouter"):
        return format_openai_results(tool_call_order, tool_results)
    if provider == "openai_responses":
        return format_openai_responses_results(tool_call_order, tool_results)
    if provider == "anthropic":
        return format_anthropic_results(tool_call_order, tool_results)
    if provider == "gemini":
        return format_gemini_results(tool_call_order, tool_results)
    return []


def _reject_mixed_agent_end_batch(
    tool_calls: List[dict],
    provider: str,
    tool_call_counts: Dict[str, int],
) -> tuple[List[dict], List[str], Dict[str, int], List[dict]]:
    tool_names = [_get_tool_name(tool_call) for tool_call in tool_calls]
    batch_summary = ", ".join(name or "<unknown>" for name in tool_names)
    shared_reason = (
        "Batched tool request rejected: 'agent_end' must be sent by itself with no other tool calls in the same response. "
        "Do not batch 'agent_end' with any other tools. If you still need tool outputs, re-send those tools without "
        "'agent_end'. After you review their results, send a new response containing only 'agent_end'."
    )

    rejected_tool_calls = []
    tool_results = []
    for tool_call in tool_calls:
        tool_name = _get_tool_name(tool_call)
        rejected_tool_call = deepcopy(tool_call)
        rejected_tool_call["_batch_rejected"] = True
        rejected_tool_call["_batch_rejected_reason"] = "mixed_agent_end"
        rejected_tool_calls.append(rejected_tool_call)
        tool_results.append(json.dumps({
            "error": shared_reason,
            "tool_name": tool_name,
            "batched_tools": tool_names,
            "batch_summary": batch_summary,
        }))

    LOG.warning("Rejected mixed agent_end batch: %s", batch_summary)
    formatted_messages = _format_results_for_provider(provider, rejected_tool_calls, tool_results)
    return formatted_messages, tool_results, tool_call_counts, rejected_tool_calls


def execute_tool_calls(
    tool_calls: List[dict],
    tool_executors: Dict[str, Callable],
    provider: str,
    timeout: float = 900.0,
    tool_metadata: Optional[Dict[str, dict]] = None,
    agent_hierarchy: Optional[List[str]] = None,
    step: int = 0,
    tool_call_counts: Optional[Dict[str, int]] = None
) -> tuple[List[dict], List[str], Dict[str, int], List[dict]]:
    if not tool_calls or not tool_executors:
        return [], [], tool_call_counts or {}, []

    tool_metadata = tool_metadata or {}
    tool_call_counts = tool_call_counts or {}
    tool_names = [_get_tool_name(tool_call) for tool_call in tool_calls]
    if "agent_end" in tool_names and len(tool_calls) > 1:
        return _reject_mixed_agent_end_batch(tool_calls, provider, tool_call_counts)

    tool_call_order = tool_calls.copy()
    tool_call_id_to_result: Dict[str, str] = {}

    parallel_calls = []
    sequential_calls = []
    seen_tool_signatures = set()

    for idx, tool_call in enumerate(tool_calls):
        fn = tool_call.get("function", {})
        tool_name = fn.get("name") or tool_call.get("name", "")
        args_raw = fn.get("arguments") or "{}"
        tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"

        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception as e:
            LOG.warning(f"Failed to parse tool arguments for {tool_name}: {e}")
            error_msg = json.dumps({
                "error": f"Malformed JSON in arguments for tool '{tool_name}': {e}. "
                         f"Fix the JSON syntax and retry. Raw arguments were: {args_raw[:200]}"
            })
            tool_call_id_to_result[tool_call_id] = error_msg
            continue

        # Detect empty args when tool has required parameters — give the model
        # a clear schema hint so it knows what to provide on retry.
        if (not args or args == {}) and tool_name != "agent_end":
            empty_err = _build_empty_args_error(tool_name, tool_metadata, tool_executors)
            if empty_err:
                LOG.warning(f"Empty args for tool '{tool_name}', returning schema hint")
                tool_call_id_to_result[tool_call_id] = empty_err
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                continue

        tool_signature = (tool_name, json.dumps(args, sort_keys=True))
        if tool_signature in seen_tool_signatures:
            LOG.warning(f"Duplicate tool call detected: {tool_name} with same arguments. Skipping duplicate.")
            error_msg = json.dumps({"error": f"Duplicate tool call for '{tool_name}' detected. Only executing once."})
            tool_call_id_to_result[tool_call_id] = error_msg
            continue
        seen_tool_signatures.add(tool_signature)

        metadata = tool_metadata.get(tool_name, {})
        limit_calls = metadata.get("limit_calls", 0)
        current_count = tool_call_counts.get(tool_name, 0)

        if limit_calls > 0 and current_count >= limit_calls:
            LOG.warning(f"Tool '{tool_name}' has reached its call limit ({limit_calls}). Skipping this call.")
            error_msg = json.dumps({"error": f"Tool '{tool_name}' call limit ({limit_calls}) reached. Skipping execution."})
            tool_call_id_to_result[tool_call_id] = error_msg
            continue

        is_parallel = metadata.get("parallel", True)

        if is_parallel:
            parallel_calls.append((tool_name, args, tool_call_id))
        else:
            sequential_calls.append((tool_name, args, tool_call_id))

    if parallel_calls:
        LOG.debug(f"[TOOL PARALLEL] Starting {len(parallel_calls)} parallel tools: {[t[0] for t in parallel_calls]}")
        executor_pool = ThreadPoolExecutor(max_workers=len(parallel_calls))
        try:
            futures = {}
            for tool_name, args, tool_call_id in parallel_calls:
                future = executor_pool.submit(execute_tool, tool_name, args, tool_executors, timeout, agent_hierarchy, step)
                effective_timeout = _resolve_tool_timeout(tool_name, tool_executors, timeout)
                futures[future] = (tool_name, tool_call_id, effective_timeout)

            for future in futures:
                tool_name, tool_call_id, effective_timeout = futures[future]
                try:
                    # Guard against wrapper-level hangs as well.
                    result = future.result(timeout=effective_timeout + 5.0)
                    tool_call_id_to_result[tool_call_id] = result
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                except FutureTimeoutError:
                    future.cancel()
                    error_msg = f"Parallel tool execution timed out after {effective_timeout + 5.0}s for '{tool_name}'"
                    LOG.warning(error_msg)
                    tool_call_id_to_result[tool_call_id] = json.dumps({"error": error_msg})
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                except Exception as e:
                    error_msg = f"Parallel tool execution error: {str(e)}"
                    LOG.error(error_msg)
                    tool_call_id_to_result[tool_call_id] = json.dumps({"error": error_msg})
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
        finally:
            executor_pool.shutdown(wait=False, cancel_futures=True)

    for tool_name, args, tool_call_id in sequential_calls:
        result = execute_tool(tool_name, args, tool_executors, timeout, agent_hierarchy, step)
        tool_call_id_to_result[tool_call_id] = result
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

    tool_results = []
    for idx, tool_call in enumerate(tool_call_order):
        fn = tool_call.get("function", {})
        tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"
        tool_results.append(tool_call_id_to_result.get(tool_call_id, json.dumps({"error": "No result"})))

    formatted_messages = _format_results_for_provider(provider, tool_call_order, tool_results)

    return formatted_messages, tool_results, tool_call_counts, tool_call_order


async def async_execute_tool(
    tool_name: str,
    tool_args: dict,
    tool_executors: Dict[str, Callable],
    timeout: float = 900.0,
    agent_hierarchy: Optional[List[str]] = None,
    step: int = 0
) -> str:
    LOG.debug(f"[TOOL START] Executing tool '{tool_name}' (async)")
    cli = get_cli_output()
    hierarchy = list(agent_hierarchy or []) + [tool_name]
    effective_timeout = _resolve_tool_timeout(tool_name, tool_executors, timeout)

    if tool_name not in tool_executors:
        error_msg = f"Tool '{tool_name}' not found in tool executors"
        LOG.error(error_msg)
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        return json.dumps({"error": error_msg})

    try:
        validated_args = validate_tool_args(tool_name, tool_args)
    except ValueError as e:
        error_msg = f"Validation error for tool '{tool_name}': {str(e)}"
        LOG.warning(error_msg)
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        return json.dumps({"error": error_msg})

    executor_fn = tool_executors[tool_name]
    cli.tool_call(tool_name, tool_args, hierarchy, step)

    try:
        loop = asyncio.get_event_loop()
        if asyncio.iscoroutinefunction(executor_fn):
            result = await asyncio.wait_for(executor_fn(validated_args), timeout=effective_timeout)
        else:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, executor_fn, validated_args),
                timeout=effective_timeout
            )

        result_str = result if isinstance(result, str) else json.dumps(result)
        cli.tool_result(tool_name, result_str, hierarchy, step)

        LOG.debug(f"[TOOL END] Finished tool '{tool_name}' (async)")
        if isinstance(result, str):
            return result
        return json.dumps(result)
    except asyncio.TimeoutError:
        timeout_msg = f"Tool '{tool_name}' execution timed out after {effective_timeout}s"
        cli.tool_result(tool_name, timeout_msg, hierarchy, step, is_timeout=True)
        LOG.warning(timeout_msg)
        return json.dumps({"error": timeout_msg})
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        LOG.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


async def async_execute_tool_calls(
    tool_calls: List[dict],
    tool_executors: Dict[str, Callable],
    provider: str,
    timeout: float = 900.0,
    tool_metadata: Optional[Dict[str, dict]] = None,
    agent_hierarchy: Optional[List[str]] = None,
    step: int = 0,
    tool_call_counts: Optional[Dict[str, int]] = None
) -> tuple[List[dict], List[str], Dict[str, int], List[dict]]:
    if not tool_calls or not tool_executors:
        return [], [], tool_call_counts or {}, []

    tool_metadata = tool_metadata or {}
    tool_call_counts = tool_call_counts or {}
    tool_names = [_get_tool_name(tool_call) for tool_call in tool_calls]
    if "agent_end" in tool_names and len(tool_calls) > 1:
        return _reject_mixed_agent_end_batch(tool_calls, provider, tool_call_counts)

    tool_call_order = tool_calls.copy()
    tool_call_id_to_result: Dict[str, str] = {}

    parallel_calls = []
    sequential_calls = []
    seen_tool_signatures = set()

    for idx, tool_call in enumerate(tool_calls):
        fn = tool_call.get("function", {})
        tool_name = fn.get("name") or tool_call.get("name", "")
        args_raw = fn.get("arguments") or "{}"
        tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"

        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception as e:
            LOG.warning(f"Failed to parse tool arguments for {tool_name}: {e}")
            # Skip execution — return an explicit error so the model knows exactly
            # what went wrong instead of running the tool with empty args (which
            # produces a confusing validation error and causes GPT to spiral).
            error_msg = json.dumps({
                "error": f"Malformed JSON in arguments for tool '{tool_name}': {e}. "
                         f"Fix the JSON syntax and retry. Raw arguments were: {args_raw[:200]}"
            })
            tool_call_id_to_result[tool_call_id] = error_msg
            continue

        tool_signature = (tool_name, json.dumps(args, sort_keys=True))
        if tool_signature in seen_tool_signatures:
            LOG.warning(f"Duplicate tool call detected: {tool_name} with same arguments. Skipping duplicate.")
            error_msg = json.dumps({"error": f"Duplicate tool call for '{tool_name}' detected. Only executing once."})
            tool_call_id_to_result[tool_call_id] = error_msg
            continue
        seen_tool_signatures.add(tool_signature)

        metadata = tool_metadata.get(tool_name, {})
        limit_calls = metadata.get("limit_calls", 0)
        current_count = tool_call_counts.get(tool_name, 0)

        if limit_calls > 0 and current_count >= limit_calls:
            LOG.warning(f"Tool '{tool_name}' has reached its call limit ({limit_calls}). Skipping this call.")
            error_msg = json.dumps({"error": f"Tool '{tool_name}' call limit ({limit_calls}) reached. Skipping execution."})
            tool_call_id_to_result[tool_call_id] = error_msg
            continue

        is_parallel = metadata.get("parallel", True)

        if is_parallel:
            parallel_calls.append((tool_name, args, tool_call_id))
        else:
            sequential_calls.append((tool_name, args, tool_call_id))

    if parallel_calls:
        tasks = []
        tool_call_ids_for_parallel = []
        for tool_name, args, tool_call_id in parallel_calls:
            task = async_execute_tool(tool_name, args, tool_executors, timeout, agent_hierarchy, step)
            tasks.append(task)
            tool_call_ids_for_parallel.append(tool_call_id)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            tool_call_id = tool_call_ids_for_parallel[i]
            tool_name = parallel_calls[i][0] if i < len(parallel_calls) else ""
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            if isinstance(result, Exception):
                error_msg = f"Parallel tool execution error: {str(result)}"
                LOG.error(error_msg)
                tool_call_id_to_result[tool_call_id] = json.dumps({"error": error_msg})
            else:
                tool_call_id_to_result[tool_call_id] = result if isinstance(result, str) else json.dumps(result)

    for tool_name, args, tool_call_id in sequential_calls:
        result = await async_execute_tool(tool_name, args, tool_executors, timeout, agent_hierarchy, step)
        tool_call_id_to_result[tool_call_id] = result if isinstance(result, str) else json.dumps(result)
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

    tool_results = []
    for idx, tool_call in enumerate(tool_call_order):
        fn = tool_call.get("function", {})
        tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"
        tool_results.append(tool_call_id_to_result.get(tool_call_id, json.dumps({"error": "No result"})))

    formatted_messages = _format_results_for_provider(provider, tool_call_order, tool_results)

    return formatted_messages, tool_results, tool_call_counts, tool_call_order
