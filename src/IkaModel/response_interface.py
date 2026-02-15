import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional, Callable

from IkaCore.cli_output import get_cli_output

LOG = logging.getLogger(__name__)


def extract_usage(provider: str, data: dict) -> Dict[str, Any]:
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "input_cached_tokens": 0}
    raw_usage = data.get("usage", {}) or {}

    if provider in ["deepseek", "openai", "openrouter"]:
        usage["input_tokens"] = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
        usage["output_tokens"] = raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0))
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
        with ThreadPoolExecutor(max_workers=1) as executor_pool:
            future = executor_pool.submit(executor_fn, validated_args)
            result = future.result(timeout=timeout)

            result_str = result if isinstance(result, str) else json.dumps(result)
            cli.tool_result(tool_name, result_str, hierarchy, step)

        LOG.debug(f"[TOOL END] Finished tool '{tool_name}'")
        if isinstance(result, str):
            return result
        return json.dumps(result)
    except FutureTimeoutError:
        timeout_msg = f"Tool '{tool_name}' execution timed out after {timeout}s"
        cli.tool_result(tool_name, timeout_msg, hierarchy, step, is_timeout=True)
        LOG.warning(timeout_msg)
        return json.dumps({"error": timeout_msg})
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        LOG.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


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
        LOG.debug(f"[TOOL PARALLEL] Starting {len(parallel_calls)} parallel tools: {[t[0] for t in parallel_calls]}")
        with ThreadPoolExecutor(max_workers=len(parallel_calls)) as executor_pool:
            futures = {}
            for tool_name, args, tool_call_id in parallel_calls:
                future = executor_pool.submit(execute_tool, tool_name, args, tool_executors, timeout, agent_hierarchy, step)
                futures[future] = (tool_name, tool_call_id)

            for future in futures:
                tool_name, tool_call_id = futures[future]
                try:
                    result = future.result()
                    tool_call_id_to_result[tool_call_id] = result
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                except Exception as e:
                    error_msg = f"Parallel tool execution error: {str(e)}"
                    LOG.error(error_msg)
                    tool_call_id_to_result[tool_call_id] = json.dumps({"error": error_msg})
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

    for tool_name, args, tool_call_id in sequential_calls:
        result = execute_tool(tool_name, args, tool_executors, timeout, agent_hierarchy, step)
        tool_call_id_to_result[tool_call_id] = result
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

    tool_results = []
    for idx, tool_call in enumerate(tool_call_order):
        fn = tool_call.get("function", {})
        tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"
        tool_results.append(tool_call_id_to_result.get(tool_call_id, json.dumps({"error": "No result"})))

    if provider in ("deepseek", "openai", "openrouter"):
        formatted_messages = format_openai_results(tool_call_order, tool_results)
    elif provider == "anthropic":
        formatted_messages = format_anthropic_results(tool_call_order, tool_results)
    elif provider == "gemini":
        formatted_messages = format_gemini_results(tool_call_order, tool_results)
    else:
        formatted_messages = []

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
            result = await asyncio.wait_for(executor_fn(validated_args), timeout=timeout)
        else:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, executor_fn, validated_args),
                timeout=timeout
            )

        result_str = result if isinstance(result, str) else json.dumps(result)
        cli.tool_result(tool_name, result_str, hierarchy, step)

        LOG.debug(f"[TOOL END] Finished tool '{tool_name}' (async)")
        if isinstance(result, str):
            return result
        return json.dumps(result)
    except asyncio.TimeoutError:
        timeout_msg = f"Tool '{tool_name}' execution timed out after {timeout}s"
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

    if provider in ("deepseek", "openai", "openrouter"):
        formatted_messages = format_openai_results(tool_call_order, tool_results)
    elif provider == "anthropic":
        formatted_messages = format_anthropic_results(tool_call_order, tool_results)
    elif provider == "gemini":
        formatted_messages = format_gemini_results(tool_call_order, tool_results)
    else:
        formatted_messages = []

    return formatted_messages, tool_results, tool_call_counts, tool_call_order
