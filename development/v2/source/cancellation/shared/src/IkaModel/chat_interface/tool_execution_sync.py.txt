# pyright: strict

"""Synchronous tool execution and planning helpers."""

from __future__ import annotations

import json
import logging
import os
import threading
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict
from IkaCore.cli_output import get_cli_output
from IkaCore.tools import ToolExecutor, ToolParameters

from ..base import AgentEndException, HumanInputRequired
from .tool_result_formatters import format_provider_tool_results

LOG = logging.getLogger(__name__)
def _tool_executor_max_workers() -> int:
    raw = os.environ.get("IKA_TOOL_EXECUTOR_MAX_WORKERS", "32")
    try:
        return max(8, int(raw))
    except (TypeError, ValueError):
        return 32


_tool_executor_pool = ThreadPoolExecutor(
    max_workers=_tool_executor_max_workers(),
    thread_name_prefix="tool-exec",
)
_TOOL_EXECUTOR_LOCK = threading.Lock()
ToolExecutorMap = dict[str, ToolExecutor]
ToolMetadata = dict[str, JsonDict]
ScheduledToolCall = tuple[str, object, str]
ToolCallBatchResult = tuple[list[JsonDict], list[str], dict[str, int], list[JsonDict], Optional[JsonDict]]


@dataclass
class ToolCallPlan:
    tool_call_order: list[JsonDict]
    tool_call_id_to_result: dict[str, str] = field(default_factory=lambda: {})
    parallel_calls: list[ScheduledToolCall] = field(default_factory=lambda: [])
    sequential_calls: list[ScheduledToolCall] = field(default_factory=lambda: [])


def _ensure_tool_executor_pool() -> ThreadPoolExecutor:
    """Ensure the global tool executor is alive before scheduling work."""
    global _tool_executor_pool
    with _TOOL_EXECUTOR_LOCK:
        is_shutdown = getattr(_tool_executor_pool, "_shutdown", False)
        if is_shutdown:
            _tool_executor_pool = ThreadPoolExecutor(
                max_workers=_tool_executor_max_workers(),
                thread_name_prefix="tool-exec",
            )
    return _tool_executor_pool


def _reset_tool_executor_pool() -> ThreadPoolExecutor:
    """Force-create a fresh executor pool after a scheduling race."""
    global _tool_executor_pool
    old_pool: ThreadPoolExecutor | None = None
    with _TOOL_EXECUTOR_LOCK:
        old_pool = _tool_executor_pool
        _tool_executor_pool = ThreadPoolExecutor(
            max_workers=_tool_executor_max_workers(),
            thread_name_prefix="tool-exec",
        )
        new_pool = _tool_executor_pool
    if old_pool is not None:
        old_pool.shutdown(wait=False, cancel_futures=True)
    return new_pool


def validate_tool_args(
    tool_name: str,
    tool_args: object,
    max_size: int = 10000,
    max_keys: int = 50,
) -> ToolParameters:
    if tool_args is None:
        raise ValueError(f"Tool '{tool_name}' requires arguments, but got None")

    if not isinstance(tool_args, dict):
        raise ValueError(f"Tool '{tool_name}' arguments must be a JSON object, got {type(tool_args).__name__}")

    try:
        json.dumps(tool_args)
    except TypeError as e:
        raise ValueError(f"Tool '{tool_name}' arguments must be JSON-serializable: {e}")

    raw_args = cast(JsonDict, tool_args)
    coerced_args: ToolParameters = {}
    for key, value in raw_args.items():
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


def execute_tool(
    tool_name: str,
    tool_args: object,
    tool_executors: ToolExecutorMap,
    timeout: float = 900.0,
    agent_hierarchy: Optional[list[str]] = None,
    step: int = 0,
) -> str:
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
    cli.tool_call(tool_name, validated_args, hierarchy, step)

    future = None
    try:
        executor_pool = _ensure_tool_executor_pool()
        try:
            ctx = copy_context()
            future = executor_pool.submit(ctx.run, executor_fn, validated_args)
        except RuntimeError:
            # Pool can be shut down by lifecycle races; recreate once and retry.
            executor_pool = _reset_tool_executor_pool()
            ctx = copy_context()
            future = executor_pool.submit(ctx.run, executor_fn, validated_args)
        result: object = future.result(timeout=timeout)

        result_str = result if isinstance(result, str) else json.dumps(result)
        cli.tool_result(tool_name, result_str, hierarchy, step)

        LOG.debug(f"[TOOL END] Finished tool '{tool_name}'")
        if isinstance(result, str):
            return result
        return json.dumps(result)
    except AgentEndException:
        raise
    except HumanInputRequired:
        raise
    except FutureTimeoutError:
        if future is not None:
            future.cancel()
        _reset_tool_executor_pool()
        timeout_msg = f"Tool '{tool_name}' execution timed out after {timeout}s"
        cli.tool_result(tool_name, timeout_msg, hierarchy, step, is_timeout=True)
        LOG.warning(timeout_msg)
        return json.dumps({"error": timeout_msg})
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        cli.tool_result(tool_name, error_msg, hierarchy, step, is_error=True)
        LOG.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


def _schema_dict(value: object) -> JsonDict:
    return cast(JsonDict, value) if isinstance(value, dict) else {}


def _schema_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in cast(list[object], value) if isinstance(item, str)]


def _build_empty_args_error(
    tool_name: str,
    tool_metadata: ToolMetadata,
    tool_executors: ToolExecutorMap,
) -> Optional[str]:
    """If a tool has required params, return an error message with schema hints.
    Returns None if the tool has no required params (empty args are OK)."""
    _schema = _schema_dict(tool_metadata.get(tool_name, {}).get("parameters"))
    _required = _schema_list(_schema.get("required"))
    _props = _schema_dict(_schema.get("properties"))
    # Fallback: check __tool_schema__ on executor
    if not _required and not _props:
        _exec = tool_executors.get(tool_name)
        if _exec and hasattr(_exec, "__tool_schema__"):
            _schema = _schema_dict(getattr(_exec, "__tool_schema__"))
            _required = _schema_list(_schema.get("required"))
            _props = _schema_dict(_schema.get("properties"))
    if not _required and not _props:
        return None  # No schema info — let the tool handle it
    param_hints: list[str] = []
    for pname, pdef in _props.items():
        if pname == "__required__":
            continue
        req_marker = " (REQUIRED)" if pname in _required else ""
        param_def = _schema_dict(pdef)
        raw_type = param_def.get("type", "string")
        raw_desc = param_def.get("description", "")
        ptype = raw_type if isinstance(raw_type, str) else "string"
        pdesc = raw_desc if isinstance(raw_desc, str) else ""
        param_hints.append(f'  "{pname}": <{ptype}>{req_marker} — {pdesc}')
    schema_hint = "\n".join(param_hints)
    return json.dumps({
        "error": f"Tool '{tool_name}' was called with empty arguments {{}}. "
                 f"You MUST provide the required parameters. Expected schema:\n{schema_hint}"
    })


def _handle_empty_required_args(
    tool_name: str,
    args: Any,
    tool_metadata: ToolMetadata,
    tool_executors: ToolExecutorMap,
    tool_call_counts: dict[str, int],
) -> Optional[str]:
    if (not args or args == {}) and tool_name != "agent_end":
        empty_err = _build_empty_args_error(tool_name, tool_metadata, tool_executors)
        if empty_err:
            LOG.warning(f"Empty args for tool '{tool_name}', returning schema hint")
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            return empty_err
    return None


def _decode_tool_arguments(tool_name: str, args_raw: object) -> tuple[object, Optional[str]]:
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    except (json.JSONDecodeError, TypeError) as e:
        LOG.warning(f"Failed to parse tool arguments for {tool_name}: {e}")
        return None, json.dumps({
            "error": f"Malformed JSON in arguments for tool '{tool_name}': {e}. "
                     f"Fix the JSON syntax and retry. Raw arguments were: {str(args_raw)[:200]}"
        })
    return args, None


def _schedule_tool_call(
    plan: ToolCallPlan,
    tool_name: str,
    args: object,
    tool_call_id: str,
    metadata: JsonDict,
) -> None:
    is_parallel = bool(metadata.get("parallel", True))
    if is_parallel:
        plan.parallel_calls.append((tool_name, args, tool_call_id))
    else:
        plan.sequential_calls.append((tool_name, args, tool_call_id))


def _tool_function_payload(tool_call: JsonDict) -> JsonDict:
    return _schema_dict(tool_call.get("function"))


def _prepare_tool_call_plan(
    tool_calls: list[JsonDict],
    tool_executors: ToolExecutorMap,
    tool_metadata: ToolMetadata,
    tool_call_counts: dict[str, int],
) -> ToolCallPlan:
    plan = ToolCallPlan(tool_call_order=tool_calls.copy())
    seen_tool_signatures: set[tuple[str, str]] = set()

    for idx, tool_call in enumerate(tool_calls):
        fn = _tool_function_payload(tool_call)
        raw_name = fn.get("name") or tool_call.get("name", "")
        tool_name = raw_name if isinstance(raw_name, str) else str(raw_name)
        args_raw = fn.get("arguments") or "{}"
        raw_tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"
        tool_call_id = raw_tool_call_id if isinstance(raw_tool_call_id, str) else str(raw_tool_call_id)

        args, error_msg = _decode_tool_arguments(tool_name, args_raw)
        if error_msg:
            plan.tool_call_id_to_result[tool_call_id] = error_msg
            continue

        empty_err = _handle_empty_required_args(
            tool_name, args, tool_metadata, tool_executors, tool_call_counts
        )
        if empty_err:
            plan.tool_call_id_to_result[tool_call_id] = empty_err
            continue

        tool_signature = (tool_name, json.dumps(args, sort_keys=True))
        if tool_signature in seen_tool_signatures:
            LOG.warning(f"Duplicate tool call detected: {tool_name} with same arguments. Skipping duplicate.")
            error_msg = json.dumps({"error": f"Duplicate tool call for '{tool_name}' detected. Only executing once."})
            plan.tool_call_id_to_result[tool_call_id] = error_msg
            continue
        seen_tool_signatures.add(tool_signature)

        metadata = tool_metadata.get(tool_name, {})
        raw_limit_calls = metadata.get("limit_calls", 0)
        limit_calls = raw_limit_calls if isinstance(raw_limit_calls, int) else 0
        current_count = tool_call_counts.get(tool_name, 0)

        if limit_calls > 0 and current_count >= limit_calls:
            LOG.warning(f"Tool '{tool_name}' has reached its call limit ({limit_calls}). Skipping this call.")
            error_msg = json.dumps({"error": f"Tool '{tool_name}' call limit ({limit_calls}) reached. Skipping execution."})
            plan.tool_call_id_to_result[tool_call_id] = error_msg
            continue

        _schedule_tool_call(plan, tool_name, args, tool_call_id, metadata)
    return plan


prepare_tool_call_plan = _prepare_tool_call_plan


def _execute_parallel_tool_plan(
    plan: ToolCallPlan,
    tool_executors: ToolExecutorMap,
    timeout: float,
    agent_hierarchy: Optional[list[str]],
    step: int,
    tool_call_counts: dict[str, int],
) -> Optional[JsonDict]:
    if not plan.parallel_calls:
        return None

    LOG.debug(f"[TOOL PARALLEL] Starting {len(plan.parallel_calls)} parallel tools: {[t[0] for t in plan.parallel_calls]}")
    executor_pool = ThreadPoolExecutor(max_workers=len(plan.parallel_calls))
    interrupt_data: Optional[JsonDict] = None
    try:
        futures: dict[Any, tuple[str, str]] = {}
        for tool_name, args, tool_call_id in plan.parallel_calls:
            ctx = copy_context()
            future = executor_pool.submit(
                ctx.run,
                execute_tool,
                tool_name,
                args,
                tool_executors,
                timeout,
                agent_hierarchy,
                step,
            )
            futures[future] = (tool_name, tool_call_id)

        pending = set(futures)
        wrapper_timeout = timeout + 5.0
        try:
            for future in as_completed(pending, timeout=wrapper_timeout):
                pending.discard(future)
                tool_name, tool_call_id = futures[future]
                try:
                    result = future.result()
                    plan.tool_call_id_to_result[tool_call_id] = result
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                except AgentEndException:
                    raise
                except HumanInputRequired as exc:
                    interrupt_data = exc.payload
                    plan.tool_call_id_to_result[tool_call_id] = json.dumps({"__ika_interrupt__": True, **(exc.payload or {})})
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                    for remaining in pending:
                        remaining.cancel()
                    break
                except Exception as e:
                    error_msg = f"Parallel tool execution error: {str(e)}"
                    LOG.error(error_msg)
                    plan.tool_call_id_to_result[tool_call_id] = json.dumps({"error": error_msg})
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
        except FutureTimeoutError:
            _reset_tool_executor_pool()
            timeout_label = f"{wrapper_timeout}s"
            for future in pending:
                tool_name, tool_call_id = futures[future]
                future.cancel()
                error_msg = f"Parallel tool execution timed out after {timeout_label} for '{tool_name}'"
                LOG.warning(error_msg)
                plan.tool_call_id_to_result[tool_call_id] = json.dumps({"error": error_msg})
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
    finally:
        executor_pool.shutdown(wait=False, cancel_futures=True)
    return interrupt_data


def _execute_sequential_tool_plan(
    plan: ToolCallPlan,
    tool_executors: ToolExecutorMap,
    timeout: float,
    agent_hierarchy: Optional[list[str]],
    step: int,
    tool_call_counts: dict[str, int],
) -> Optional[JsonDict]:
    interrupt_data: Optional[JsonDict] = None
    for tool_name, args, tool_call_id in plan.sequential_calls:
        try:
            result = execute_tool(tool_name, args, tool_executors, timeout, agent_hierarchy, step)
        except HumanInputRequired as exc:
            interrupt_data = exc.payload
            result = json.dumps({"__ika_interrupt__": True, **(exc.payload or {})})
            plan.tool_call_id_to_result[tool_call_id] = result
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            break
        plan.tool_call_id_to_result[tool_call_id] = result
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
    return interrupt_data


def _ordered_tool_results(plan: ToolCallPlan) -> list[str]:
    tool_results: list[str] = []
    for idx, tool_call in enumerate(plan.tool_call_order):
        fn = _tool_function_payload(tool_call)
        raw_tool_call_id = tool_call.get("id") or fn.get("id") or f"call_{idx}"
        tool_call_id = raw_tool_call_id if isinstance(raw_tool_call_id, str) else str(raw_tool_call_id)
        tool_results.append(plan.tool_call_id_to_result.get(tool_call_id, json.dumps({"error": "No result"})))
    return tool_results


def _finalize_tool_call_plan(
    provider: str,
    plan: ToolCallPlan,
    tool_call_counts: dict[str, int],
    interrupt_data: Optional[JsonDict],
) -> ToolCallBatchResult:
    tool_results = _ordered_tool_results(plan)
    formatted_messages = format_provider_tool_results(provider, plan.tool_call_order, tool_results)
    return formatted_messages, tool_results, tool_call_counts, plan.tool_call_order, interrupt_data


finalize_tool_call_plan = _finalize_tool_call_plan


def execute_tool_calls(
    tool_calls: list[JsonDict],
    tool_executors: ToolExecutorMap,
    provider: str,
    timeout: float = 900.0,
    tool_metadata: Optional[ToolMetadata] = None,
    agent_hierarchy: Optional[list[str]] = None,
    step: int = 0,
    tool_call_counts: Optional[dict[str, int]] = None,
) -> ToolCallBatchResult:
    if not tool_calls or not tool_executors:
        return [], [], tool_call_counts or {}, [], None

    tool_metadata = tool_metadata or {}
    tool_call_counts = tool_call_counts or {}
    plan = prepare_tool_call_plan(tool_calls, tool_executors, tool_metadata, tool_call_counts)
    interrupt_data = _execute_parallel_tool_plan(
        plan,
        tool_executors,
        timeout,
        agent_hierarchy,
        step,
        tool_call_counts,
    )
    if not interrupt_data:
        interrupt_data = _execute_sequential_tool_plan(
            plan,
            tool_executors,
            timeout,
            agent_hierarchy,
            step,
            tool_call_counts,
        )
    return finalize_tool_call_plan(provider, plan, tool_call_counts, interrupt_data)
