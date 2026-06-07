# pyright: strict

"""Asynchronous tool execution helpers."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable
from typing import Optional

from IkaCore.agent_runtime_payloads import JsonDict
from IkaCore.cli_output import get_cli_output

from ..base import AgentEndException, HumanInputRequired
from .tool_execution_sync import (
    ToolCallBatchResult,
    ToolCallPlan,
    ToolExecutorMap,
    ToolMetadata,
    finalize_tool_call_plan,
    prepare_tool_call_plan,
    validate_tool_args,
)

LOG = logging.getLogger(__name__)

async def async_execute_tool(
    tool_name: str,
    tool_args: object,
    tool_executors: ToolExecutorMap,
    timeout: float = 900.0,
    agent_hierarchy: Optional[list[str]] = None,
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
    cli.tool_call(tool_name, validated_args, hierarchy, step)

    try:
        loop = asyncio.get_event_loop()
        if inspect.iscoroutinefunction(executor_fn):
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
    except AgentEndException:
        raise
    except HumanInputRequired:
        raise
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


async def _execute_parallel_tool_plan_async(
    plan: ToolCallPlan,
    tool_executors: ToolExecutorMap,
    timeout: float,
    agent_hierarchy: Optional[list[str]],
    step: int,
    tool_call_counts: dict[str, int],
) -> Optional[JsonDict]:
    if not plan.parallel_calls:
        return None

    tasks: list[Awaitable[str]] = []
    tool_call_ids_for_parallel: list[str] = []
    for tool_name, args, tool_call_id in plan.parallel_calls:
        task = async_execute_tool(tool_name, args, tool_executors, timeout, agent_hierarchy, step)
        tasks.append(task)
        tool_call_ids_for_parallel.append(tool_call_id)

    interrupt_data: Optional[JsonDict] = None
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        tool_call_id = tool_call_ids_for_parallel[i]
        tool_name = plan.parallel_calls[i][0] if i < len(plan.parallel_calls) else ""
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
        if isinstance(result, Exception):
            if isinstance(result, AgentEndException):
                raise result
            if isinstance(result, HumanInputRequired):
                interrupt_data = result.payload
                plan.tool_call_id_to_result[tool_call_id] = json.dumps({"__ika_interrupt__": True, **(result.payload or {})})
                break
            error_msg = f"Parallel tool execution error: {str(result)}"
            LOG.error(error_msg)
            plan.tool_call_id_to_result[tool_call_id] = json.dumps({"error": error_msg})
        else:
            plan.tool_call_id_to_result[tool_call_id] = result if isinstance(result, str) else json.dumps(result)
    return interrupt_data


async def _execute_sequential_tool_plan_async(
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
            result = await async_execute_tool(tool_name, args, tool_executors, timeout, agent_hierarchy, step)
        except HumanInputRequired as exc:
            interrupt_data = exc.payload
            plan.tool_call_id_to_result[tool_call_id] = json.dumps({"__ika_interrupt__": True, **(exc.payload or {})})
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            break
        plan.tool_call_id_to_result[tool_call_id] = result
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
    return interrupt_data


async def async_execute_tool_calls(
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
    interrupt_data = await _execute_parallel_tool_plan_async(
        plan,
        tool_executors,
        timeout,
        agent_hierarchy,
        step,
        tool_call_counts,
    )
    if not interrupt_data:
        interrupt_data = await _execute_sequential_tool_plan_async(
            plan,
            tool_executors,
            timeout,
            agent_hierarchy,
            step,
            tool_call_counts,
        )
    return finalize_tool_call_plan(provider, plan, tool_call_counts, interrupt_data)
