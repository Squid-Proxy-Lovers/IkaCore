"""Translate legacy-shaped fixture dictionaries to the grouped constructor for unchanged behavioral cases."""
from IkaCore import (
    AgentTopology,
    ExecutionLimits,
    ExecutionPolicy,
    OutputConfig,
    PromptConfig,
    ProviderConfig,
)


def agent_kwargs(values):
    values = dict(values)
    group = {key: values.pop(key) for key in ('api_url', 'max_tokens', 'temperature', 'use_responses_api', 'reasoning_effort') if key in values}
    if group:
        values["provider"] = ProviderConfig(**group)
    group = {key: values.pop(key) for key in ('system_prompt', 'start_prompt', 'end_prompt') if key in values}
    if group:
        values["prompts"] = PromptConfig(**group)
    group = {key: values.pop(key) for key in ('maxsteps', 'step_timeout', 'max_tool_rounds', 'max_tool_calls', 'rate_limit_per_min', 'per_tool_rate_limit') if key in values}
    if group:
        values["limits"] = ExecutionLimits(**group)
    group = {key: values.pop(key) for key in ('final_answer_check', 'max_step_extensions', 'extend_steps_by', 'max_stage_extensions', 'extend_stage_steps_by', 'async_requests') if key in values}
    if "final_answer_check" in group:
        group["final_answer_checks"] = tuple(group.pop("final_answer_check") or ())
    if group:
        values["policy"] = ExecutionPolicy(**group)
    group = {key: values.pop(key) for key in ('subagents', 'next_agent') if key in values}
    if group:
        values["topology"] = AgentTopology(**group)
    group = {key: values.pop(key) for key in ('logging_level', 'logging_file', 'show_usage_level0', 'summarize_final') if key in values}
    if group:
        values["output"] = OutputConfig(**group)
    return values
