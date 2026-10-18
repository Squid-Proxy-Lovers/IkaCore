"""Reject invalid public configuration before a request or run starts."""
from dataclasses import fields
from typing import get_type_hints

import httpx
import pytest

from IkaCore import (
    AgentConfig,
    AgentTopology,
    ExecutionLimits,
    ExecutionPolicy,
    OutputConfig,
    PromptConfig,
    ProviderConfig,
)
from IkaModel.request_transport import _RequestTransport, _SyncRequestAbort


@pytest.mark.parametrize("factory,field", [
    (ProviderConfig, "max_tokens"), (ProviderConfig, "thinking_budget_tokens"),
    (ExecutionLimits, "maxsteps"), (ExecutionLimits, "max_tool_rounds"),
    (ExecutionLimits, "max_tool_calls"), (ExecutionLimits, "context_budget"),
    *[(ExecutionPolicy, name) for name in ("max_step_extensions", "extend_steps_by",
        "max_stage_extensions", "extend_stage_steps_by", "validation_retries", "validation_extra_steps")],
    (OutputConfig, "logging_level"),
])
@pytest.mark.parametrize("value", [-1, 1.5, True, "1", float("nan"), float("inf")])
def test_integer_domains(factory, field, value):
    with pytest.raises(ValueError, match=field):
        factory(**{field: value})


@pytest.mark.parametrize("factory,field", [
    (ProviderConfig, "max_tokens"), (ProviderConfig, "thinking_budget_tokens"),
    (ExecutionLimits, "context_budget"), (ExecutionLimits, "step_timeout"),
    (ExecutionLimits, "rate_limit_per_min"),
])
def test_positive_domains_reject_zero(factory, field):
    with pytest.raises(ValueError, match=field):
        factory(**{field: 0})


@pytest.mark.parametrize("factory,field", [
    (ProviderConfig, "temperature"), (ExecutionLimits, "step_timeout"),
    (ExecutionLimits, "rate_limit_per_min"),
])
@pytest.mark.parametrize("value", [-1, True, "1", float("nan"), float("inf"), -float("inf")])
def test_finite_numeric_domains(factory, field, value):
    with pytest.raises(ValueError, match=field):
        factory(**{field: value})


@pytest.mark.parametrize("kwargs,match", [
    ({"temperature": 1.1}, "temperature"),
    ({"thinking_budget_tokens": 1023}, "thinking_budget_tokens"),
    ({"max_tokens": 1024, "thinking_budget_tokens": 1024}, "thinking_budget_tokens"),
    ({"client_factory": 1}, "client_factory"),
    ({"client": object(), "client_factory": lambda: object()}, "borrowed client"),
    ({"api_url": 1}, "api_url"), ({"reasoning_effort": 1}, "reasoning_effort"),
    ({"use_responses_api": 1}, "use_responses_api"),
])
def test_provider_domains(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ProviderConfig(**kwargs)


@pytest.mark.parametrize("rates", [None, [], {"lookup": 0}, {"lookup": -1},
    {"lookup": float("nan")}, {"lookup": float("inf")}, {"lookup": True}, {1: 10}])
def test_tool_rate_domains(rates):
    with pytest.raises(ValueError, match="per_tool_rate_limit"):
        ExecutionLimits(per_tool_rate_limit=rates)


@pytest.mark.parametrize("factory,kwargs,match", [
    (PromptConfig, {"start_prompt": 1}, "start_prompt"),
    (ExecutionPolicy, {"final_answer_checks": None}, "final_answer_checks"),
    (ExecutionPolicy, {"final_answer_checks": [False]}, "final_answer_checks"),
    (ExecutionPolicy, {"fail_closed": 1}, "fail_closed"),
    (ExecutionPolicy, {"async_requests": 1}, "async_requests"),
    (AgentTopology, {"subagents": "agent"}, "subagents"),
    (AgentTopology, {"subagents": [object()]}, "subagents"),
    (AgentTopology, {"next_agent": object()}, "next_agent"),
    (OutputConfig, {"subscribers": [object()]}, "subscribers"),
    (OutputConfig, {"subscribers": None}, "subscribers"),
    (OutputConfig, {"logging_level": 4}, "logging_level"),
    (OutputConfig, {"logging_file": None}, "logging_file"),
    (OutputConfig, {"cli": 1}, "cli"),
])
def test_other_config_domains(factory, kwargs, match):
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)


@pytest.mark.parametrize("kwargs,match", [
    ({"name": None}, "name"), ({"tools": [object()]}, "tools"),
    ({"stages": [object()]}, "stages"), ({"tools": None}, "tools"),
    ({"provider": None}, "provider"), ({"prompts": None}, "prompts"),
    ({"limits": None}, "limits"), ({"policy": None}, "policy"),
    ({"topology": None}, "topology"), ({"output": None}, "output"),
])
def test_agent_config_domains(kwargs, match):
    values = {"name": "agent", "description": "", "prompt": "", "model_id": "", "api_key": ""}
    with pytest.raises(ValueError, match=match):
        AgentConfig(**(values | kwargs))


def test_zero_allowances_and_runtime_introspection():
    limits = ExecutionLimits(maxsteps=0, max_tool_rounds=0, max_tool_calls=0)
    policy = ExecutionPolicy(max_step_extensions=0, extend_steps_by=0,
        max_stage_extensions=0, extend_stage_steps_by=0, validation_retries=0, validation_extra_steps=0)
    assert limits.maxsteps == policy.validation_retries == 0
    assert ProviderConfig(temperature=0).temperature == 0
    assert set(get_type_hints(AgentConfig)) == {field.name for field in fields(AgentConfig)}


@pytest.mark.parametrize("missing", ["_pool", "_connections", "_requests",
    "_optional_thread_lock", "_max_keepalive_connections"])
def test_missing_private_pool_contract_has_actionable_error(missing):
    source = httpx.HTTPTransport()
    pool = source._pool
    target = source if missing == "_pool" else pool
    original = getattr(target, missing)
    delattr(target, missing)
    try:
        with pytest.raises(RuntimeError, match=r"httpcore==1\.0\.9"):
            _RequestTransport(source, _SyncRequestAbort())
    finally:
        setattr(target, missing, original)
        source.close()
