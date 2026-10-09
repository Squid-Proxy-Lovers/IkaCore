import asyncio
import dataclasses
from concurrent.futures import ThreadPoolExecutor

import pytest

from IkaModel import RuntimeOptions, runtime_options
from IkaModel.base import AgentTool, ToolArgs
from IkaModel.connection_scope import scoped_client
from IkaModel.http_configuration import clear_ssl_context_cache, shared_ssl_context
from IkaModel.runtime_policy import current_runtime_options
from IkaModel.tool_schema import build_provider_tool_payload
from IkaModel.worker_context import submit_with_controls


def test_options_are_opt_in_and_nested_scopes_restore():
    assert current_runtime_options() == RuntimeOptions()
    with runtime_options(optimize_provider_payloads=True):
        with runtime_options(modern_retries=True):
            assert current_runtime_options().optimize_provider_payloads
            assert current_runtime_options().modern_retries
        assert not current_runtime_options().modern_retries
    assert current_runtime_options() == RuntimeOptions()


@pytest.mark.parametrize("changes", [
    {"compact_at_fraction": 0}, {"compact_at_fraction": 1.1},
    {"summary_context_threshold": -1}, {"summary_context_threshold": 1.5},
    {"retry_initial_seconds": float("inf")}, {"reuse_connections": "yes"},
])
def test_invalid_options_do_not_replace_previous_scope(changes):
    with pytest.raises(ValueError), runtime_options(**changes):
        pass
    assert current_runtime_options() == RuntimeOptions()


def test_worker_options_propagate_only_inside_explicit_scope():
    with ThreadPoolExecutor(max_workers=1) as pool:
        with runtime_options(optimize_provider_payloads=True):
            assert submit_with_controls(pool, current_runtime_options).result().optimize_provider_payloads
        assert not submit_with_controls(pool, current_runtime_options).result().optimize_provider_payloads


def test_control_tool_cache_is_separate_for_opt_in_choice_policy():
    tool = AgentTool(id="finish", name="agent_end", description="finish", args=ToolArgs("object", "args"), required=True)
    tools = [tool]
    assert build_provider_tool_payload("anthropic", tools).required_names == ("agent_end",)
    with runtime_options(optimize_provider_payloads=True):
        assert build_provider_tool_payload("anthropic", tools).required_names == ()
    assert build_provider_tool_payload("anthropic", tools).required_names == ("agent_end",)


def test_legacy_tool_args_required_option_preserves_default_serialization():
    original = ToolArgs("object", "args")
    assert set(dataclasses.asdict(original)) == {"type", "description", "agent", "data", "metadata", "properties"}
    explicit = ToolArgs("object", "args", properties={"value": {"type": "string"}}, required=["value"])
    assert explicit.properties["__required__"] == ["value"]


def test_scoped_clients_reuse_and_close_without_crossing_authentication():
    with runtime_options(reuse_connections=True):
        first = scoped_client("https://example.invalid", {"Authorization": "test-one"}, 1, False)
        assert scoped_client("https://example.invalid", {"Authorization": "test-one"}, 2, False) is first
        second = scoped_client("https://example.invalid", {"Authorization": "test-two"}, 1, False)
        assert second is not first
        assert not first.is_closed and not second.is_closed
    assert first.is_closed and second.is_closed


def test_scoped_clients_preserve_environment_proxy_routing(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    import httpx
    with runtime_options(reuse_connections=True):
        client = scoped_client("https://example.invalid", {}, 1, False)
        route = client._transport_for_url(httpx.URL("https://example.invalid"))
        assert "Proxy" in type(route._pool).__name__


def test_ssl_context_cache_can_be_explicitly_invalidated():
    clear_ssl_context_cache()
    first = shared_ssl_context()
    assert shared_ssl_context() is first
    clear_ssl_context_cache()
    assert shared_ssl_context() is not first


def test_async_tasks_get_independent_options():
    async def job(enabled):
        with runtime_options(optimize_provider_payloads=enabled):
            await asyncio.sleep(0)
            return current_runtime_options().optimize_provider_payloads

    async def run():
        return await asyncio.gather(job(True), job(False))

    assert asyncio.run(run()) == [True, False]
