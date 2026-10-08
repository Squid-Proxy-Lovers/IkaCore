import ssl

import httpx
import pytest

from IkaModel import http_config
from IkaModel.http_config import clear_ssl_context_cache, shared_ssl_context


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE"):
        monkeypatch.delenv(name, raising=False)
    clear_ssl_context_cache()
    yield
    clear_ssl_context_cache()


def test_context_is_built_once_and_shared(monkeypatch):
    builds = []
    real_build = http_config._build_context

    def counting_build():
        builds.append(1)
        return real_build()

    monkeypatch.setattr(http_config, "_build_context", counting_build)
    first = shared_ssl_context()
    assert shared_ssl_context() is first
    assert len(builds) == 1


def test_context_matches_httpx_default_verification():
    shared = shared_ssl_context()
    reference = httpx.create_ssl_context()
    assert isinstance(shared, ssl.SSLContext)
    assert shared.verify_mode == reference.verify_mode == ssl.CERT_REQUIRED
    assert shared.check_hostname is reference.check_hostname is True
    assert shared.cert_store_stats() == reference.cert_store_stats()


def test_environment_change_rebuilds_context(monkeypatch):
    built_for = []

    def fake_build():
        built_for.append(http_config._env_key())
        return ssl.create_default_context()

    monkeypatch.setattr(http_config, "_build_context", fake_build)
    first = shared_ssl_context()
    monkeypatch.setenv("SSL_CERT_DIR", "/tmp/custom-ca-dir")
    second = shared_ssl_context()
    assert second is not first
    assert shared_ssl_context() is second
    assert built_for == [(None, None, None), (None, "/tmp/custom-ca-dir", None)]


def test_clear_cache_forces_rebuild():
    first = shared_ssl_context()
    clear_ssl_context_cache()
    assert shared_ssl_context() is not first


def _pool_of(client):
    return client._transport._pool


def test_agent_clients_share_one_connection_pool_but_stay_owned():
    from IkaCore.agents import IkaBaseAgent

    first = IkaBaseAgent(name="a", description="d", prompt="p", model_id="gpt-4o", api_key="k")
    second = IkaBaseAgent(name="b", description="d", prompt="p", model_id="gpt-4o", api_key="k")
    c1, c2 = first._get_chat_client(), second._get_chat_client()
    assert c1 is not c2 and first._get_chat_client() is c1
    assert _pool_of(c1) is _pool_of(c2)
    assert c1.timeout.read == first.step_timeout

    first.shutdown()  # an owner closing its client must not tear down the shared pool
    assert c1.is_closed and not c2.is_closed
    third = IkaBaseAgent(name="c", description="d", prompt="p", model_id="gpt-4o", api_key="k")
    assert _pool_of(third._get_chat_client()) is _pool_of(c2)
    second.shutdown()
    third.shutdown()


def test_pool_uses_shared_ssl_context_and_is_rebuilt_when_it_changes():
    from IkaModel import http_pool

    pool = _pool_of(http_pool.new_pooled_client(5.0))
    assert pool is _pool_of(http_pool.new_pooled_client(5.0))
    clear_ssl_context_cache()
    assert _pool_of(http_pool.new_pooled_client(5.0)) is not pool


def test_async_pools_are_per_event_loop():
    import asyncio

    from IkaModel.http_pool import new_pooled_async_client

    async def pool_ids():
        a, b = new_pooled_async_client(5.0), new_pooled_async_client(5.0)
        try:
            return _pool_of(a), _pool_of(b)
        finally:
            await a.aclose()
            await b.aclose()

    first_a, first_b = asyncio.run(pool_ids())
    second_a, _ = asyncio.run(pool_ids())
    assert first_a is first_b and second_a is not first_a


def test_run_coroutine_reuses_loop_but_isolates_context_and_cancels_leftovers():
    import asyncio
    import contextvars

    from IkaModel.async_runner import run_coroutine

    var = contextvars.ContextVar("var", default="unset")
    leftovers = []

    async def step(value):
        seen = var.get()
        var.set(value)
        leftovers.append(asyncio.get_running_loop().create_task(asyncio.sleep(3600)))
        return asyncio.get_running_loop(), seen

    loop1, seen1 = run_coroutine(step("first"))
    loop2, seen2 = run_coroutine(step("second"))
    assert loop1 is loop2  # connections pooled on this loop survive between rounds
    assert seen1 == seen2 == "unset"  # each run gets a fresh context, like asyncio.run
    assert all(task.cancelled() for task in leftovers)  # pending tasks are cancelled, like asyncio.run

    async def nested():
        inner = step("x")
        try:
            return run_coroutine(inner)
        finally:
            inner.close()

    import pytest
    with pytest.raises(RuntimeError):
        asyncio.run(nested())
