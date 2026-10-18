"""Tool scheduling failure isolation and portable auth helper behavior."""

import builtins
import contextlib
import json
import sys
from concurrent.futures import Future, TimeoutError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from IkaModel.base import AgentEndException, HumanInputRequired
from IkaModel.chat_interface import tool_execution_sync as tools
from IkaModel.codex import auth
from IkaModel.runtime_errors import IkaRequestCancelled


@pytest.mark.parametrize("error", [AgentEndException({"content": "done"}), IkaRequestCancelled("stop"), HumanInputRequired({"question": "approve?"}), TimeoutError(), RuntimeError("executor broke")])
@pytest.mark.parametrize("timeout", [None, 0.01])
def test_parallel_plan_surfaces_interruptions_and_counts_failures(monkeypatch, error, timeout):
    future = Future()
    future.set_exception(error)
    monkeypatch.setattr(tools, "submit_with_controls", lambda *a: future)
    plan = tools.ToolCallPlan([], parallel_calls=[("fixture", {}, "id")])
    counts = {}
    if isinstance(error, (AgentEndException, IkaRequestCancelled)):
        with pytest.raises(type(error)) as caught:
            tools._execute_parallel_tool_plan(plan, {}, timeout, [], 0, counts)
        assert caught.value is error
    else:
        interrupted = tools._execute_parallel_tool_plan(plan, {}, timeout, [], 0, counts)
        assert counts == {"fixture": 1}
        result = json.loads(plan.tool_call_id_to_result["id"])
        if isinstance(error, HumanInputRequired):
            assert interrupted == {"question": "approve?"} and result["__ika_interrupt__"]
        else:
            assert interrupted is None
            assert "timed out" in result["error"] if isinstance(error, TimeoutError) else "executor broke" in result["error"]


def test_pool_shutdown_recovery_retries_submission_with_new_pool(monkeypatch):
    monkeypatch.setattr(tools, "_tool_executor_pool", SimpleNamespace(_shutdown=True))
    pool = tools._ensure_tool_executor_pool()
    try:
        assert pool.submit(lambda: "recovered").result() == "recovered"
    finally:
        pool.shutdown()
    monkeypatch.setattr(tools, "_ensure_tool_executor_pool", lambda: SimpleNamespace())
    monkeypatch.setattr(tools, "hooks_enabled", lambda: False)
    submissions = Mock(side_effect=[RuntimeError("shutdown race"), "new future"])
    monkeypatch.setattr(tools, "submit_with_controls", submissions)
    replacement = tools._submit_executor(lambda args: "ok", {}, "fixture")
    assert replacement == "new future" and submissions.call_count == 2
    tools._tool_executor_pool.shutdown()


def test_tool_arguments_reject_absent_nonobjects_and_nonserializable_values():
    for value in (None, [], {"value": object()}):
        with pytest.raises(ValueError):
            tools.validate_tool_args("fixture", value)
    assert tools._schema_dict("bad") == {} and tools._schema_list(None) == []
    assert tools._schema_list([1, "input", None]) == ["input"]
    schema = {"parameters": {"properties": {"__required__": ["value"], "value": {"type": None, "description": None}}}}
    assert '"value": <string>' in json.loads(tools._build_empty_args_error("fixture", {"fixture": schema}, {}))["error"]


def test_auth_default_token_url_and_windows_lock_fallback(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("CODEX_REFRESH_TOKEN_URL_OVERRIDE", raising=False)
    assert auth._resolve_oauth_token_url() == auth.CODEX_OAUTH_TOKEN_URL
    monkeypatch.setattr(auth, "_auth_json_path", lambda: tmp_path / "missing" / "auth.json")
    monkeypatch.setattr(auth, "_fcntl_warned", False)
    monkeypatch.setitem(sys.modules, "fcntl", None)
    for _ in range(2):
        with auth._interprocess_lock():
            pass
    assert caplog.text.count("fcntl unavailable") == 1


def test_auth_lock_parent_and_cleanup_errors_do_not_mask_body(monkeypatch, tmp_path):
    import fcntl
    from pathlib import Path

    monkeypatch.setattr(auth, "_lock_file_path", lambda: tmp_path / "auth.lock")
    monkeypatch.setattr(Path, "mkdir", Mock(side_effect=OSError("denied")))
    descriptor = Mock()
    descriptor.fileno.return_value = 123
    descriptor.close.side_effect = OSError("close failed")
    monkeypatch.setattr(builtins, "open", lambda *a: descriptor)
    flock = Mock(side_effect=[None, OSError("unlock failed")])
    monkeypatch.setattr(fcntl, "flock", flock)
    with auth._interprocess_lock():
        assert flock.call_args.args == (123, fcntl.LOCK_EX)
    descriptor.close.assert_called_once()


def test_auth_refresh_without_rotated_token_keeps_original(monkeypatch):
    data = {"tokens": {"access_token": "old", "refresh_token": "original", "id_token": "id"}}
    monkeypatch.setattr(auth, "_read_auth", lambda: data)
    monkeypatch.setattr(auth, "_refresh_tokens", lambda token: {"access_token": "new"})
    persisted = Mock()
    monkeypatch.setattr(auth, "_write_auth", persisted)
    result = auth._refresh_and_persist()
    assert result["tokens"] == {"access_token": "new", "refresh_token": "original", "id_token": "id"}
    persisted.assert_called_once_with(result)


def test_auth_double_check_adopts_another_process_token(monkeypatch):
    reads = Mock(side_effect=[{"tokens": {"access_token": "expired"}}, {"tokens": {"access_token": "fresh"}}])
    monkeypatch.setattr(auth, "_read_auth", reads)
    monkeypatch.setattr(auth, "_seconds_until_expiry", lambda token: 0 if token == "expired" else 9999)
    monkeypatch.setattr(auth, "_interprocess_lock", contextlib.nullcontext)
    refresh = Mock(side_effect=AssertionError("must reuse refreshed credential"))
    monkeypatch.setattr(auth, "_refresh_and_persist", refresh)
    assert auth.get_bearer() == "fresh"
    refresh.assert_not_called()
