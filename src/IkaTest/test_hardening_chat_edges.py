"""Chat termination and malformed-response edge contracts."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from IkaCore.logging_utils import IkaLogger
from IkaModel import execution_hooks, model_metadata, request_interface, retry_policy, summarization
from IkaModel.base import AgentEndException, BareBoneModel
from IkaModel.chat_interface import chat_request, chat_response
from IkaModel.chat_interface.debug_dump import dump_api_round
from IkaModel.chat_interface.response_usage import extract_usage
from IkaModel.chat_interface.tool_result_formatters import _coerce_gemini_tool_result, format_gemini_results
from IkaModel.chat_interface.types import ToolRuntimeState, UsageInfo
from IkaModel.history_replay import history_entries_to_replay, plan_history_replay
from IkaModel.runtime_errors import IkaProviderPayloadError, IkaRequestCancelled


@pytest.mark.parametrize("asynchronous", [False, True])
def test_force_completion_never_converts_cancellation_to_success(monkeypatch, asynchronous):
    error = IkaRequestCancelled("cancelled during final summarization")
    monkeypatch.setattr(summarization, "run_summarization", Mock(side_effect=error))
    monkeypatch.setattr(summarization, "async_summarise_message_history", AsyncMock(side_effect=error))
    history = chat_request.init_message_history()
    history["_pending_provider_data"] = {"provider": "responses", "items": [
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque-pending"}]}
    before = copy.deepcopy(history)
    model = BareBoneModel("gpt-4o", "test", "url")
    with pytest.raises(IkaRequestCancelled) as caught:
        if asynchronous:
            asyncio.run(chat_response._async_force_complete_with_summary(model, history, {}, None, [], "", "", "limit"))
        else:
            chat_response._force_complete_with_summary(model, history, {}, None, [], "", "", "limit")
    assert caught.value is error and history == before


@pytest.mark.parametrize("asynchronous", [False, True])
def test_summary_control_errors_propagate_without_history_mutation(monkeypatch, asynchronous):
    model = BareBoneModel("gpt-4o", "test", "https://api.openai.com/v1/chat/completions")
    history = chat_request.init_message_history()
    history["first_input"]["message"] = "task"
    before = copy.deepcopy(history)
    monkeypatch.setattr(summarization, "api_request_retry", Mock(side_effect=IkaRequestCancelled("stop")))
    monkeypatch.setattr(summarization, "async_api_request_retry", AsyncMock(side_effect=IkaRequestCancelled("stop")))
    with pytest.raises(IkaRequestCancelled):
        if asynchronous:
            asyncio.run(summarization.async_summarise_message_history(model, history))
        else:
            summarization.run_summarization(model, history)
    assert history == before


def test_chat_payload_validation_and_debug_dump_failure(monkeypatch, tmp_path):
    build = Mock(side_effect=ValueError("invalid tool schema"))
    with pytest.raises(IkaProviderPayloadError, match="invalid tool schema"):
        asyncio.run(chat_request._api_request_with_context_fallback_async(build, None, {}, client=SimpleNamespace()))
    monkeypatch.setenv("IKA_DUMP_REQUESTS", str(tmp_path / "not-directory"))
    (tmp_path / "not-directory").write_text("file")
    dump_api_round(SimpleNamespace(agent_name="test"), {}, None)
    assert (tmp_path / "not-directory").read_text() == "file"
    assert chat_response._estimate_tokens("") == 0


def test_signature_cache_is_bounded_and_bad_json_has_canonical_empty_signature():
    call = {"function": {"name": "fixture", "arguments": "{"}}
    assert chat_response._tool_signature_from_call(call) == ("fixture", "{}")
    recent = []
    chat_response._remember_tool_calls([{"function": {"name": str(i), "arguments": "{}"}} for i in range(15)], recent)
    assert len(recent) == 10 and recent[0][0] == "5"
    model = SimpleNamespace(_recent_tool_calls=[7, ("bad",), (7, "bad"), ("ok", "sig")])
    assert ToolRuntimeState.from_model(model).recent_tool_calls == [("ok", "sig")]
    assert UsageInfo.from_mapping({"input_tokens": True}).input_tokens == 1
    assert extract_usage("unknown", {"usage": {"total_tokens": 9}})["total_tokens"] == 9
    assert _coerce_gemini_tool_result(7) == 7
    assert _coerce_gemini_tool_result("{bad") == {"result": "{bad"}
    assert format_gemini_results([{"function": {"name": "f"}}], ["7"])[0]["functionResponse"]["response"] == {"result": 7}


def test_hooks_direct_invocation_and_generator_return_value():
    assert execution_hooks.invoke_tool(lambda args: args["v"], {"v": 7}, "fixture", {}) == 7
    async def tool(args):
        return args["v"]
    assert asyncio.run(execution_hooks.invoke_async_tool(tool, {"v": 8}, "fixture", {})) == 8
    def generator():
        yield {"__ika_checkpoint__": {"label": "saved", "payload": {"v": 1}}}
        return "returned"
    assert execution_hooks._consume_generator(generator()) == "returned"


def test_malformed_history_metadata_and_retry_hints_are_conservative(caplog):
    assert history_entries_to_replay({"messages": ["bad"]}, []) == []
    assert plan_history_replay({"summary": "bad"}, []).first_input_first
    assert model_metadata.get_provider_for_model("completely-unlisted") == "openai_responses"
    assert model_metadata.resolve_model_cost("") is None
    assert model_metadata.resolve_model_cost("gpt-4o@vendor") == model_metadata.MODEL_COSTS["gpt-4o"]
    assert retry_policy.retry_hint_seconds({"retry-after-ms": "bad"}, {"error": {"details": [7, {"@type": "Other"}, {"@type": "RetryInfo", "retryDelay": 7}]}}) is None
    request_interface._log_http_response("TEST", httpx.Response(200, json={"ok": True}))


def test_logger_empty_calls_color_and_shutdown_cleanup_failure(monkeypatch, tmp_path):
    logger = IkaLogger(0, str(tmp_path / "log.txt"))
    logger._writer_loop()
    logger.log_tool_results([], [])
    assert "\033[" in logger._color("text", "cyan")
    file_logger = IkaLogger(1, str(tmp_path / "out.txt"))
    file_logger.write_line("persisted")
    monkeypatch.setattr("IkaCore.logging_utils_support.atexit.unregister", Mock(side_effect=ValueError("unavailable")))
    file_logger.shutdown()
    assert "persisted" in (tmp_path / "out.txt").read_text()
    assert not file_logger._atexit_registered


def test_tool_future_poll_timeout_continues_until_completion():
    from concurrent.futures import TimeoutError

    from IkaModel.request_control import request_controls
    from IkaModel.request_lifecycle import wait_for_tool
    future = Mock()
    future.result.side_effect = [TimeoutError(), "completed"]
    future.done.return_value = False
    with request_controls(timeout=1):
        assert wait_for_tool(future, None) == "completed"
    assert future.result.call_count == 2


def test_history_replay_ignores_invalid_summary_section():
    history = {"summary": [], "messages": {"stage": {"type": "stage_input", "message": "task"}}}
    assert plan_history_replay(history, []).first_input_first is False


@pytest.mark.parametrize("asynchronous", [False, True])
def test_chat_returns_flush_interruption_without_finalizing_it(monkeypatch, asynchronous):
    from IkaModel.chat_interface import chat_runtime

    model = BareBoneModel("gpt-4o", "test", "https://api.openai.com/v1/chat/completions")
    history = chat_request.init_message_history()
    interruption = {"content": "paused", "interrupted": True, "interrupt_data": {"question": "Approve?"}}
    round_result = ("text", None, [], 0, {})
    monkeypatch.setattr(chat_runtime, "_request_provider_round", Mock(return_value=round_result))
    monkeypatch.setattr(chat_runtime, "_request_provider_round_async", AsyncMock(return_value=round_result))
    monkeypatch.setattr(chat_runtime, "_flush_sync_remaining_tools", Mock(return_value=interruption))
    monkeypatch.setattr(chat_runtime, "_flush_async_remaining_tools", AsyncMock(return_value=interruption))
    if asynchronous:
        result = asyncio.run(chat_runtime.async_chat(model, [{"role": "user", "content": "task"}], history, client=object()))
    else:
        result = chat_runtime.chat(model, [{"role": "user", "content": "task"}], history)
    assert result is interruption and history["messages"] == {}


def test_async_forced_completion_logs_the_final_saved_output(monkeypatch):
    model = BareBoneModel("gpt-4o", "test", "url")
    history = chat_request.init_message_history()
    logger = Mock()
    logger.compute_cost.return_value = {"total_cost": 1}
    monkeypatch.setattr(summarization, "async_summarise_message_history", AsyncMock(return_value="final"))
    with pytest.raises(AgentEndException) as ended:
        asyncio.run(chat_response._async_force_complete_with_summary(model, history, {"total_tokens": 5}, logger, [], "", "", "limit"))
    assert ended.value.response["content"] == "final"
    logger.log_output.assert_called_once_with("final", UsageInfo(total_tokens=5).to_dict(), {"total_cost": 1}, history)


def test_sync_repeated_guard_stops_before_tool_effects_and_async_loop_stops_when_told(monkeypatch):
    from IkaModel.chat_interface import chat_tool_loop_async, chat_tool_loop_sync

    model = BareBoneModel("gpt-4o", "test", "url")
    call = {"id": "a", "function": {"name": "fixture", "arguments": "{}"}}
    state = chat_response._new_chat_state("openai", "text", None, [call], 1, {})
    runtime = ToolRuntimeState()
    effect = Mock()
    stopped = {"content": "stopped", "hijacked": True}
    monkeypatch.setattr(chat_tool_loop_sync, "_sync_repeated_tool_response_for_state", Mock(return_value=(stopped, ["fixture"])))
    args = (model, [], {}, {"fixture": effect}, None, 1, None, 5, None, state, runtime)
    assert chat_tool_loop_sync._sync_tool_loop_iteration(*args) == (stopped, False)
    effect.assert_not_called()
    iteration = AsyncMock(return_value=(None, False))
    monkeypatch.setattr(chat_tool_loop_async, "_async_tool_loop_iteration", iteration)
    assert asyncio.run(chat_tool_loop_async._async_tool_loop(*args)) is None
    iteration.assert_awaited_once()


def test_async_repeated_guard_does_not_swallow_summary_cancellation(monkeypatch):
    from IkaModel.chat_interface.chat_tool_loop_async import _async_repeated_tool_guard

    model = BareBoneModel("gpt-4o", "test", "url")
    call = {"id": "a", "function": {"name": "fixture", "arguments": "{}"}}
    runtime = ToolRuntimeState(recent_tool_calls=[("fixture", "{}")] * 5)
    error = IkaRequestCancelled("stop")
    monkeypatch.setattr(summarization, "async_summarise_message_history", AsyncMock(side_effect=error))
    with pytest.raises(IkaRequestCancelled) as caught:
        asyncio.run(_async_repeated_tool_guard(model, None, [call], runtime, {}, {}, [], "", object()))
    assert caught.value is error


def test_summary_token_bool_and_missing_endpoint_resolution():
    assert summarization._token_count(True) == 1
    model = SimpleNamespace(model_id="gpt-4o", api_url="")
    assert summarization._resolve_summary_target(model, "openai", True) == ("gpt-4o", "https://api.openai.com/v1/chat/completions")
