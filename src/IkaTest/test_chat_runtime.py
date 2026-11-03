import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from IkaModel.base import AgentEndException, AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_interface import chat_runtime as cr
from IkaModel.chat_interface.types import ToolRuntimeState, UsageInfo
from IkaModel.request_interface import IkaAPIError
from IkaModel.runtime_errors import IkaContextWindowError, IkaProviderResponseError


def _model(**overrides):
    defaults = {
        "model_id": "gpt-4o",
        "api_key": "test-key",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "suppress_init_output": True,
        "use_responses_api": False,
    }
    defaults.update(overrides)
    return BareBoneModel(**defaults)


def _history():
    return cr.init_message_history()


def _tool_call(name="search", arguments='{"q":"x"}', call_id="call_1"):
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def test_dump_api_round_writes_debug_file_when_enabled(tmp_path, monkeypatch):
    model = _model(agent_name="Agent/One")
    payload = {"messages": [{"role": "user", "content": "hello"}]}

    monkeypatch.setenv("IKA_DUMP_REQUESTS", str(tmp_path))
    cr._dump_api_round(model, payload, {"ok": True})

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["agent_name"] == "Agent/One"
    assert saved["model_id"] == "gpt-4o"
    assert saved["payload"] == payload
    assert saved["response"] == {"ok": True}
    assert len(saved["messages_sha256_12"]) == 12


def test_tool_metadata_extracts_parallel_limits_and_required_properties():
    model = _model()
    model.agent_tools = [
        AgentTool(
            "search",
            "search",
            "Search",
            ToolArgs(
                "object",
                "payload",
                properties={
                    "query": {"type": "string", "description": "Query text"},
                    "__required__": ["query"],
                    "ignored": "not a schema object",
                },
            ),
            parallel=False,
            limit_calls=2,
        )
    ]

    metadata = cr._build_tool_metadata(model)

    assert cr._build_tool_metadata(SimpleNamespace()) == {}
    assert metadata["search"]["parallel"] is False
    assert metadata["search"]["limit_calls"] == 2
    assert metadata["search"]["parameters"] == {
        "properties": {"query": {"type": "string", "description": "Query text"}},
        "required": ["query"],
    }


def test_prepare_chat_inputs_validates_contract_and_logs_input():
    model = _model()
    logger = MagicMock()

    message_history, tools = cr._prepare_chat_inputs(
        model,
        [{"role": "user", "content": "hi"}],
        None,
        None,
        logger,
    )

    assert set(message_history) == {"system", "first_input", "summary", "messages"}
    assert tools == {}
    logger.log_input.assert_called_once()

    with pytest.raises(ValueError, match="barebone_model"):
        cr._prepare_chat_inputs(None, [{"role": "user", "content": "hi"}], None, None, None)
    with pytest.raises(ValueError, match="cannot be empty"):
        cr._prepare_chat_inputs(model, [], None, None, None)
    with pytest.raises(ValueError, match="must be a list"):
        cr._prepare_chat_inputs(model, "hi", None, None, None)
    with pytest.raises(ValueError, match="model_id"):
        cr._prepare_chat_inputs(SimpleNamespace(model_id="", api_key="k", api_url="u"), [{"role": "user"}], None, None, None)
    with pytest.raises(ValueError, match="api_key"):
        cr._prepare_chat_inputs(SimpleNamespace(model_id="m", api_key="", api_url="u"), [{"role": "user"}], None, None, None)
    with pytest.raises(ValueError, match="api_url"):
        cr._prepare_chat_inputs(SimpleNamespace(model_id="m", api_key="k", api_url=""), [{"role": "user"}], None, None, None)
    with pytest.raises(ValueError, match="tool_executors"):
        cr._prepare_chat_inputs(model, [{"role": "user"}], None, [], None)


def test_force_complete_with_summary_falls_back_and_clears_forced_tool_choice():
    model = _model()
    model.forced_tool_name = "agent_end"
    history = _history()
    logger = MagicMock()
    logger.compute_cost.return_value = {"total_cost": 0.1}

    with patch("IkaModel.chat_interface.chat_response.run_summarization", side_effect=RuntimeError("summary failed")):
        with pytest.raises(AgentEndException) as exc_info:
            cr._force_complete_with_summary(
                model,
                history,
                {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                logger,
                [_tool_call("search")],
                content_before_tools="partial answer",
                content="ignored",
                reason="max tools",
            )

    response = exc_info.value.response
    assert response["content"] == "partial answer"
    assert response["hijacked"] is True
    assert response["executed_tool_calls"][-1]["function"]["name"] == "agent_end"
    assert model.forced_tool_name is None
    assert history["messages"]
    logger.log_output.assert_called_once()


def test_async_force_complete_with_summary_uses_reason_when_summary_and_content_are_empty():
    async def run_case():
        model = _model()
        model.forced_tool_name = "agent_end"
        history = _history()
        with patch(
            "IkaModel.chat_interface.chat_response.async_summarise_message_history",
            side_effect=RuntimeError("summary failed"),
        ):
            with pytest.raises(AgentEndException) as exc_info:
                await cr._async_force_complete_with_summary(
                    model,
                    history,
                    UsageInfo(),
                    None,
                    [],
                    content_before_tools=".",
                    content="",
                    reason="max tools reached",
                    client=None,
                )
        return model, history, exc_info.value.response

    model, history, response = asyncio.run(run_case())

    assert response["content"] == "max tools reached"
    assert response["executed_tool_calls"][-1]["function"]["name"] == "agent_end"
    assert model.forced_tool_name is None
    assert history["messages"]


def test_interrupt_and_control_completion_helpers_build_expected_responses():
    model = _model()
    model.forced_tool_name = "stage_end"
    history = _history()
    logger = MagicMock()
    logger.compute_cost.return_value = {"total_cost": 0.0}
    messages = []
    executed = [_tool_call("ask_user")]
    tool_messages = [{"role": "tool", "content": "paused", "tool_call_id": "call_1"}]
    interrupt_data = {"question": "Proceed?", "stage_name": "Review"}

    assert cr._interrupt_response_if_needed(
        model,
        logger,
        None,
        "openai",
        messages,
        history,
        "content",
        None,
        executed,
        executed,
        tool_messages,
        ["paused"],
        tokens=2,
        content_before_tools="content",
        total_usage=UsageInfo(total_tokens=2),
        hijacked=False,
    ) is None

    response = cr._interrupt_response_if_needed(
        model,
        logger,
        interrupt_data,
        "openai",
        messages,
        history,
        "content",
        None,
        executed,
        executed,
        tool_messages,
        ["paused"],
        tokens=2,
        content_before_tools="content",
        total_usage=UsageInfo(total_tokens=2),
        hijacked=False,
        hitl_stage_name="Review",
    )

    assert response["interrupted"] is True
    assert response["content"] == "Proceed?"
    assert response["interrupt_data"] == interrupt_data
    logger.log_hitl_prompt.assert_called_once_with("Review")

    assert cr._handle_control_tool_completion(
        model,
        logger,
        "done",
        None,
        [_tool_call("stage_end")],
        [_tool_call("stage_end")],
        "done",
        history,
        UsageInfo(total_tokens=2),
        False,
        tokens=2,
    ) is True
    assert model.forced_tool_name is None

    with pytest.raises(AgentEndException) as exc_info:
        cr._handle_control_tool_completion(
            model,
            logger,
            "final",
            "reasoning",
            [_tool_call("agent_end")],
            [_tool_call("agent_end")],
            "before",
            history,
            UsageInfo(total_tokens=2),
            False,
            tokens=2,
        )
    assert exc_info.value.response["content"] == "final"


def test_repeated_tool_guards_warn_then_hijack_after_five_identical_calls():
    model = _model()
    history = _history()
    calls = [_tool_call("search")]
    runtime = SimpleNamespace(recent_tool_calls=[cr._tool_signature_from_call(calls[0])] * 3)

    response, names = cr._sync_repeated_tool_guard(
        model,
        None,
        calls,
        runtime,
        history,
        UsageInfo(),
        [],
        "partial",
        client=None,
    )

    assert response is None
    assert names == ["search"]
    assert "search" in cr._repeated_tool_warning(names)

    runtime.recent_tool_calls = [cr._tool_signature_from_call(calls[0])] * 5
    with patch("IkaModel.chat_interface.chat_response.run_summarization", return_value=""):
        response, names = cr._sync_repeated_tool_guard(
            model,
            None,
            calls,
            runtime,
            history,
            UsageInfo(total_tokens=1),
            [],
            "partial",
            client=None,
        )

    assert response["hijacked"] is True
    assert "Agent stuck in loop" in response["content"]
    assert names == ["search"]


def test_async_repeated_tool_guard_falls_back_when_force_summary_fails():
    async def run_case():
        model = _model()
        history = _history()
        calls = [_tool_call("search")]
        runtime = SimpleNamespace(recent_tool_calls=[cr._tool_signature_from_call(calls[0])] * 5)
        with patch(
            "IkaModel.chat_interface.chat_response.async_summarise_message_history",
            side_effect=RuntimeError("summary failed"),
        ):
            return await cr._async_repeated_tool_guard(
                model,
                None,
                calls,
                runtime,
                history,
                UsageInfo(total_tokens=1),
                [],
                "partial",
                client=MagicMock(),
            )

    response, names = asyncio.run(run_case())

    assert response["hijacked"] is True
    assert "Partial results" in response["content"]
    assert names == ["search"]


def test_tool_execution_result_updates_runtime_state_and_appends_messages():
    model = _model(agent_hierarchy=["A"])
    history = _history()
    messages = []
    logger = MagicMock()
    runtime = ToolRuntimeState.from_model(model)
    state = cr._new_chat_state("openai", "content", "reasoning", [_tool_call("search")], 4, {"total_tokens": 4})
    batch = (
        [{"role": "tool", "content": '{"ok":true}', "tool_call_id": "call_1"}],
        ['{"ok":true}'],
        {"search": 1},
        [_tool_call("search")],
        None,
    )

    cr._apply_tool_execution_result(
        model,
        logger,
        runtime,
        state.executed_tool_calls,
        batch[2],
        batch[3],
        batch[1],
        remember_signatures=True,
    )
    cr._append_tool_loop_messages(messages, history, state, batch, repeated_tool_names={"search"})

    assert model._tool_call_counts == {"search": 1}
    assert model._current_step == 1
    assert state.executed_tool_calls == [_tool_call("search")]
    assert runtime.recent_tool_calls == [cr._tool_signature_from_call(_tool_call("search"))]
    assert messages
    assert history["messages"]
    assert "System note" in messages[-1].get("content", "")
    logger.log_tool_results.assert_called_once()


def test_sync_and_async_advance_tool_round_update_usage_or_stop_at_limit():
    model = _model()
    history = _history()
    state = cr._new_chat_state("openai", "content", None, [_tool_call("search")], 1, {"total_tokens": 1})

    assert cr._sync_advance_tool_round(model, [], history, 1, None, max_tool_rounds=1, state=state) is False
    assert state.tool_calls == []

    state = cr._new_chat_state("openai", "content", None, [_tool_call("search")], 1, {"total_tokens": 1})
    with patch(
        "IkaModel.chat_interface.chat_tool_loop_sync._request_provider_round",
        return_value=("next", None, [], 3, {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}),
    ):
        assert cr._sync_advance_tool_round(model, [], history, 1, None, max_tool_rounds=2, state=state) is True
    assert state.content == "next"
    assert state.usage.total_tokens == 4

    async def run_case():
        async_state = cr._new_chat_state("openai", "content", None, [_tool_call("search")], 1, {"total_tokens": 1})
        assert await cr._async_advance_tool_round(
            model,
            [],
            history,
            1,
            MagicMock(),
            max_tool_rounds=1,
            state=async_state,
        ) is False
        assert async_state.tool_calls == []

        async_state = cr._new_chat_state("openai", "content", None, [_tool_call("search")], 1, {"total_tokens": 1})
        with patch(
            "IkaModel.chat_interface.chat_tool_loop_async._request_provider_round_async",
            return_value=("next async", None, [], 2, {"total_tokens": 2}),
        ):
            assert await cr._async_advance_tool_round(
                model,
                [],
                history,
                1,
                MagicMock(),
                max_tool_rounds=2,
                state=async_state,
            ) is True
        return async_state

    async_state = asyncio.run(run_case())
    assert async_state.content == "next async"
    assert async_state.usage.total_tokens == 3


def test_flush_remaining_tools_handles_empty_interrupt_stage_end_and_append_paths():
    model = _model()
    history = _history()
    messages = []
    logger = MagicMock()
    runtime = ToolRuntimeState.from_model(model)

    empty_state = cr._new_chat_state("openai", "content", None, [], 1, {"total_tokens": 1})
    assert cr._flush_sync_remaining_tools(model, messages, history, {}, logger, 1, empty_state, runtime) is None

    state = cr._new_chat_state("openai", "content", None, [_tool_call("ask")], 1, {"total_tokens": 1})
    interrupt_batch = ([], ["paused"], {"ask": 1}, [_tool_call("ask")], {"question": "Proceed?"})
    with patch("IkaModel.chat_interface.chat_tool_loop_sync._sync_execute_tool_batch", return_value=interrupt_batch):
        response = cr._flush_sync_remaining_tools(model, messages, history, {"ask": lambda _args: "unused"}, logger, 1, state, runtime)
    assert response["interrupted"] is True
    assert state.tool_calls == [_tool_call("ask")]

    state = cr._new_chat_state("openai", "content", None, [_tool_call("stage_end")], 1, {"total_tokens": 1})
    stage_batch = ([], ["ok"], {"stage_end": 1}, [_tool_call("stage_end")], None)
    with patch("IkaModel.chat_interface.chat_tool_loop_sync._sync_execute_tool_batch", return_value=stage_batch):
        assert cr._flush_sync_remaining_tools(
            model,
            messages,
            history,
            {"stage_end": lambda _args: "unused"},
            logger,
            1,
            state,
            runtime,
        ) is None
    assert state.tool_calls == []

    state = cr._new_chat_state("openai", "content", None, [_tool_call("search")], 1, {"total_tokens": 1})
    append_batch = (
        [{"role": "tool", "content": "ok", "tool_call_id": "call_1"}],
        ["ok"],
        {"search": 1},
        [_tool_call("search")],
        None,
    )
    with patch("IkaModel.chat_interface.chat_tool_loop_sync._sync_execute_tool_batch", return_value=append_batch):
        assert cr._flush_sync_remaining_tools(model, messages, history, {"search": lambda _args: "unused"}, logger, 1, state, runtime) is None
    assert state.tool_calls == []
    assert messages


def test_async_flush_remaining_tools_handles_empty_interrupt_and_append_paths():
    async def run_case():
        model = _model()
        history = _history()
        messages = []
        logger = MagicMock()
        runtime = ToolRuntimeState.from_model(model)
        client = MagicMock()

        empty_state = cr._new_chat_state("openai", "content", None, [], 1, {"total_tokens": 1})
        assert await cr._flush_async_remaining_tools(
            model,
            messages,
            history,
            {},
            logger,
            1,
            client,
            empty_state,
            runtime,
        ) is None

        state = cr._new_chat_state("openai", "content", None, [_tool_call("ask")], 1, {"total_tokens": 1})
        interrupt_batch = ([], ["paused"], {"ask": 1}, [_tool_call("ask")], {"question": "Proceed?"})
        with patch(
            "IkaModel.chat_interface.chat_tool_loop_async._async_execute_tool_batch",
            AsyncMock(return_value=interrupt_batch),
        ):
            response = await cr._flush_async_remaining_tools(
                model,
                messages,
                history,
                {"ask": lambda _args: "unused"},
                logger,
                1,
                client,
                state,
                runtime,
            )
        assert response["interrupted"] is True
        assert state.tool_calls == [_tool_call("ask")]

        state = cr._new_chat_state("openai", "content", None, [_tool_call("search")], 1, {"total_tokens": 1})
        append_batch = (
            [{"role": "tool", "content": "ok", "tool_call_id": "call_1"}],
            ["ok"],
            {"search": 1},
            [_tool_call("search")],
            None,
        )
        with patch(
            "IkaModel.chat_interface.chat_tool_loop_async._async_execute_tool_batch",
            AsyncMock(return_value=append_batch),
        ):
            assert await cr._flush_async_remaining_tools(
                model,
                messages,
                history,
                {"search": lambda _args: "unused"},
                logger,
                1,
                client,
                state,
                runtime,
            ) is None
        assert state.tool_calls == []
        assert messages

    asyncio.run(run_case())


def test_async_tool_loop_iteration_handles_repeated_response_and_continue_path():
    async def run_case():
        model = _model()
        history = _history()
        messages = []
        logger = MagicMock()
        client = MagicMock()
        runtime = ToolRuntimeState.from_model(model)
        state = cr._new_chat_state("openai", "content", None, [_tool_call("search")], 1, {"total_tokens": 1})
        repeated_response = {"content": "stopped", "hijacked": True}

        with patch(
            "IkaModel.chat_interface.chat_tool_loop_async._async_repeated_tool_response_for_state",
            AsyncMock(return_value=(repeated_response, ["search"])),
        ):
            assert await cr._async_tool_loop_iteration(
                model,
                messages,
                history,
                {"search": lambda _args: "unused"},
                logger,
                1,
                client,
                5,
                None,
                state,
                runtime,
            ) == (repeated_response, False)

        state = cr._new_chat_state("openai", "content", None, [_tool_call("search")], 1, {"total_tokens": 1})
        batch = (
            [{"role": "tool", "content": "ok", "tool_call_id": "call_1"}],
            ["ok"],
            {"search": 1},
            [_tool_call("search")],
            None,
        )
        with patch(
            "IkaModel.chat_interface.chat_tool_loop_async._async_repeated_tool_response_for_state",
            AsyncMock(return_value=(None, ["search"])),
        ):
            with patch(
                "IkaModel.chat_interface.chat_tool_loop_async._async_execute_tool_batch",
                AsyncMock(return_value=batch),
            ):
                with patch("IkaModel.chat_interface.chat_tool_common._tool_loop_post_batch_result", return_value=None):
                    with patch(
                        "IkaModel.chat_interface.chat_tool_loop_async._async_continue_tool_loop_after_batch",
                        AsyncMock(return_value=True),
                    ):
                        assert await cr._async_tool_loop_iteration(
                            model,
                            messages,
                            history,
                            {"search": lambda _args: "unused"},
                            logger,
                            1,
                            client,
                            5,
                            None,
                            state,
                            runtime,
                        ) == (None, True)

    asyncio.run(run_case())


def test_max_tool_limit_force_completion_delegates_to_sync_and_async_helpers():
    model = _model()
    history = _history()
    logger = MagicMock()
    state = cr._new_chat_state("openai", "content", None, [_tool_call("search")], 1, {"total_tokens": 1})
    runtime = ToolRuntimeState.from_model(model)
    runtime.current_step = 2

    with patch("IkaModel.chat_interface.chat_tool_loop_sync._force_complete_with_summary") as force_complete:
        cr._sync_force_complete_if_max_tools(model, logger, history, state, runtime, max_tool_calls=2)
    force_complete.assert_called_once()

    async def run_async_case():
        async_runtime = ToolRuntimeState.from_model(model)
        async_runtime.current_step = 3
        async_state = cr._new_chat_state("openai", "content", None, [_tool_call("search")], 1, {"total_tokens": 1})
        with patch(
            "IkaModel.chat_interface.chat_tool_loop_async._async_force_complete_with_summary",
            AsyncMock(),
        ) as async_force_complete:
            await cr._async_force_complete_if_max_tools(
                model,
                logger,
                history,
                async_state,
                async_runtime,
                max_tool_calls=3,
                client=MagicMock(),
            )
        async_force_complete.assert_awaited_once()

    asyncio.run(run_async_case())


def test_api_request_with_context_fallback_success_non_context_and_async_context_retry():
    model = _model()
    history = _history()

    non_json_response = MagicMock()
    non_json_response.json.side_effect = ValueError("not json")
    with patch("IkaModel.chat_interface.chat_request.api_request_retry", return_value=non_json_response):
        response = cr._api_request_with_context_fallback(
            lambda: ("https://example.test", {}, {"messages": []}),
            model,
            history,
            timeout=1,
        )
    assert response is non_json_response

    with patch("IkaModel.chat_interface.chat_request.api_request_retry", side_effect=IkaAPIError("temporary")):
        with pytest.raises(IkaAPIError):
            cr._api_request_with_context_fallback(
                lambda: ("https://example.test", {}, {"messages": []}),
                model,
                history,
                timeout=1,
            )

    async def run_async_context_retry():
        calls = {"count": 0}

        def build_payload():
            calls["count"] += 1
            return "https://example.test", {}, {"messages": [calls["count"]]}

        response = MagicMock()
        response.json.side_effect = ValueError("not json")
        with patch(
            "IkaModel.chat_interface.chat_request.async_api_request_retry",
            side_effect=[IkaAPIError("maximum context length exceeded"), response],
        ):
            with patch("IkaModel.chat_interface.chat_request.async_summarise_message_history", return_value="summary"):
                with patch("IkaModel.chat_interface.chat_request.get_cli_output", return_value=MagicMock()):
                    out = await cr._api_request_with_context_fallback_async(
                        build_payload,
                        model,
                        history,
                        timeout=1,
                        client=MagicMock(),
                    )
        return calls["count"], out

    calls, response = asyncio.run(run_async_context_retry())
    assert calls == 2
    assert response.json.side_effect


def test_sync_context_fallback_summarizes_rebuilds_and_dumps_retry_response(tmp_path, monkeypatch):
    model = _model(agent_name="Retry/Agent")
    history = _history()
    monkeypatch.setenv("IKA_DUMP_REQUESTS", str(tmp_path))

    def build_payload():
        summary = history["summary"]["message"]
        return "https://example.test", {}, {"messages": [summary or "full"]}

    def summarize(_model, message_history, **_kwargs):
        message_history["summary"]["message"] = "summary"
        return "summary"

    response = MagicMock()
    response.json.side_effect = ValueError("not json")
    with patch(
        "IkaModel.chat_interface.chat_request.api_request_retry",
        side_effect=[IkaAPIError("maximum context length exceeded"), response],
    ) as request:
        with patch("IkaModel.chat_interface.chat_request.summarise_message_history", side_effect=summarize):
            with patch("IkaModel.chat_interface.chat_request.get_cli_output", return_value=MagicMock()):
                out = cr._api_request_with_context_fallback(build_payload, model, history, timeout=1)

    assert out is response
    assert [call.args[2] for call in request.call_args_list] == [
        {"messages": ["full"]},
        {"messages": ["summary"]},
    ]
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["error"] == "response not json (post-summarization)"
    assert saved["payload"] == {"messages": ["summary"]}


def test_context_fallback_wraps_sync_and_async_summary_failures():
    model = _model()
    history = _history()

    with patch(
        "IkaModel.chat_interface.chat_request.api_request_retry",
        side_effect=IkaAPIError("maximum context length exceeded"),
    ):
        with patch("IkaModel.chat_interface.chat_request.summarise_message_history", side_effect=RuntimeError("down")):
            with patch("IkaModel.chat_interface.chat_request.get_cli_output", return_value=MagicMock()):
                with pytest.raises(IkaContextWindowError, match="Failed to summarize context"):
                    cr._api_request_with_context_fallback(
                        lambda: ("https://example.test", {}, {"messages": []}),
                        model,
                        history,
                    )

    async def run_async_case():
        with patch(
            "IkaModel.chat_interface.chat_request.async_api_request_retry",
            side_effect=IkaAPIError("maximum context length exceeded"),
        ):
            with patch(
                "IkaModel.chat_interface.chat_request.async_summarise_message_history",
                side_effect=RuntimeError("down"),
            ):
                with patch("IkaModel.chat_interface.chat_request.get_cli_output", return_value=MagicMock()):
                    with pytest.raises(IkaContextWindowError, match="Failed to summarize context"):
                        await cr._api_request_with_context_fallback_async(
                            lambda: ("https://example.test", {}, {"messages": []}),
                            model,
                            history,
                            client=MagicMock(),
                        )

    asyncio.run(run_async_case())


def test_request_provider_rounds_parse_response_and_initial_async_state_syncs_model():
    model = _model()
    history = _history()
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": "hello", "tool_calls": []}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    response.raise_for_status = MagicMock()

    with patch("IkaModel.chat_interface.chat_request._api_request_with_context_fallback", return_value=response):
        content, reasoning, tool_calls, tokens, usage = cr._request_provider_round(
            "openai",
            model,
            [{"role": "user", "content": "hi"}],
            history,
            timeout=1,
            client=None,
        )

    assert (content, reasoning, tool_calls, tokens) == ("hello", None, [], 3)
    assert usage["total_tokens"] == 3
    response.raise_for_status.assert_called_once()

    async def run_case():
        async_response = MagicMock()
        async_response.json.return_value = response.json.return_value
        async_response.raise_for_status = MagicMock()
        with patch(
            "IkaModel.chat_interface.chat_request._api_request_with_context_fallback_async",
            return_value=async_response,
        ):
            state, runtime = await cr._initial_async_chat_state(
                "openai",
                model,
                [{"role": "user", "content": "hi"}],
                history,
                timeout=1,
                client=MagicMock(),
            )
        return state, runtime, async_response

    state, runtime, async_response = asyncio.run(run_case())
    assert state.content == "hello"
    assert state.usage.total_tokens == 3
    assert model._current_step == runtime.current_step
    async_response.raise_for_status.assert_called_once()


def test_provider_round_wraps_non_json_and_malformed_responses_as_domain_errors():
    model = _model()
    history = _history()

    non_json = MagicMock()
    non_json.raise_for_status = MagicMock()
    non_json.json.side_effect = ValueError("not json")
    with patch("IkaModel.chat_interface.chat_request._api_request_with_context_fallback", return_value=non_json):
        with pytest.raises(IkaProviderResponseError, match="response was not valid JSON"):
            cr._request_provider_round("openai", model, [], history, timeout=1, client=None)

    wrong_shape = MagicMock()
    wrong_shape.raise_for_status = MagicMock()
    wrong_shape.json.return_value = []
    with patch("IkaModel.chat_interface.chat_request._api_request_with_context_fallback", return_value=wrong_shape):
        with pytest.raises(IkaProviderResponseError, match="response JSON must be an object"):
            cr._request_provider_round("openai", model, [], history, timeout=1, client=None)

    malformed = MagicMock()
    malformed.raise_for_status = MagicMock()
    malformed.json.return_value = {"choices": []}
    with patch("IkaModel.chat_interface.chat_request._api_request_with_context_fallback", return_value=malformed):
        with pytest.raises(IkaProviderResponseError, match="Failed to parse openai response"):
            cr._request_provider_round("openai", model, [], history, timeout=1, client=None)


def test_async_provider_round_wraps_malformed_response_as_domain_error():
    async def run_case():
        model = _model()
        history = _history()
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": []}
        with patch(
            "IkaModel.chat_interface.chat_request._api_request_with_context_fallback_async",
            return_value=response,
        ):
            with pytest.raises(IkaProviderResponseError, match="Failed to parse openai response"):
                await cr._request_provider_round_async(
                    "openai",
                    model,
                    [],
                    history,
                    timeout=1,
                    client=MagicMock(),
                )

    asyncio.run(run_case())


def test_chat_and_async_chat_dispatch_loop_flush_and_finalize_paths():
    model = _model()
    history = _history()
    state = cr._new_chat_state("openai", "content", None, [], 1, {"total_tokens": 1})
    runtime = ToolRuntimeState.from_model(model)

    with patch("IkaModel.chat_interface.chat_runtime._prepare_chat_inputs", return_value=(history, {})):
        with patch("IkaModel.chat_interface.chat_runtime._summarize_sync_if_near_budget") as summarize:
            with patch("IkaModel.chat_interface.chat_runtime._provider_for_model", return_value="openai"):
                with patch("IkaModel.chat_interface.chat_runtime._request_provider_round", return_value=("content", None, [], 1, {"total_tokens": 1})):
                    with patch("IkaModel.chat_interface.chat_runtime._new_chat_state", return_value=state):
                        with patch("IkaModel.chat_interface.chat_runtime.ToolRuntimeState.from_model", return_value=runtime):
                            with patch("IkaModel.chat_interface.chat_runtime._sync_tool_loop", return_value=None):
                                with patch("IkaModel.chat_interface.chat_runtime._flush_sync_remaining_tools", return_value=None):
                                    out = cr.chat(model, [{"role": "user", "content": "hi"}], message_history=history)

    assert out["content"] == "content"
    summarize.assert_called_once()
    assert history["first_input"]["message"] == "hi"

    async def run_async_case():
        async_model = _model()
        async_history = _history()
        async_state = cr._new_chat_state("openai", "async content", None, [], 1, {"total_tokens": 1})
        async_runtime = ToolRuntimeState.from_model(async_model)
        with patch("IkaModel.chat_interface.chat_runtime._prepare_chat_inputs", return_value=(async_history, {})):
            with patch("IkaModel.chat_interface.chat_runtime._summarize_async_if_near_budget") as async_summarize:
                with patch("IkaModel.chat_interface.chat_runtime._provider_for_model", return_value="openai"):
                    with patch(
                        "IkaModel.chat_interface.chat_runtime._initial_async_chat_state",
                        return_value=(async_state, async_runtime),
                    ):
                        with patch("IkaModel.chat_interface.chat_runtime._async_tool_loop", return_value=None):
                            with patch("IkaModel.chat_interface.chat_runtime._flush_async_remaining_tools", return_value=None):
                                out = await cr.async_chat(
                                    async_model,
                                    [{"role": "user", "content": "hi"}],
                                    message_history=async_history,
                                    client=MagicMock(),
                                )
        return out, async_history, async_summarize

    async_out, async_history, async_summarize = asyncio.run(run_async_case())
    assert async_out["content"] == "async content"
    assert async_history["first_input"]["message"] == "hi"
    async_summarize.assert_called_once()
