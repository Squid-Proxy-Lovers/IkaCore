"""Regression tests for broad cleanup hardening."""

import asyncio
import json
import os
import sqlite3
import stat
import subprocess
import sys
from contextlib import closing
from importlib.resources import files
from unittest.mock import MagicMock, patch

import httpx
import pytest

from IkaCore.agents import IkaBaseAgent
from IkaCore.checkpoint import CheckpointStore
from IkaCore.logging_utils import IkaLogger
from IkaMem.long_term_memory import LTMemory
from IkaMem.short_term_memory import STMemory
from IkaModel import model_metadata
from IkaModel.base import AgentEndException, AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_interface import chat_runtime
from IkaModel.chat_interface.chat_interface import async_chat, chat, init_message_history
from IkaModel.chat_interface.response_interface import (
    async_execute_tool_calls,
    execute_tool_calls,
    format_gemini_results,
)
from IkaModel.chat_interface.types import ChatLoopState, ChatResponsePayload, ToolRuntimeState, UsageInfo
from IkaModel.codex import auth as codex_auth
from IkaModel.gemini.chat_helpers_gemini import build_gemini_request
from IkaModel.request_interface import (
    IkaAPIError,
    IkaRateLimitError,
    IkaTimeoutError,
    _redact_headers,
    _redact_url,
    api_request_retry,
    model_for_payload,
)
from IkaModel.runtime_errors import IkaContextWindowError, IkaProviderPayloadError
from IkaModel.tool_schema import build_provider_tool_payload, build_tool_parameters


def test_public_package_imports_work_in_clean_subprocess():
    env = os.environ.copy()
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import IkaCore, IkaModel; "
        "from IkaCore import IkaBaseAgent; "
        "from IkaModel import BareBoneModel; "
        "from IkaModel.base import AgentTool; "
        "print(IkaBaseAgent.__name__, BareBoneModel.__name__, AgentTool.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "IkaBaseAgent BareBoneModel AgentTool" in result.stdout


def test_short_term_search_tool_is_exposed_when_executor_is_available():
    agent = IkaBaseAgent(
        name="A",
        description="D",
        prompt="P",
        model_id="gpt-4o",
        api_key="k",
        memory=True,
        memory_access={"short_term_save": True, "short_term_search": True},
    )
    agent.short_term_memory = MagicMock()

    exposed = {tool.name for tool in agent._build_memory_tools(agent.memory_access, "short_term")}
    executors = agent.build_tool_executors([], memory_access=agent.memory_access)

    assert {"short_term_save", "short_term_search"}.issubset(exposed)
    assert {"short_term_save", "short_term_search"}.issubset(executors)


def test_async_tool_calls_return_schema_hint_for_empty_required_args():
    calls = []

    def executor(args):
        calls.append(args)
        return "should not execute"

    tool_calls = [{"id": "call_1", "function": {"name": "needs_query", "arguments": "{}"}}]
    metadata = {
        "needs_query": {
            "parameters": {
                "properties": {"query": {"type": "string", "description": "Query text"}},
                "required": ["query"],
            }
        }
    }

    _, tool_results, counts, _, _ = asyncio.run(
        async_execute_tool_calls(
            tool_calls,
            {"needs_query": executor},
            "openai",
            tool_metadata=metadata,
        )
    )

    assert calls == []
    assert counts["needs_query"] == 1
    assert "Expected schema" in tool_results[0]
    assert "query" in tool_results[0]


def test_chat_runtime_types_preserve_public_dict_shape():
    usage = UsageInfo.from_mapping(
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3, "input_cached_tokens": 4}
    )
    usage.add_mapping({"input_tokens": 5, "total_tokens": 6})
    state = ChatLoopState(
        provider="openai",
        content="content",
        reasoning_content=None,
        tool_calls=[],
        tokens=9,
        content_before_tools="content",
        usage=usage,
    )
    state.add_usage({"output_tokens": 7, "total_tokens": 8})
    payload = ChatResponsePayload(
        content=state.content,
        reasoning_content=state.reasoning_content,
        tool_calls=state.tool_calls,
        executed_tool_calls=[],
        content_before_tools=state.content_before_tools,
        message_history=init_message_history(),
        usage=state.usage,
        cost={"total_cost": 0.0},
        hijacked=False,
    )

    out = payload.to_dict()

    assert out["content"] == "content"
    assert out["usage"] == {
        "input_tokens": 6,
        "output_tokens": 9,
        "total_tokens": 17,
        "input_cached_tokens": 4,
    }
    assert out["cost"] == {"total_cost": 0.0}
    assert out["interrupted"] is False
    assert out["interrupt_data"] is None


def test_tool_runtime_state_syncs_compatibility_attributes():
    model = BareBoneModel(
        model_id="gpt-4o",
        api_key="k",
        api_url="https://api.openai.com/v1/chat/completions",
        suppress_init_output=True,
        use_responses_api=False,
    )
    model._tool_call_counts = {"search": 1}
    model._recent_tool_calls = [("search", '{"q":"a"}')]
    model._current_step = 2

    state = ToolRuntimeState.from_model(model)
    state.apply_updated_counts({"search": 2, "fetch": 1})
    state.record_executed([{"function": {"name": "fetch", "arguments": "{}"}}])
    state.recent_tool_calls.append(("fetch", "{}"))
    state.sync_to_model(model)

    assert state.current_step == 3
    assert state.total_tool_calls_in_cycle == 1
    assert model._tool_call_counts == {"search": 2, "fetch": 1}
    assert model._recent_tool_calls == [("search", '{"q":"a"}'), ("fetch", "{}")]
    assert model._current_step == 3


def test_chat_context_fallback_classifies_initial_payload_build_failure():
    model = BareBoneModel(
        model_id="gpt-4o",
        api_key="k",
        api_url="https://api.openai.com/v1/chat/completions",
        suppress_init_output=True,
        use_responses_api=False,
    )

    def build_payload():
        raise KeyError("missing required field")

    with pytest.raises(IkaProviderPayloadError, match="provider request payload"):
        chat_runtime._api_request_with_context_fallback(build_payload, model, init_message_history())


def test_chat_context_fallback_classifies_rebuild_failure_after_summarization():
    model = BareBoneModel(
        model_id="gpt-4o",
        api_key="k",
        api_url="https://api.openai.com/v1/chat/completions",
        suppress_init_output=True,
        use_responses_api=False,
    )
    calls = {"count": 0}

    def build_payload():
        calls["count"] += 1
        if calls["count"] == 1:
            return "https://example.test", {}, {}
        raise ValueError("bad rebuilt payload")

    with patch(
        "IkaModel.chat_interface.chat_request.api_request_retry",
        side_effect=IkaAPIError("maximum context length exceeded"),
    ):
        with patch("IkaModel.chat_interface.chat_request.summarise_message_history", return_value="summary"):
            with pytest.raises(IkaContextWindowError, match="rebuild provider request"):
                chat_runtime._api_request_with_context_fallback(build_payload, model, init_message_history())


def test_sync_tool_calls_reject_malformed_json_without_execution():
    calls = []

    def executor(args):
        calls.append(args)
        return "should not execute"

    _, tool_results, counts, executed, interrupt = execute_tool_calls(
        [{"id": "call_1", "function": {"name": "needs_query", "arguments": '{"query":'}}],
        {"needs_query": executor},
        "openai",
    )

    assert calls == []
    assert counts == {}
    assert interrupt is None
    assert executed[0]["function"]["name"] == "needs_query"
    assert "Malformed JSON" in tool_results[0]


def test_sync_tool_calls_skip_duplicate_arguments_after_first_execution():
    calls = []

    def executor(args):
        calls.append(args)
        return {"ok": args["query"]}

    tool_call = {"function": {"name": "search", "arguments": '{"query":"x"}'}}
    _, tool_results, counts, _, interrupt = execute_tool_calls(
        [{"id": "call_1", **tool_call}, {"id": "call_2", **tool_call}],
        {"search": executor},
        "openai",
    )

    assert calls == [{"query": "x"}]
    assert counts == {"search": 1}
    assert interrupt is None
    assert json.loads(tool_results[0]) == {"ok": "x"}
    assert "Duplicate tool call" in tool_results[1]


def test_async_repeated_tool_guard_returns_forced_completion_without_execution():
    model = BareBoneModel(
        model_id="gpt-4o",
        api_key="k",
        api_url="https://api.openai.com/v1/chat/completions",
        suppress_init_output=True,
        use_responses_api=False,
    )
    model.agent_tools = []
    model._recent_tool_calls = [("loop", "{}")] * 5

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"id": "call_1", "function": {"name": "loop", "arguments": "{}"}}],
                }
            }
        ],
        "usage": {"total_tokens": 1},
    }
    calls = []

    def executor(args):
        calls.append(args)
        return "should not execute"

    async def run_case():
        with patch("IkaModel.chat_interface.chat_interface.async_api_request_retry", return_value=response):
            with patch(
                "IkaModel.chat_interface.chat_interface.async_summarise_message_history",
                return_value="forced final",
            ):
                return await async_chat(
                    model,
                    [{"role": "user", "content": "hi"}],
                    message_history=init_message_history(),
                    tool_executors={"loop": executor},
                    max_tool_rounds=1,
                )

    out = asyncio.run(run_case())

    assert calls == []
    assert out["hijacked"] is True
    assert "forced final" in out["content"]


def test_sync_max_tool_calls_force_completes_with_synthetic_agent_end():
    model = BareBoneModel(
        model_id="gpt-4o",
        api_key="k",
        api_url="https://api.openai.com/v1/chat/completions",
        suppress_init_output=True,
        use_responses_api=False,
    )
    model.agent_tools = [AgentTool("loop", "loop", "Loop", ToolArgs("object", "payload"))]

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "before tools",
                    "tool_calls": [{"id": "call_1", "function": {"name": "loop", "arguments": "{}"}}],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }

    with patch("IkaModel.chat_interface.chat_interface.api_request_retry", return_value=response):
        with patch("IkaModel.chat_interface.chat_interface.run_summarization", return_value="forced done"):
            with pytest.raises(AgentEndException) as exc_info:
                chat(
                    model,
                    [{"role": "user", "content": "hi"}],
                    message_history=init_message_history(),
                    tool_executors={"loop": lambda _args: "ok"},
                    max_tool_calls=1,
                )

    out = exc_info.value.response
    assert out["hijacked"] is True
    assert out["content"] == "forced done"
    assert out["usage"]["total_tokens"] == 3
    assert [call["function"]["name"] for call in out["executed_tool_calls"]] == ["loop", "agent_end"]


def test_async_chat_preserves_caller_owned_client_lifecycle():
    model = BareBoneModel(
        model_id="gpt-4o",
        api_key="k",
        api_url="https://api.openai.com/v1/chat/completions",
        suppress_init_output=True,
        use_responses_api=False,
    )
    model.agent_tools = []

    async def run_case():
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "done", "tool_calls": []}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        )
        client = httpx.AsyncClient(transport=transport)
        try:
            out = await async_chat(
                model,
                [{"role": "user", "content": "hi"}],
                message_history=init_message_history(),
                client=client,
            )
            assert out["content"] == "done"
            assert out["usage"]["total_tokens"] == 2
            assert client.is_closed is False
        finally:
            await client.aclose()
        assert client.is_closed is True

    asyncio.run(run_case())


def test_gemini_api_key_uses_header_and_redaction_hides_secrets():
    model = BareBoneModel(
        model_id="gemini-1.5-pro",
        api_key="secret-key",
        api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        suppress_init_output=True,
    )
    model.agent_tools = []

    url, headers, _ = build_gemini_request(model, [{"role": "user", "content": "hi"}], init_message_history())

    assert "secret-key" not in url
    assert headers["x-goog-api-key"] == "secret-key"
    assert _redact_headers(headers)["x-goog-api-key"] == "***"
    assert _redact_headers({"Authorization": "Bearer abc"})["Authorization"] == "Bearer ***"
    redacted_url = _redact_url("https://example.test/path?key=secret&safe=value&token=t")
    assert "secret" not in redacted_url
    assert "token=t" not in redacted_url
    assert "safe=value" in redacted_url


def test_checkpoint_database_is_owner_private(tmp_path):
    if os.name == "nt":
        return
    db_path = tmp_path / "checkpoints.db"
    CheckpointStore(str(db_path))

    mode = stat.S_IMODE(os.stat(db_path).st_mode)
    assert mode == 0o600


def test_corrupt_checkpoint_payload_raises(tmp_path):
    db_path = tmp_path / "checkpoints.db"
    store = CheckpointStore(str(db_path))

    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            conn.execute(
                "INSERT INTO checkpoints (uid, scope, payload_json, created_at) VALUES (?, ?, ?, ?)",
                ("bad-json", "test", "{not-json", "2026-06-05T00:00:00+00:00"),
            )

    with pytest.raises(ValueError, match="invalid JSON"):
        store.load_checkpoint("bad-json")


def test_agent_end_empty_functions_list_is_rejected():
    agent = IkaBaseAgent(
        name="A",
        description="D",
        prompt="P",
        model_id="gpt-4o",
        api_key="k",
    )

    with pytest.raises(ValueError, match="empty functions list"):
        agent.parse_control_calls(
            [{"function": {"name": "agent_end", "arguments": '{"input":"{\\"functions\\":[]}"}'}}],
            None,
        )


def test_long_term_filter_failure_is_reported():
    agent = IkaBaseAgent(
        name="A",
        description="D",
        prompt="P",
        model_id="gpt-4o",
        api_key="k",
    )
    agent.long_term_memory = MagicMock()
    agent.long_term_memory.search.return_value = [{"task": "task", "expected_output": "output"}]

    result = agent._search_long_term("query", filter_func=lambda _results: (_ for _ in ()).throw(RuntimeError("boom")))

    assert "long-term memory filter failed: boom" == result["error"]


def test_level_zero_logger_stays_lazy_until_enabled():
    logger = IkaLogger(level=0)

    logger.write_line("not written")
    logger.log_json({"not": "written"})
    logger.flush()
    logger.shutdown()

    assert logger._writer_thread is None
    assert logger._queue is None


def test_model_metadata_is_loaded_from_packaged_json():
    data = json.loads(files("IkaModel").joinpath("data/model_metadata.json").read_text(encoding="utf-8"))

    assert data["tokenmax_mapping"]["gpt-4o"] == model_metadata.TOKENMAX_MAPPING["gpt-4o"]
    assert tuple(data["model_costs"]["gpt-4o"]) == model_metadata.MODEL_COSTS["gpt-4o"]
    assert data["api_url_by_provider"]["codex"] == model_metadata.API_URL_BY_PROVIDER["codex"]


def test_codex_refresh_url_override_requires_explicit_allow(monkeypatch):
    monkeypatch.setenv("CODEX_REFRESH_TOKEN_URL_OVERRIDE", "https://example.test/oauth/token")
    monkeypatch.delenv("IKACORE_ALLOW_CODEX_TOKEN_URL_OVERRIDE", raising=False)

    assert codex_auth._resolve_oauth_token_url() == codex_auth.CODEX_OAUTH_TOKEN_URL

    monkeypatch.setenv("IKACORE_ALLOW_CODEX_TOKEN_URL_OVERRIDE", "1")
    assert codex_auth._resolve_oauth_token_url() == "https://example.test/oauth/token"


def test_agent_clone_for_run_isolates_runtime_history_and_client():
    agent = IkaBaseAgent(
        name="A",
        description="D",
        prompt="P",
        model_id="gpt-4o",
        api_key="k",
    )
    agent.message_history["first_input"]["message"] = "original history"
    agent._parent_hierarchy = ["root"]
    client = agent._get_chat_client()

    clone = agent.clone_for_run(name="A_clone", prompt="new prompt")

    assert clone.name == "A_clone"
    assert clone.prompt == "new prompt"
    assert clone.message_history["first_input"]["message"] == ""
    assert clone.logger is agent.logger
    assert clone.client is None
    assert agent._get_chat_client() is client

    clone.message_history["first_input"]["message"] = "clone history"
    assert agent.message_history["first_input"]["message"] == "original history"

    agent.shutdown()
    assert client.is_closed


def test_tool_schema_and_payload_filters_cache_without_mutating_source_model():
    schema_tool = AgentTool(
        "schema-tool",
        "schema_tool",
        "Schema tool",
        ToolArgs("object", "payload", properties={"value": {"type": "string"}, "__required__": ["value"]}),
    )

    first_schema = build_tool_parameters(schema_tool)
    second_schema = build_tool_parameters(schema_tool)
    assert second_schema is first_schema

    schema_tool.args.description = "new payload"
    assert build_tool_parameters(schema_tool) is not first_schema
    provider_payload = build_provider_tool_payload("openai", [schema_tool])
    assert build_provider_tool_payload("openai", [schema_tool]) is provider_payload

    always_available = AgentTool("always", "always", "Always", ToolArgs("input", "payload"))
    exhausted = AgentTool("limited", "limited", "Limited", ToolArgs("input", "payload"), limit_calls=1)
    model = BareBoneModel(
        model_id="gpt-4o",
        api_key="k",
        api_url="https://api.openai.com/v1/responses",
        suppress_init_output=True,
    )
    model.agent_tools = [always_available, exhausted]
    model._tool_call_counts = {"limited": 1}

    first_payload_model = model_for_payload(model)
    second_payload_model = model_for_payload(model)

    assert model.agent_tools == [always_available, exhausted]
    assert first_payload_model is not model
    assert first_payload_model.agent_tools == [always_available]
    assert second_payload_model.agent_tools is first_payload_model.agent_tools


def test_gemini_tool_result_formatting_skips_json_parse_for_plain_text():
    formatted = format_gemini_results(
        [{"name": "plain"}, {"name": "json"}, {"name": "dict"}],
        ["plain text", '{"ok":true}', {"already": "structured"}],
    )

    assert formatted[0]["functionResponse"]["response"] == {"result": "plain text"}
    assert formatted[1]["functionResponse"]["response"] == {"ok": True}
    assert formatted[2]["functionResponse"]["response"] == {"already": "structured"}


def test_memory_operations_do_not_write_to_stdout(capsys):
    class DummyStorage:
        def __init__(self):
            self.saved = []

        def save(self, value, metadata):
            self.saved.append((value, metadata))

        def search(self, query, limit, score_threshold):
            return [{"query": query, "limit": limit, "score_threshold": score_threshold}]

    short_memory = STMemory(storage=DummyStorage())
    long_memory = LTMemory(storage=DummyStorage())

    short_memory.save("short insight", metadata={"scope": "short"})
    assert short_memory.search("short") == [{"query": "short", "limit": 5, "score_threshold": 0.6}]

    long_memory.save("long insight", metadata={"scope": "long"})
    assert long_memory.search("long") == [{"query": "long", "limit": 10, "score_threshold": 0.6}]

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_mem0_store_missing_dependency_has_clear_error(monkeypatch):
    from IkaMem.storage import mem0_storage

    monkeypatch.setattr(mem0_storage, "_MEM0_AVAILABLE", False)

    with pytest.raises(ImportError, match="optional 'mem0' package"):
        mem0_storage.Mem0Store(memory_type="short_term")


def test_api_retry_uses_typed_rate_limit_error():
    response = httpx.Response(429, content=b"rate limit")

    with patch("IkaModel.request_interface.httpx.post", return_value=response):
        with pytest.raises(IkaRateLimitError) as exc_info:
            api_request_retry("https://example.test", {}, {}, max_retries=1)

    assert exc_info.value.status_code == 429


def test_api_retry_uses_typed_timeout_error():
    with patch("IkaModel.request_interface.httpx.post", side_effect=httpx.ReadTimeout("slow")):
        with pytest.raises(IkaTimeoutError, match="timed out"):
            api_request_retry("https://example.test", {}, {}, max_retries=1, timeout=0.01)
