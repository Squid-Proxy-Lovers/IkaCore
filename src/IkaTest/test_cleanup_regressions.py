"""Regression tests for broad cleanup hardening."""

import asyncio
import json
import os
import sqlite3
import stat
import subprocess
import sys
from importlib.resources import files
from unittest.mock import MagicMock, patch

import pytest

from IkaCore.agents import IkaBaseAgent
from IkaCore.checkpoint import CheckpointStore
from IkaCore.logging_utils import IkaLogger
from IkaMem.long_term_memory import LTMemory
from IkaMem.short_term_memory import STMemory
from IkaModel import model_metadata
from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_interface.chat_interface import async_chat, init_message_history
from IkaModel.chat_interface.response_interface import async_execute_tool_calls, format_gemini_results
from IkaModel.codex import auth as codex_auth
from IkaModel.gemini.chat_helpers_gemini import build_gemini_request
from IkaModel.request_interface import _redact_headers, _redact_url, model_for_payload
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

    with sqlite3.connect(db_path) as conn:
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
