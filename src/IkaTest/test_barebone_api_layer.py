"""
Validate BareBoneModel through the API request/response layer:
request building (tools, history, messages), response parsing (content, tool_calls, usage),
message_history updates, and chat() return shape (content, history, costs, answers).
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from IkaCore.logging_utils import IkaLogger
from IkaModel.base import AgentEndException, AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_helpers_common import (
    append_provider_tool_messages,
    build_provider_request,
    parse_provider_response,
)
from IkaModel.chat_interface.chat_interface import (
    chat,
    get_total_tokens,
    init_message_history,
)
from IkaModel.chat_interface.response_interface import extract_usage, format_gemini_results
from IkaModel.request_interface import get_max_tokens, get_provider

PROVIDERS = ("openai", "deepseek", "anthropic", "gemini", "openrouter")


def _barebone_with_tools():
    agent_end = AgentTool(
        "agent_end", "agent_end", "End with final answer",
        ToolArgs(type="input", description="Final answer"),
        required=True, limit_calls=1,
    )
    m = BareBoneModel(
        model_id="gpt-4o",
        api_key="test-key",
        api_url="https://api.openai.com/v1/chat/completions",
        system_prompt="You are helpful.",
        content_prompt="",
        max_tokens=4096,
        temperature=0,
        suppress_init_output=True,
    )
    m.agent_tools = [agent_end]
    return m


def _barebone_for_provider(provider: str):
    agent_end = AgentTool(
        "agent_end", "agent_end", "End",
        ToolArgs(type="input", description="Final answer"),
        required=True, limit_calls=1,
    )
    urls = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "deepseek": "https://api.deepseek.com/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5:generateContent",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }
    model_ids = {"openai": "gpt-4o", "deepseek": "deepseek-chat", "anthropic": "claude-3-sonnet", "gemini": "gemini-1.5-pro", "openrouter": "meta-llama/llama-3.1-70b-instruct"}
    m = BareBoneModel(
        model_id=model_ids[provider],
        api_key="test-key",
        api_url=urls[provider],
        system_prompt="Help.",
        content_prompt="",
        max_tokens=4096,
        temperature=0,
        suppress_init_output=True,
    )
    m.agent_tools = [agent_end]
    return m


class TestBareBoneToRequestPayload:
    def test_build_request_has_model_and_messages(self):
        model = _barebone_with_tools()
        history = init_message_history()
        history["system"]["message"] = "System"
        history["first_input"]["message"] = "First"
        messages = [{"role": "user", "content": "Hello"}]
        url, headers, payload = build_provider_request("openai", model, messages, history)
        assert url == model.api_url
        assert "Authorization" in headers
        assert "Bearer test-key" in headers["Authorization"]
        assert payload["model"] == model.model_id
        assert "messages" in payload
        msgs = payload["messages"]
        assert any(m.get("role") == "system" and m.get("content") == "System" for m in msgs)
        assert any(m.get("role") == "user" and m.get("content") == "First" for m in msgs)
        assert any(m.get("role") == "user" and m.get("content") == "Hello" for m in msgs)

    def test_build_request_includes_tools_and_tool_choice(self):
        model = _barebone_with_tools()
        history = init_message_history()
        messages = [{"role": "user", "content": "Hi"}]
        _, _, payload = build_provider_request("openai", model, messages, history)
        assert "tools" in payload
        tools = payload["tools"]
        assert len(tools) >= 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "agent_end"
        assert "parameters" in tools[0]["function"]
        assert "tool_choice" in payload
        assert payload["tool_choice"]["function"]["name"] == "agent_end"

    def test_build_request_summary_in_messages(self):
        model = _barebone_with_tools()
        history = init_message_history()
        history["summary"]["message"] = "[SUMMARY]\nPrior context"
        messages = [{"role": "user", "content": "Next"}]
        _, _, payload = build_provider_request("openai", model, messages, history)
        msgs = payload["messages"]
        assert any(m.get("role") == "assistant" and "Prior context" in (m.get("content") or "") for m in msgs)


class TestAllProvidersRequestBuild:
    def test_build_request_deepseek(self):
        model = _barebone_for_provider("deepseek")
        history = init_message_history()
        history["system"]["message"] = "Sys"
        messages = [{"role": "user", "content": "Hi"}]
        url, headers, payload = build_provider_request("deepseek", model, messages, history)
        assert "deepseek.com" in url
        assert "Authorization" in headers and "Bearer" in headers["Authorization"]
        assert payload["model"] == model.model_id
        assert "messages" in payload
        if model.agent_tools:
            assert "tools" in payload

    def test_build_request_anthropic(self):
        model = _barebone_for_provider("anthropic")
        history = init_message_history()
        messages = [{"role": "user", "content": "Hi"}]
        url, headers, payload = build_provider_request("anthropic", model, messages, history)
        assert "anthropic.com" in url
        assert "x-api-key" in headers
        assert payload["model"] == model.model_id
        assert "messages" in payload or "system" in payload

    def test_build_request_gemini(self):
        model = _barebone_for_provider("gemini")
        history = init_message_history()
        messages = [{"role": "user", "content": "Hi"}]
        url, headers, payload = build_provider_request("gemini", model, messages, history)
        assert "generativelanguage.googleapis.com" in url
        assert "key=" not in url
        assert headers["x-goog-api-key"] == model.api_key
        assert "contents" in payload or "generationConfig" in payload

    def test_build_request_openrouter(self):
        model = _barebone_for_provider("openrouter")
        history = init_message_history()
        messages = [{"role": "user", "content": "Hi"}]
        url, headers, payload = build_provider_request("openrouter", model, messages, history)
        assert "openrouter.ai" in url
        assert "Authorization" in headers and "Bearer" in headers["Authorization"]
        assert payload["model"] == model.model_id
        assert "messages" in payload
        if model.agent_tools:
            assert "tools" in payload


class TestApiResponseParsing:
    def test_parse_openai_content_and_usage(self):
        data = {
            "choices": [{"message": {"content": "Hello back", "tool_calls": []}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        content, reasoning, tool_calls, tokens = parse_provider_response("openai", data, "gpt-4o")
        assert content == "Hello back"
        assert reasoning is None
        assert tool_calls == []
        assert tokens == 8
        usage = extract_usage("openai", data)
        assert usage["input_tokens"] == 5
        assert usage["output_tokens"] == 3
        assert usage["total_tokens"] == 8

    def test_parse_openai_tool_calls(self):
        data = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "agent_end", "arguments": '{"input": "Done"}'}},
                    ],
                },
            }],
            "usage": {"total_tokens": 20},
        }
        content, _, tool_calls, tokens = parse_provider_response("openai", data, "gpt-4o")
        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "agent_end"
        assert "input" in json.loads(tool_calls[0]["function"]["arguments"])
        assert tokens == 20

    def test_parse_anthropic_usage(self):
        data = {
            "content": [{"type": "text", "text": "Claude says hi"}],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }
        content, _, tool_calls, _ = parse_provider_response("anthropic", data, "claude-3")
        assert content == "Claude says hi"
        usage = extract_usage("anthropic", data)
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 4
        assert usage["total_tokens"] == 14

    def test_parse_deepseek_reasoning_content(self):
        data = {
            "choices": [{"message": {"content": "Answer", "reasoning_content": "Think step by step", "tool_calls": []}}],
            "usage": {"total_tokens": 15},
        }
        content, reasoning, _, tokens = parse_provider_response("deepseek", data, "deepseek-chat")
        assert content == "Answer"
        assert reasoning == "Think step by step"
        assert tokens == 15


class TestModelCostResolution:
    def test_deepseek_v4_flash_has_pricing(self):
        assert IkaLogger.get_model_cost("deepseek-v4-flash") == IkaLogger.get_model_cost("deepseek-chat")

    def test_deepseek_v4_1_frontier_pricing(self):
        assert IkaLogger.get_model_cost("deepseek-v4.1-pro") == (0.435, 0.003625, 0.870)
        assert IkaLogger.get_model_cost("deepseek-v4.1-thinking") == IkaLogger.get_model_cost("deepseek-v4.1")

    def test_openai_frontier_pricing(self):
        assert IkaLogger.get_model_cost("gpt-5.5") == (5.00, 0.50, 30.00)
        assert IkaLogger.get_model_cost("gpt-5.5-pro") == (30.00, 30.00, 180.00)
        assert IkaLogger.get_model_cost("gpt-5.4-mini-2026-03-17") == (0.75, 0.075, 4.50)

    def test_gemini_frontier_pricing(self):
        assert IkaLogger.get_model_cost("gemini-3.5-flash") == (1.50, 0.15, 9.00)
        assert IkaLogger.get_model_cost("google/gemini-2.5-pro") == (1.25, 0.125, 10.00)

    def test_provider_prefixed_minimax_pricing(self):
        assert IkaLogger.get_model_cost("minimax/minimax-m2.7") == (0.30, 0.30, 1.20)

    def test_provider_prefixed_versioned_anthropic_pricing(self):
        assert IkaLogger.get_model_cost("us.anthropic.claude-opus-4-6-v1:0") == (5.00, 0.50, 25.00)

    def test_anthropic_frontier_pricing(self):
        assert IkaLogger.get_model_cost("claude-opus-4-8") == (5.00, 0.50, 25.00)
        assert IkaLogger.get_model_cost("us.anthropic.claude-sonnet-4-6-v1:0") == (3.00, 0.30, 15.00)

    def test_compute_cost_uses_cached_token_rate(self):
        logger = IkaLogger(level=0)
        cost = logger.compute_cost(
            "deepseek-v4-flash",
            {
                "input_tokens": 1_000_000,
                "input_cached_tokens": 250_000,
                "output_tokens": 100_000,
            },
        )
        assert cost == {
            "input_cost": 0.217,
            "output_cost": 0.042,
            "total_cost": 0.259,
        }

    def test_parse_gemini_usage(self):
        data = {
            "candidates": [{"content": {"parts": [{"text": "Gemini reply"}]}}],
            "usageMetadata": {"totalTokenCount": 7},
        }
        content, _, _, _ = parse_provider_response("gemini", data, "gemini-1.5")
        assert content == "Gemini reply"
        usage = extract_usage("gemini", data)
        assert usage["total_tokens"] == 7

    def test_parse_anthropic_tool_calls(self):
        data = {
            "content": [
                {"type": "text", "text": "Calling tool"},
                {"type": "tool_use", "id": "t1", "name": "agent_end", "input": {"input": "Done"}},
            ],
        }
        content, _, tool_calls, _ = parse_provider_response("anthropic", data, "claude-3")
        assert content == "Calling tool"
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "agent_end"
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args.get("input") == "Done"

    def test_parse_gemini_tool_calls(self):
        data = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Using tool"},
                        {"functionCall": {"name": "agent_end", "args": {"input": "Done"}}},
                    ],
                },
            }],
        }
        content, _, tool_calls, _ = parse_provider_response("gemini", data, "gemini-1.5")
        assert content == "Using tool"
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "agent_end"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"input": "Done"}

    def test_extract_usage_deepseek(self):
        data = {"usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}}
        usage = extract_usage("deepseek", data)
        assert usage["input_tokens"] == 6
        assert usage["output_tokens"] == 4
        assert usage["total_tokens"] == 10

    def test_parse_openrouter_response_and_usage(self):
        data = {
            "choices": [{"message": {"content": "OpenRouter reply", "tool_calls": []}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        content, reasoning, tool_calls, tokens = parse_provider_response("openrouter", data, "meta-llama/llama-3.1-70b-instruct")
        assert content == "OpenRouter reply"
        assert tool_calls == []
        assert tokens == 8
        usage = extract_usage("openrouter", data)
        assert usage["total_tokens"] == 8


class TestMessageHistoryUpdates:
    def test_append_tool_messages_updates_messages_and_history(self):
        messages = []
        history = init_message_history()
        executed = [{"id": "c1", "function": {"name": "agent_end", "arguments": '{"input": "Done"}'}}]
        tool_messages = [{"role": "tool", "content": "ok", "tool_call_id": "c1"}]
        tool_results = ["ok"]
        append_provider_tool_messages(
            "openai",
            messages,
            history,
            content="",
            reasoning_content=None,
            executed_tool_call_list=executed,
            tool_messages=tool_messages,
            tool_results=tool_results,
            tokens=10,
        )
        assert len(messages) >= 2
        assert messages[0]["role"] == "assistant"
        assert messages[0].get("tool_calls") == executed
        assert messages[1]["role"] == "tool"
        assert len(history["messages"]) >= 2
        types = {v.get("type") for v in history["messages"].values()}
        assert "assistant_with_tools" in types
        assert "tool" in types

    def test_append_tool_messages_deepseek(self):
        messages = []
        history = init_message_history()
        executed = [{"id": "c1", "function": {"name": "agent_end", "arguments": "{}"}}]
        tool_messages = [{"role": "tool", "content": "ok", "tool_call_id": "c1"}]
        append_provider_tool_messages(
            "deepseek", messages, history, "", None, executed, tool_messages, ["ok"], 5,
        )
        assert len(messages) >= 2
        assert messages[0]["role"] == "assistant"
        assert len(history["messages"]) >= 2

    def test_append_tool_messages_anthropic(self):
        messages = []
        history = init_message_history()
        executed = [{"id": "t1", "function": {"name": "agent_end", "arguments": "{}"}}]
        tool_messages = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]
        append_provider_tool_messages(
            "anthropic", messages, history, "", None, executed, tool_messages, ["ok"], 5,
        )
        assert len(messages) >= 1

    def test_append_tool_messages_gemini(self):
        messages = []
        history = init_message_history()
        executed = [{"name": "agent_end", "function": {"name": "agent_end", "arguments": "{}"}}]
        tool_results = ["ok"]
        append_provider_tool_messages(
            "gemini", messages, history, "", None, executed, [], tool_results, 5,
            format_gemini_results_fn=format_gemini_results,
        )
        assert len(messages) >= 1

    def test_append_tool_messages_openrouter(self):
        messages = []
        history = init_message_history()
        executed = [{"id": "c1", "function": {"name": "agent_end", "arguments": "{}"}}]
        tool_messages = [{"role": "tool", "content": "ok", "tool_call_id": "c1"}]
        append_provider_tool_messages(
            "openrouter", messages, history, "", None, executed, tool_messages, ["ok"], 5,
        )
        assert len(messages) >= 2
        assert messages[0]["role"] == "assistant"
        assert len(history["messages"]) >= 2


class TestChatReturnShape:
    def test_chat_returns_expected_keys(self):
        model = _barebone_with_tools()
        messages = [{"role": "user", "content": "Say one word"}]
        history = init_message_history()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "Hi", "tool_calls": []}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        with patch("IkaModel.chat_interface.chat_interface.api_request_retry", return_value=resp):
            out = chat(model, messages, message_history=history)
        assert "content" in out
        assert out["content"] == "Hi"
        assert "reasoning_content" in out
        assert "tool_calls" in out
        assert out["tool_calls"] == []
        assert "executed_tool_calls" in out
        assert "content_before_tools" in out
        assert "message_history" in out
        assert "usage" in out
        assert out["usage"]["total_tokens"] == 3
        assert "cost" in out
        assert "hijacked" in out
        assert out["hijacked"] is False

    def test_chat_updates_message_history_with_answer(self):
        model = _barebone_with_tools()
        messages = [{"role": "user", "content": "Reply"}]
        history = init_message_history()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "Answer here", "tool_calls": []}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
        with patch("IkaModel.chat_interface.chat_interface.api_request_retry", return_value=resp):
            out = chat(model, messages, message_history=history)
        assert out["message_history"] is history
        assert history["first_input"]["message"] == "Reply"
        assert len(history["messages"]) >= 1
        found = False
        for entry in history["messages"].values():
            if entry.get("message") and "Answer here" in str(entry["message"]):
                found = True
                break
        assert found

    def test_chat_passes_usage_to_cost_when_logger_present(self):
        model = _barebone_with_tools()
        messages = [{"role": "user", "content": "Hi"}]
        history = init_message_history()
        logger = MagicMock()
        logger.compute_cost.return_value = {"input_cost": 0.01, "output_cost": 0.02, "total_cost": 0.03}
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "Hi", "tool_calls": []}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        with patch("IkaModel.chat_interface.chat_interface.api_request_retry", return_value=resp):
            out = chat(model, messages, message_history=history, logger=logger)
        assert out["usage"]["total_tokens"] == 15
        assert out["cost"] == {"input_cost": 0.01, "output_cost": 0.02, "total_cost": 0.03}
        assert logger.compute_cost.call_count >= 1
        call_args = logger.compute_cost.call_args[0]
        assert call_args[0] == model.model_id
        assert call_args[1]["total_tokens"] == 15


class TestChatWithToolCallsFlow:
    def test_chat_with_tool_executors_returns_executed_tool_calls_and_updates_history(self):
        model = _barebone_with_tools()
        messages = [{"role": "user", "content": "End now"}]
        history = init_message_history()
        tool_executors = {"agent_end": lambda args: "Agent execution ended successfully."}
        first_resp = MagicMock()
        first_resp.status_code = 200
        first_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "agent_end", "arguments": '{"input": "Final answer"}'}}],
                },
            }],
            "usage": {"total_tokens": 25},
        }
        with patch("IkaModel.chat_interface.chat_interface.api_request_retry", return_value=first_resp):
            with pytest.raises(AgentEndException) as excinfo:
                chat(model, messages, message_history=history, tool_executors=tool_executors)
        out = excinfo.value.response
        assert out["content_before_tools"] == ""
        assert len(out["executed_tool_calls"]) == 1
        assert out["executed_tool_calls"][0]["function"]["name"] == "agent_end"
        assert out["tool_calls"] == []
        assert out["content"] == ""
        assert len(history["messages"]) >= 1


class TestAllProvidersCoverage:
    def test_build_request_succeeds_for_each_provider(self):
        for provider in PROVIDERS:
            model = _barebone_for_provider(provider)
            history = init_message_history()
            messages = [{"role": "user", "content": "Hi"}]
            url, headers, payload = build_provider_request(provider, model, messages, history)
            assert url
            assert isinstance(headers, dict)
            assert isinstance(payload, dict)

    def test_parse_response_succeeds_for_each_provider(self):
        openai_data = {"choices": [{"message": {"content": "x", "tool_calls": []}}], "usage": {"total_tokens": 1}}
        anthropic_data = {"content": [{"type": "text", "text": "x"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
        deepseek_data = {"choices": [{"message": {"content": "x", "tool_calls": []}}], "usage": {"total_tokens": 1}}
        gemini_data = {"candidates": [{"content": {"parts": [{"text": "x"}]}}], "usageMetadata": {"totalTokenCount": 1}}
        openrouter_data = {"choices": [{"message": {"content": "x", "tool_calls": []}}], "usage": {"total_tokens": 1}}
        data_by_provider = {"openai": openai_data, "anthropic": anthropic_data, "deepseek": deepseek_data, "gemini": gemini_data, "openrouter": openrouter_data}
        for provider in PROVIDERS:
            content, _, tool_calls, tokens = parse_provider_response(provider, data_by_provider[provider], "model")
            assert content == "x"
            assert isinstance(tool_calls, list)
            usage = extract_usage(provider, data_by_provider[provider])
            assert "total_tokens" in usage

    def test_append_tool_messages_succeeds_for_each_provider(self):
        executed = [{"id": "c1", "function": {"name": "agent_end", "arguments": "{}"}}]
        tool_messages_openai = [{"role": "tool", "content": "ok", "tool_call_id": "c1"}]
        for provider in PROVIDERS:
            messages = []
            history = init_message_history()
            tool_messages = tool_messages_openai if provider in ("openai", "deepseek", "openrouter") else []
            tool_results = ["ok"]
            kwargs = {"executed_tool_call_list": executed, "tool_messages": tool_messages, "tool_results": tool_results, "tokens": 0}
            if provider == "gemini":
                kwargs["format_gemini_results_fn"] = format_gemini_results
            append_provider_tool_messages(provider, messages, history, "", None, **kwargs)
            assert len(messages) >= 1
            assert len(history["messages"]) >= 1


class TestRequestLayerContract:
    """Assert request layer produces payloads with required shape for each provider."""

    def test_openai_payload_has_required_keys(self):
        model = _barebone_with_tools()
        history = init_message_history()
        history["system"]["message"] = "Sys"
        _, _, payload = build_provider_request("openai", model, [{"role": "user", "content": "Hi"}], history)
        assert "model" in payload
        assert "messages" in payload
        assert isinstance(payload["messages"], list)
        assert all("role" in m and "content" in m or "role" in m for m in payload["messages"])
        if model.agent_tools:
            assert "tools" in payload
            for t in payload["tools"]:
                assert t.get("type") == "function"
                assert "function" in t and "name" in t["function"] and "parameters" in t["function"]

    def test_anthropic_payload_has_messages_or_system(self):
        model = _barebone_for_provider("anthropic")
        _, _, payload = build_provider_request("anthropic", model, [{"role": "user", "content": "Hi"}], init_message_history())
        assert "model" in payload
        assert "messages" in payload or "system" in payload

    def test_gemini_payload_has_contents_or_generation_config(self):
        model = _barebone_for_provider("gemini")
        _, _, payload = build_provider_request("gemini", model, [{"role": "user", "content": "Hi"}], init_message_history())
        assert "contents" in payload or "generationConfig" in payload or "candidates" in payload


class TestResponseParsingContract:
    """Assert response parsing returns the exact shape IkaCore expects: content, tool_calls, usage."""

    def test_parse_returns_four_tuple(self):
        data = {"choices": [{"message": {"content": "x", "tool_calls": []}}], "usage": {"total_tokens": 1}}
        out = parse_provider_response("openai", data, "gpt-4o")
        assert len(out) == 4
        content, reasoning, tool_calls, tokens = out
        assert isinstance(content, str)
        assert isinstance(tool_calls, list)
        assert isinstance(tokens, (int, float))

    def test_tool_call_item_has_function_name_and_arguments(self):
        data = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "agent_end", "arguments": '{"input": "done"}'}}],
                },
            }],
            "usage": {"total_tokens": 10},
        }
        _, _, tool_calls, _ = parse_provider_response("openai", data, "gpt-4o")
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert "function" in tc
        assert tc["function"]["name"] == "agent_end"
        assert "arguments" in tc["function"]

    def test_extract_usage_returns_normalized_dict(self):
        data = {"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
        usage = extract_usage("openai", data)
        assert set(usage.keys()) >= {"input_tokens", "output_tokens", "total_tokens"}
        assert usage["input_tokens"] == 2
        assert usage["output_tokens"] == 3
        assert usage["total_tokens"] == 5

    def test_anthropic_parse_tool_calls_have_function_with_arguments_string(self):
        data = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "my_tool", "input": {"x": 1}},
            ],
        }
        _, _, tool_calls, _ = parse_provider_response("anthropic", data, "claude-3")
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "my_tool"
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args == {"x": 1}

    def test_gemini_parse_tool_calls_have_name_and_arguments(self):
        data = {
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "fn", "args": {"a": "b"}}}],
                },
            }],
        }
        _, _, tool_calls, _ = parse_provider_response("gemini", data, "gemini-1.5")
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "fn"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"a": "b"}

    def test_openrouter_usage_fallback_total_tokens(self):
        data = {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 12}}
        usage = extract_usage("openrouter", data)
        assert usage["total_tokens"] == 12


class TestProviderAndTokenHelpers:
    def test_get_provider_maps_model_id(self):
        assert get_provider("gpt-4o") == "openai_responses"
        assert get_provider("deepseek-chat") == "deepseek"
        assert get_provider("claude-3-sonnet") == "anthropic"
        assert get_provider("gemini-1.5-pro") == "gemini"
        assert get_provider("meta-llama/llama-3.1-70b-instruct") == "openrouter"
        assert get_provider("gpt-4o", "https://api.openai.com/v1/chat/completions") == "openai"
        assert get_provider("gpt-4o", "https://api.openai.com/v1/responses") == "openai_responses"

    def test_get_max_tokens_returns_positive(self):
        assert get_max_tokens("gpt-4o") > 0
        assert get_max_tokens("claude-3-sonnet") > 0

    def test_get_max_tokens_prefers_specific_model_slug(self):
        assert get_max_tokens("gpt-5.3-codex") == 400000
        assert get_max_tokens("gpt-5.5-pro") == 1050000
        assert get_max_tokens("gpt-5.4-mini") == 400000
        assert get_max_tokens("claude-opus-4-8") == 1000000
        assert get_max_tokens("gemini-3.5-flash") == 1048576
        assert get_max_tokens("deepseek-v4.1-pro") == 1000000

    def test_init_message_history_structure(self):
        h = init_message_history()
        assert "system" in h and "first_input" in h and "summary" in h and "messages" in h
        assert h["messages"] == {}

    def test_get_total_tokens_uses_stored_and_estimated(self):
        h = init_message_history()
        h["system"]["tokens"] = 100
        h["first_input"]["message"] = "hello"
        h["first_input"]["tokens"] = 0
        total = get_total_tokens(h)
        assert total >= 100
