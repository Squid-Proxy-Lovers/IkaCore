"""Tests for OpenRouter API integration."""
import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from IkaModel.base import BareBoneModel, AgentTool, ToolArgs
from IkaModel.request_interface import get_provider
from IkaModel.openrouter.openrouter import openrouter_fill_payload
from IkaModel.openrouter.chat_helpers_openrouter import (
    build_openrouter_request,
    parse_openrouter_response,
    append_openrouter_tool_messages
)
from IkaModel.chat_helpers_common import build_provider_request


class TestOpenRouterProviderDetection:
    """Test OpenRouter model identification."""

    def test_detects_llama_model(self):
        assert get_provider("meta-llama/llama-3.1-70b-instruct") == "openrouter"

    def test_detects_qwen_model(self):
        assert get_provider("qwen/qwen-2.5-72b-instruct") == "openrouter"

    def test_does_not_detect_direct_openai(self):
        assert get_provider("gpt-4o") == "openai_responses"

    def test_claude_detected_as_anthropic(self):
        # Slash-delimited model ids are treated as OpenRouter routes.
        assert get_provider("anthropic/claude-3.5-sonnet") == "openrouter"


class TestOpenRouterPayloadBuilder:
    """Test payload construction."""

    def test_basic_payload_structure(self):
        model = BareBoneModel(
            model_id="meta-llama/llama-3.1-70b-instruct",
            api_key="test-key",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            suppress_init_output=True
        )
        messages = [{"role": "user", "content": "Hello"}]
        payload = openrouter_fill_payload(model, messages, None)

        assert payload["model"] == "meta-llama/llama-3.1-70b-instruct"
        assert "messages" in payload
        assert payload["messages"][0]["content"] == "Hello"

    def test_payload_with_tools(self):
        agent_end = AgentTool(
            "agent_end", "agent_end", "End with answer",
            ToolArgs(type="input", description="Final answer"),
            required=True
        )
        model = BareBoneModel(
            model_id="meta-llama/llama-3.1-70b-instruct",
            api_key="test-key",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            suppress_init_output=True
        )
        model.agent_tools = [agent_end]
        payload = openrouter_fill_payload(model, [{"role": "user", "content": "Test"}], None)

        assert "tools" in payload
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "agent_end"


class TestOpenRouterRequestBuilder:
    """Test request construction."""

    def test_basic_request_headers(self):
        model = BareBoneModel(
            model_id="meta-llama/llama-3.1-70b-instruct",
            api_key="sk-or-test-123",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            suppress_init_output=True
        )
        messages = [{"role": "user", "content": "Test"}]
        history = {
            "system": {"message": ""},
            "first_input": {"message": ""},
            "summary": {"message": ""},
            "messages": {}
        }

        api_url, headers, payload = build_openrouter_request(model, messages, history)

        assert api_url == "https://openrouter.ai/api/v1/chat/completions"
        assert headers["Authorization"] == "Bearer sk-or-test-123"
        assert headers["Content-Type"] == "application/json"


class TestOpenRouterResponseParser:
    """Test response parsing."""

    def test_parse_basic_response(self):
        response_data = {
            "id": "gen-abc123",
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help?"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        content, reasoning, tools, tokens = parse_openrouter_response(
            response_data, "meta-llama/llama-3.1-70b-instruct"
        )

        assert content == "Hello! How can I help?"
        assert reasoning is None
        assert tools == []
        assert tokens == 15

    def test_parse_response_with_tool_calls(self):
        response_data = {
            "id": "gen-abc123",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "agent_end",
                            "arguments": '{"input": "Final answer"}'
                        }
                    }]
                }
            }],
            "usage": {"total_tokens": 50}
        }

        content, reasoning, tools, tokens = parse_openrouter_response(
            response_data, "meta-llama/llama-3.1-70b-instruct"
        )

        assert content == ""
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "agent_end"


class TestOpenRouterProviderRouting:
    """Test provider routing integration."""

    def test_build_provider_request_routes_correctly(self):
        model = BareBoneModel(
            model_id="meta-llama/llama-3.1-70b-instruct",
            api_key="test-key",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            suppress_init_output=True
        )
        messages = [{"role": "user", "content": "Test"}]
        history = {
            "system": {"message": ""},
            "first_input": {"message": ""},
            "summary": {"message": ""},
            "messages": {}
        }

        api_url, headers, payload = build_provider_request(
            "openrouter", model, messages, history
        )

        assert "Authorization" in headers
        assert "model" in payload
