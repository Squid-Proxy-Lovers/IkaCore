"""Compatibility contracts for Z.AI's OpenAI-style Chat Completions API."""

from types import SimpleNamespace

from IkaModel.openai.chat_helpers_openai import parse_openai_response
from IkaModel.openai.openai import openai_fill_payload
from IkaModel.request_interface import get_provider


def _history():
    return {
        "system": {"message": "system", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def test_zai_chat_completion_url_uses_openai_chat_protocol():
    endpoint = "https://api.z.ai/api/coding/paas/v4/chat/completions"

    assert get_provider("glm-5.3", endpoint) == "openai"


def test_zai_payload_enables_thinking_and_uses_glm_output_limit():
    model = SimpleNamespace(
        model_id="glm-5.3",
        api_url="https://api.z.ai/api/coding/paas/v4/chat/completions",
        max_tokens=50000,
        temperature=0.0,
        agent_tools=[],
        reasoning_effort="high",
        parallel_tool_calls=False,
        forced_tool_name=None,
    )

    payload = openai_fill_payload(
        model,
        [{"role": "user", "content": "inspect the target"}],
        _history(),
    )

    assert payload["thinking"] == {"type": "enabled"}
    assert payload["max_tokens"] == 50000
    assert "reasoning_effort" not in payload


def test_zai_reasoning_content_is_preserved_for_tool_turns():
    content, reasoning, tool_calls, tokens = parse_openai_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "Inspect the parser first.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
            "usage": {"total_tokens": 42},
        },
        "glm-5.3",
    )

    assert content == ""
    assert reasoning == "Inspect the parser first."
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert tokens == 42
