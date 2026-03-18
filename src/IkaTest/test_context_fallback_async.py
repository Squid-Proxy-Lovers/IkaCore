import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from IkaModel.chat_interface import chat_interface
from IkaModel import summarization


class DummyModel:
    model_id = "deepseek-chat"
    api_url = "https://api.deepseek.com/chat/completions"
    api_key = "k"


def test_async_context_fallback_returns_retry_response():
    message_history = {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "task", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {"m1": {"message": "x" * 1000, "tokens": 0}},
        "compaction_count": 0,
    }

    async def run_test():
        fake_response = object()

        with (
            patch.object(chat_interface, "_is_context_length_error", return_value=True),
            patch.object(
                chat_interface,
                "async_api_request_retry",
                new=AsyncMock(side_effect=[Exception("maximum context length exceeded"), fake_response]),
            ) as retry_mock,
            patch.object(
                chat_interface,
                "async_summarise_message_history",
                new=AsyncMock(return_value="summary"),
            ) as summarise_mock,
        ):
            result = await chat_interface._api_request_with_context_fallback_async(
                lambda: ("https://api.deepseek.com/chat/completions", {}, {"messages": []}),
                DummyModel(),
                message_history,
            )

            assert result is fake_response
            assert retry_mock.await_count == 2
            summarise_mock.assert_awaited_once()

    asyncio.run(run_test())


def test_async_context_fallback_fast_path_rewrites_oversized_input():
    message_history = {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "task", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
        "compaction_count": 0,
    }
    messages = [{"role": "user", "content": "x" * 1100000}]

    async def run_test():
        fake_response = object()

        def build_payload():
            return (
                "https://api.deepseek.com/chat/completions",
                {},
                {"messages": list(messages), "max_tokens": 4096},
            )

        with (
            patch.object(
                chat_interface,
                "async_api_request_retry",
                new=AsyncMock(return_value=fake_response),
            ) as retry_mock,
            patch.object(
                chat_interface,
                "async_summarise_message_history",
                new=AsyncMock(return_value="summary"),
            ) as summarise_mock,
        ):
            result = await chat_interface._api_request_with_context_fallback_async(
                build_payload,
                DummyModel(),
                message_history,
                messages=messages,
            )

            assert result is fake_response
            assert retry_mock.await_count == 1
            summarise_mock.assert_awaited_once()
            assert len(messages) == 1
            assert "The oversized in-flight input has been deleted." in messages[0]["content"]
            assert "approximately" in messages[0]["content"]

    asyncio.run(run_test())


def test_async_context_fallback_second_overflow_rewrites_input():
    message_history = {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "task", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {"m1": {"message": "x" * 1000, "tokens": 0}},
        "compaction_count": 0,
    }
    messages = [{"role": "user", "content": "inspect a giant result"}]

    async def run_test():
        fake_response = object()
        context_error = Exception(
            "This model's maximum context length is 131072 tokens. "
            "However, you requested 139821 tokens (131629 in the messages, 8192 in the completion)."
        )

        with (
            patch.object(chat_interface, "_is_context_length_error", return_value=True),
            patch.object(
                chat_interface,
                "async_api_request_retry",
                new=AsyncMock(side_effect=[context_error, context_error, fake_response]),
            ) as retry_mock,
            patch.object(
                chat_interface,
                "async_summarise_message_history",
                new=AsyncMock(return_value="summary"),
            ) as summarise_mock,
        ):
            result = await chat_interface._api_request_with_context_fallback_async(
                lambda: ("https://api.deepseek.com/chat/completions", {}, {"messages": list(messages), "max_tokens": 8192}),
                DummyModel(),
                message_history,
                messages=messages,
            )

            assert result is fake_response
            assert retry_mock.await_count == 3
            assert summarise_mock.await_count == 2
            assert len(messages) == 1
            assert "131072 tokens" in messages[0]["content"]
            assert "131629 in messages" in messages[0]["content"]
            assert "The oversized in-flight input has been deleted." in messages[0]["content"]

    asyncio.run(run_test())


def test_tool_result_guardrail_message_for_duplicate_create_context_pack():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "create_context_pack",
                "arguments": "{\"name\":\"plan\",\"book\":\"plans\"}",
            },
        }
    ]
    tool_results = [
        "{\"status\":\"blocked_duplicate_create\",\"error\":\"CRITICAL: STOP.\","
        "\"next_required_action\":\"Use update_context_pack(name='plan', book='plans', ...) to modify it.\","
        "\"stop_repeating_tool_call\":{\"tool\":\"create_context_pack\",\"name\":\"plan\",\"book\":\"plans\",\"reason\":\"already_exists\"}}"
    ]

    msg = chat_interface._build_tool_result_guardrail_message(tool_calls, tool_results)

    assert "Do not call create_context_pack again" in msg
    assert "context pack 'plan'" in msg
    assert "update_context_pack" in msg


def test_get_conversation_text_can_exclude_first_input():
    message_history = {
        "system": {"message": "sys", "tokens": 0},
        "first_input": {"message": "original goal", "tokens": 0},
        "summary": {"message": "[SUMMARY]\nprior", "tokens": 0},
        "messages": {
            "m1": {"message": "recent assistant/tool state", "tokens": 0},
        },
    }

    included = summarization.get_conversation_text(message_history, include_first_input=True)
    excluded = summarization.get_conversation_text(message_history, include_first_input=False)

    assert "original goal" in included
    assert "original goal" not in excluded
    assert "recent assistant/tool state" in excluded


def test_rebuild_messages_after_compaction_preserves_first_input_context_packs_and_recent_tail():
    message_history = {
        "first_input": {"message": "audit v8 and verify findings", "tokens": 0},
    }
    messages = [
        {"role": "user", "content": "audit v8 and verify findings"},
        {"role": "user", "content": "[Context Pack: plans/master_plan]\nphase 1"},
        {"role": "assistant", "content": "tool caller", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "content": "{\"status\":\"ok\"}", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "latest assistant message"},
    ]

    chat_interface._rebuild_messages_after_compaction(message_history, messages)

    assert messages[0] == {"role": "user", "content": "audit v8 and verify findings"}
    assert messages[1]["content"].startswith("[Context Pack: plans/master_plan]")
    assert messages[-1]["content"] == "latest assistant message"
    assert any(message["content"] == "{\"status\":\"ok\"}" for message in messages)


def test_rebuild_messages_after_compaction_drops_orphan_tool_messages():
    message_history = {
        "first_input": {"message": "audit v8 and verify findings", "tokens": 0},
    }
    messages = [
        {"role": "user", "content": "audit v8 and verify findings"},
        {"role": "tool", "content": "{\"status\":\"ok\"}", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "latest assistant message"},
    ]

    chat_interface._rebuild_messages_after_compaction(message_history, messages)

    assert not any(message.get("role") == "tool" for message in messages)
