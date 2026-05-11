"""Tests for non-blocking HITL interrupt/resume behavior."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

mock_ika_mem = MagicMock()
with patch.dict("sys.modules", {"IkaMem": mock_ika_mem}):
    from IkaCore.agents import IkaBaseAgent
from IkaCore.checkpoint import CheckpointStore
from IkaCore.stages import IkaStage
from IkaModel.base import AgentTool, BareBoneModel, HumanInputRequired, ToolArgs
from IkaModel.chat_interface.chat_interface import async_chat, chat


def _interrupt_model() -> BareBoneModel:
    model = BareBoneModel(
        model_id="gpt-4o",
        api_key="test-key",
        api_url="https://api.openai.com/v1/responses",
        system_prompt="",
        content_prompt="",
        max_tokens=4096,
        temperature=0.0,
        use_responses_api=True,
    )
    model.agent_tools = [
        AgentTool(
            id="ask_user",
            name="ask_user",
            description="Ask the user for input.",
            args=ToolArgs(
                type="object",
                description="question payload",
                properties={
                    "question": {"type": "string", "description": "Question to ask"},
                    "__required__": ["question"],
                },
            ),
            required=False,
        )
    ]
    return model


def _hitl_response(question: str) -> dict:
    return {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": ""}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "ask_user",
                "arguments": json.dumps({"question": question}),
            },
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }


def _ask_user_executor(args: dict) -> str:
    raise HumanInputRequired(
        {
            "kind": "hitl",
            "agent_name": "Agent",
            "stage_name": "Stage 0",
            "stage_index": 0,
            "remaining_steps": 3,
            "question": args.get("question", ""),
        }
    )


class TestHitlChatInterrupt:
    def test_chat_interrupts_without_blocking(self):
        model = _interrupt_model()
        messages = [{"role": "user", "content": "Start"}]
        history = {
            "system": {"message": "", "tokens": 0},
            "first_input": {"message": "", "tokens": 0},
            "summary": {"message": "", "tokens": 0},
            "messages": {},
        }

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _hitl_response("Need more context?")

        with patch("IkaModel.chat_interface.chat_interface.api_request_retry", return_value=resp):
            out = chat(model, messages, message_history=history, tool_executors={"ask_user": _ask_user_executor})

        assert out["interrupted"] is True
        assert out["interrupt_data"]["question"] == "Need more context?"
        assert out["content"] == "Need more context?"
        assert out["executed_tool_calls"]
        assert out["executed_tool_calls"][0]["function"]["name"] == "ask_user"
        assert history["messages"]

    def test_async_chat_interrupts_without_blocking(self):
        model = _interrupt_model()
        messages = [{"role": "user", "content": "Start"}]
        history = {
            "system": {"message": "", "tokens": 0},
            "first_input": {"message": "", "tokens": 0},
            "summary": {"message": "", "tokens": 0},
            "messages": {},
        }

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _hitl_response("Need more context?")

        with patch("IkaModel.chat_interface.chat_interface.async_api_request_retry", return_value=resp):
            out = asyncio.run(
                async_chat(model, messages, message_history=history, tool_executors={"ask_user": _ask_user_executor})
            )

        assert out["interrupted"] is True
        assert out["interrupt_data"]["question"] == "Need more context?"
        assert out["content"] == "Need more context?"


class TestHitlResume:
    def test_execution_saves_checkpoint_and_resumes(self, tmp_path):
        stage = IkaStage("Stage 0", "Ask for missing details", [], hitl=True, checkpoint=True)
        agent = IkaBaseAgent(
            name="Agent",
            description="Agent desc",
            prompt="Do the task",
            model_id="gpt-4o",
            api_key="test-key",
            Stages=[stage],
            checkpoint=True,
            checkpoint_db_path=str(tmp_path / "checkpoints.db"),
        )
        agent.logger = MagicMock(level=0, use_colors=False)
        agent.logger.compute_cost.return_value = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

        interrupt_response = {
            "content": "Need more context?",
            "reasoning_content": None,
            "tool_calls": [],
            "executed_tool_calls": [],
            "content_before_tools": "",
            "message_history": agent.message_history,
            "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1, "input_cached_tokens": 0},
            "cost": {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0},
            "hijacked": False,
            "interrupted": True,
            "interrupt_data": {
                "kind": "hitl",
                "agent_name": "Agent",
                "stage_name": "Stage 0",
                "stage_index": 0,
                "remaining_steps": 99,
                "question": "Need more context?",
            },
            "status": "awaiting_user_input",
        }

        final_response = {
            "content": "All done",
            "reasoning_content": None,
            "tool_calls": [],
            "executed_tool_calls": [
                {"function": {"name": "agent_end", "arguments": '{"input": "All done"}'}}
            ],
            "content_before_tools": "All done",
            "message_history": agent.message_history,
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "input_cached_tokens": 0},
            "cost": {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0},
            "hijacked": False,
            "interrupted": False,
        }

        calls = []

        def chat_wrapper_side_effect(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 1:
                return interrupt_response
            assert args[1][0]["content"] == "User answer"
            return final_response

        agent.chat_wrapper = MagicMock(side_effect=chat_wrapper_side_effect)

        first = agent.execution()
        assert first["status"] == "awaiting_user_input"
        assert first["checkpoint_uid"]
        assert first["interrupt_data"]["question"] == "Need more context?"

        store = CheckpointStore(str(tmp_path / "checkpoints.db"))
        saved = store.load_checkpoint(first["checkpoint_uid"])
        assert saved["scope"] == "hitl"
        assert saved["interrupt_data"]["question"] == "Need more context?"
        assert saved["stage_index"] == 0

        second = agent.execution(checkpoint_uid=first["checkpoint_uid"], resume_input="User answer")
        assert second["final_message"] == "All done"
        assert second.get("status") != "awaiting_user_input"
        assert calls[1][0]["content"] == "User answer"
