"""
Tests for IkaBaseAgent helper mixin: validation, memory access, message history,
final_prompt, parse_control_calls, _fallback_final_content, _build_final_output.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

mock_ika_mem = MagicMock()
with patch.dict("sys.modules", {"IkaMem": mock_ika_mem}):
    from IkaCore.agents import IkaBaseAgent
from IkaCore.stages import IkaStage


def _minimal_agent(**kwargs):
    defaults = {"name": "A", "description": "D", "prompt": "P", "model_id": "gpt-4o", "api_key": "k"}
    defaults.update(kwargs)
    return IkaBaseAgent(**defaults)


class TestValidateRequiredFields:
    def test_all_present_passes(self):
        IkaBaseAgent._validate_required_fields({"a": "x", "b": 1})

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="a is required"):
            IkaBaseAgent._validate_required_fields({"a": ""})

    def test_none_raises(self):
        with pytest.raises(ValueError, match="b is required"):
            IkaBaseAgent._validate_required_fields({"a": "ok", "b": None})


class TestBuildMemoryAccessDefaults:
    def test_memory_disabled_all_false(self):
        out = IkaBaseAgent._build_memory_access_defaults(False)
        assert out["short_term_save"] is False
        assert out["long_term_search"] is False

    def test_memory_enabled_all_true(self):
        out = IkaBaseAgent._build_memory_access_defaults(True)
        assert out["short_term_save"] is True
        assert out["long_term_search"] is True

    def test_overrides_applied(self):
        out = IkaBaseAgent._build_memory_access_defaults(True, {"short_term_save": False})
        assert out["short_term_save"] is False
        assert out["long_term_search"] is True


class TestInitialMessageHistory:
    def test_structure(self):
        h = IkaBaseAgent._initial_message_history("System text")
        assert h["system"]["message"] == "System text"
        assert h["first_input"]["message"] == ""
        assert h["summary"]["message"] == ""
        assert h["messages"] == {}

    def test_none_system_becomes_empty(self):
        h = IkaBaseAgent._initial_message_history(None)
        assert h["system"]["message"] == ""


class TestFinalPrompt:
    def test_uses_start_stage_end(self):
        a = _minimal_agent(start_prompt="Start", end_prompt="End")
        stage = IkaStage("S1", "Stage prompt", [])
        out = a.final_prompt(stage)
        assert "Start" in out
        assert "Stage prompt" in out
        assert "End" in out

    def test_fallback_to_description(self):
        a = _minimal_agent(start_prompt="", end_prompt="")
        stage = IkaStage("S", "", [])
        out = a.final_prompt(stage)
        assert a.description in out

    def test_execute_stage_refreshes_first_input_for_each_stage(self):
        stage0 = IkaStage("S0", "Stage 0 prompt", [])
        stage1 = IkaStage("S1", "Stage 1 prompt", [])
        a = _minimal_agent(Stages=[stage0, stage1])
        a.logger = MagicMock(level=0)
        a.message_history["first_input"]["message"] = "original stage prompt"

        def fake_chat_wrapper(*args, **kwargs):
            assert "Stage 1 prompt" in a.message_history["first_input"]["message"]
            return {
                "content": "final",
                "message_history": a.message_history,
                "tool_calls": [],
                "executed_tool_calls": [
                    {"function": {"name": "agent_end", "arguments": '{"input": "final"}'}}
                ],
                "content_before_tools": "final",
                "usage": {},
                "cost": {},
                "hijacked": False,
            }

        a.chat_wrapper = MagicMock(side_effect=fake_chat_wrapper)
        next_stage, last_content, agent_end_called, end_text, used = a.execute_stage(1, remaining_steps=5)
        assert next_stage == 1
        assert last_content == "final"
        assert agent_end_called is True
        assert end_text == "final"
        assert used == 1
        assert "Stage 1 prompt" in a.message_history["first_input"]["message"]


class TestParseControlCalls:
    def test_agent_end_detected(self):
        a = _minimal_agent()
        calls = [{"function": {"name": "agent_end", "arguments": '{"input": "Final answer"}'}}]
        target, ended, text = a.parse_control_calls(calls, None)
        assert ended is True
        assert "Final answer" in (text or "")

    def test_stage_end_sets_target_next(self):
        stage = IkaStage("S", "P", [])
        a = _minimal_agent(Stages=[stage])
        calls = [{"function": {"name": "stage_end", "arguments": "{}"}}]
        target, ended, text = a.parse_control_calls(calls, stage)
        assert target == "next"
        assert ended is False

    def test_change_stage_parsed(self):
        stage = IkaStage("S", "P", [], allowed_back_to=[0])
        a = _minimal_agent(Stages=[stage, IkaStage("S1", "P1", [])])
        calls = [{"function": {"name": "change_stage", "arguments": '{"stage_index": 0, "reason": "Redo"}'}}]
        target, ended, text = a.parse_control_calls(calls, stage, current_stage_idx=1)
        assert not ended
        assert target in (0, "next") or target is None

    def test_agent_end_with_input_succeeds(self):
        a = _minimal_agent()
        calls = [{"function": {"name": "agent_end", "arguments": '{"input": "Final answer here"}'}}]
        target, ended, text = a.parse_control_calls(calls, None, response_content="")
        assert ended is True
        assert "Final answer" in (text or "")


class TestFallbackFinalContent:
    def test_prefers_agent_end_text(self):
        a = _minimal_agent()
        out = a._fallback_final_content("From tool", "Before", "Last")
        assert out == "From tool"

    def test_fallback_to_content_before_tools(self):
        a = _minimal_agent()
        out = a._fallback_final_content("", "Content before", "Last")
        assert out == "Content before"

    def test_fallback_to_last_content(self):
        a = _minimal_agent()
        out = a._fallback_final_content("", "", "Last content")
        assert out == "Last content"

    def test_all_empty_raises(self):
        a = _minimal_agent()
        with pytest.raises(ValueError, match="no output"):
            a._fallback_final_content("", "", "")


class TestBuildFinalOutput:
    def test_returns_final_message_and_summary(self):
        a = _minimal_agent()
        mock_bm = MagicMock()
        out = a._build_final_output("Final text", mock_bm)
        assert out["final_message"] == "Final text"
        assert out["summary"] == "Final text"

    def test_build_final_output_has_keys(self):
        a = _minimal_agent(summarize_final=True)
        mock_bm = MagicMock()
        out = a._build_final_output("Final", mock_bm)
        assert "final_message" in out
        assert "summary" in out
        assert out["final_message"] == "Final"
        assert out["summary"]
