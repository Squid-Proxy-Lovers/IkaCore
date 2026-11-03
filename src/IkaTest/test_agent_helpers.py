"""
Tests for IkaBaseAgent helper mixin: validation, memory access, message history,
final_prompt, parse_control_calls, _fallback_final_content, _build_final_output.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

mock_ika_mem = MagicMock()
with patch.dict("sys.modules", {"IkaMem": mock_ika_mem}):
    from IkaCore.agents import IkaBaseAgent
from IkaCore.agent_execution_helpers import (
    build_stage_content_prompt,
    resolve_stage_model_overrides,
    stage_step_limit,
)
from IkaCore.checkpoint import CheckpointStore
from IkaCore.execution_types import AgentChatTurn, SimpleRuntimeContext, StageExecutionResult, StageRuntimeContext
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


class TestFinalAnswerCheckValidation:
    def test_rejects_non_callable_check(self):
        with pytest.raises(ValueError, match="callable"):
            _minimal_agent(final_answer_check=["not-callable"])

    def test_rejects_check_without_docstring(self):
        def no_docstring(_history) -> bool:
            return True

        with pytest.raises(ValueError, match="docstring"):
            _minimal_agent(final_answer_check=[no_docstring])

    def test_rejects_check_with_non_bool_annotation(self):
        def wrong_annotation(_history) -> str:
            """Return the validation result."""
            return "yes"

        with pytest.raises(ValueError, match="return type annotation"):
            _minimal_agent(final_answer_check=[wrong_annotation])

    def test_rejects_check_returning_non_bool(self):
        def wrong_result(_history):
            """Return the validation result."""
            return "yes"

        with pytest.raises(ValueError, match="must return a boolean"):
            _minimal_agent(final_answer_check=[wrong_result])

    def test_accepts_valid_check(self):
        def valid_check(_history) -> bool:
            """Return whether the final answer is acceptable."""
            return True

        a = _minimal_agent(final_answer_check=[valid_check])
        assert a.final_answer_checks == [valid_check]


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

    def test_stage_execution_helpers_build_prompt_overrides_and_limits(self):
        stage0 = IkaStage("S0", "Stage 0 prompt", [], stage_max_step=0)
        stage1 = IkaStage(
            "S1",
            "Stage 1 prompt",
            [],
            hitl=True,
            model_id="gpt-4o-mini",
            api_key="stage-key",
            max_tokens=123,
            temperature=0.4,
            stage_max_step=2,
        )
        agent = _minimal_agent(Stages=[stage0, stage1])

        prompt = build_stage_content_prompt(agent.prompt, agent.Stages, stage1, 1)
        overrides = resolve_stage_model_overrides(agent, stage1)

        assert "Stage 0 (S0)" in prompt
        assert "CURRENT STAGE IS" in prompt
        assert "Stage 1 prompt" in prompt
        assert "HITL" in prompt
        assert "agent_end" in prompt
        assert overrides["model_id"] == "gpt-4o-mini"
        assert overrides["api_key"] == "stage-key"
        assert overrides["max_tokens"] == 123
        assert overrides["temperature"] == 0.4
        assert "api_url" in overrides
        assert stage_step_limit(stage0, remaining_steps=7) == 7
        assert stage_step_limit(stage1, remaining_steps=7) == 2

    def test_workflow_context_and_stage_wiring(self):
        stage0 = IkaStage("S0", "Stage 0 prompt", [])
        stage1 = IkaStage("S1", "Stage 1 prompt", [])
        subagent = _minimal_agent(name="Sub")
        a = _minimal_agent(Stages=[stage0, stage1])
        a.message_history["first_input"]["message"] = "original"

        a.inject_workflow_context("")
        assert a.message_history["first_input"]["message"] == "original"

        a.inject_workflow_context("workflow context")
        assert a.message_history["first_input"]["message"] == "workflow context\n\noriginal"

        a.apply_workflow_stage_wiring({1: {"subagents": [subagent]}, 9: {"subagents": []}})
        assert stage1.subagents == [subagent]
        assert stage0.subagents == []

    def test_clone_for_run_copies_selected_runtime_overrides(self):
        stage = IkaStage("S", "Prompt", [], memory_access={"short_term_save": False})
        a = _minimal_agent(Stages=[stage])
        a.message_history["first_input"]["message"] = "history"
        a.context_budget = 123
        a._base_prompt = "base"
        a._parent_hierarchy = ["Parent"]
        a.execution = MagicMock(return_value={"final_message": "done"})

        clone = a.clone_for_run(name="Clone", prompt="clone prompt", include_history=True)

        assert clone.name == "Clone"
        assert clone.prompt == "clone prompt"
        assert clone.message_history["first_input"]["message"] == "history"
        assert clone.context_budget == 123
        assert clone._base_prompt == "base"
        assert clone._parent_hierarchy == ["Parent"]
        assert clone.execution is a.execution
        assert clone.Stages[0] is not stage
        assert clone.Stages[0].memory_access == {"short_term_save": False}

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
        result = a.execute_stage(1, remaining_steps=5)
        assert result.__class__.__name__ == "StageExecutionResult"
        next_stage, last_content, agent_end_called, end_text, used = result
        assert next_stage == 1
        assert result.next_stage_index == 1
        assert last_content == "final"
        assert agent_end_called is True
        assert end_text == "final"
        assert used == 1
        assert result.used_steps == 1
        assert "Stage 1 prompt" in a.message_history["first_input"]["message"]


class TestExecutionRuntimeHelpers:
    @staticmethod
    def _turn(**overrides):
        data = {
            "response": {"usage": {}, "cost": {}},
            "agent_end_exception": False,
            "last_content": "",
            "content_before_tools": "",
            "tool_calls": [],
            "executed_tool_calls": [],
        }
        data.update(overrides)
        return AgentChatTurn(**data)

    def test_run_chat_turn_wraps_agent_end_exception_and_syncs_tool_counts(self):
        a = _minimal_agent()
        response = {
            "content": "done",
            "content_before_tools": "before",
            "tool_calls": [],
            "executed_tool_calls": [{"function": {"name": "agent_end", "arguments": '{"input":"done"}'}}],
            "message_history": a.message_history,
        }
        barebone = SimpleNamespace(_tool_call_counts={"search": 2})
        agent_end_exception = a._run_chat_turn.__func__.__globals__["AgentEndException"]
        a.chat_wrapper = MagicMock(side_effect=agent_end_exception(response=response))
        a._get_chat_client = MagicMock(return_value=None)

        turn = a._run_chat_turn(
            barebone,
            [{"role": "user", "content": "task"}],
            {},
            current_stage_index=None,
            total_stages=0,
            context_label="test",
        )

        assert turn.agent_end_exception is True
        assert turn.last_content == "done"
        assert turn.content_before_tools == "before"
        assert turn.executed_tool_calls == response["executed_tool_calls"]
        assert a._tool_call_counts == {"search": 2}

    def test_prepare_stage_runtime_records_resume_input_and_stage_prompt(self):
        stage = IkaStage("Review", "Review prompt", [])
        a = _minimal_agent(Stages=[stage])
        a.logger = MagicMock(level=0)

        runtime = a._prepare_stage_runtime(0, remaining_steps=4, resume_input="human answer")

        assert runtime.stage is stage
        assert runtime.messages == [{"role": "user", "content": "human answer"}]
        assert runtime.step_limit == 4
        assert runtime.barebone_model._tool_call_counts == {}
        assert a.message_history["first_input"]["message"].count("Review prompt") >= 1
        assert any(
            item.get("type") == "hitl_input" and item.get("message") == "human answer"
            for item in a.message_history["messages"].values()
        )
        a.logger.log_hitl_input.assert_called_once_with("Review", "human answer")

    def test_stage_extension_policy_appends_redirect_then_forces_answer(self):
        a = _minimal_agent(max_stage_extensions=1, extend_stage_steps_by=2)
        barebone = MagicMock()
        messages = []

        with patch.dict(
            a._maybe_extend_stage.__func__.__globals__,
            {"run_summarization": MagicMock(return_value="remaining work")},
        ):
            step_limit, extension_count, last_content, should_break = a._maybe_extend_stage(
                barebone,
                messages,
                step_limit=3,
                extension_count=0,
                last_content="partial",
            )

        assert step_limit == 5
        assert extension_count == 1
        assert last_content == "partial"
        assert should_break is False
        assert "remaining work" in messages[-1]["content"]

        with patch.dict(
            a._maybe_extend_stage.__func__.__globals__,
            {"run_summarization": MagicMock(return_value="forced final")},
        ):
            step_limit, extension_count, last_content, should_break = a._maybe_extend_stage(
                barebone,
                messages,
                step_limit=5,
                extension_count=1,
                last_content="partial",
            )

        assert step_limit == 5
        assert extension_count == 1
        assert last_content == "forced final"
        assert should_break is True

    def test_handle_stage_interrupt_persists_hitl_checkpoint_payload(self, tmp_path):
        stage = IkaStage("Review", "Review prompt", [])
        a = _minimal_agent(
            Stages=[stage],
            checkpoint=True,
            checkpoint_db_path=str(tmp_path / "checkpoints.db"),
        )
        a.logger = MagicMock(level=2)
        cli = MagicMock()
        turn = self._turn(
            response={
                "interrupted": True,
                "interrupt_data": {"question": "Proceed?", "details": {"risk": "low"}},
                "status": "awaiting_user_input",
                "usage": {},
                "cost": {},
            },
            last_content="Need human input",
        )

        with patch.dict(
            a._emit_stage_checkpoint.__func__.__globals__,
            {"get_cli_output": MagicMock(return_value=cli)},
        ):
            result = a._handle_stage_interrupt(stage, 0, remaining_steps=6, used_steps=2, turn=turn)

        assert result is not None
        assert result.next_stage_index == 0
        assert result.last_content == "Need human input"
        assert result.agent_end_called is False
        assert result.used_steps == 2
        assert a._last_interrupt["remaining_steps"] == 4
        assert a._last_interrupt["stage_name"] == "Review"
        assert a._last_interrupt["interrupt_data"]["question"] == "Proceed?"
        assert a._last_interrupt["checkpoint_uid"]
        a.logger.log_hitl_prompt.assert_called_once_with("Review")

        saved = CheckpointStore(str(tmp_path / "checkpoints.db")).load_checkpoint(a._last_interrupt["checkpoint_uid"])
        assert saved["scope"] == "hitl"
        assert saved["stage_index"] == 0
        assert saved["remaining_steps"] == 4
        assert saved["last_content"] == "Need human input"
        assert saved["interrupt_status"] == "awaiting_user_input"
        assert saved["interrupt_data"]["details"]["risk"] == "low"

    def test_handle_stage_interrupt_ignores_non_interrupted_turn(self):
        stage = IkaStage("Review", "Review prompt", [])
        a = _minimal_agent(Stages=[stage])
        a._last_interrupt = {"status": "previous"}
        turn = self._turn(response={"interrupted": False, "usage": {}, "cost": {}})

        assert a._handle_stage_interrupt(stage, 0, remaining_steps=3, used_steps=1, turn=turn) is None
        assert a._last_interrupt == {"status": "previous"}

    def test_simple_run_extension_appends_redirect_and_respects_extension_limit(self):
        a = _minimal_agent(maxsteps=2, max_step_extensions=1, extend_steps_by=4)
        barebone_summary = SimpleNamespace()
        a.get_barebone = MagicMock(return_value=barebone_summary)
        messages = []
        cli = MagicMock()
        summarizer = MagicMock(return_value="remaining simple work")

        with patch.dict(
            a._maybe_extend_simple_run.__func__.__globals__,
            {"run_summarization": summarizer},
        ):
            extension_count = a._maybe_extend_simple_run(
                "system",
                messages,
                ["A"],
                current_step=2,
                extension_count=0,
                cli=cli,
            )

        assert extension_count == 1
        assert a.maxsteps == 6
        assert messages[-1]["role"] == "user"
        assert "remaining simple work" in messages[-1]["content"]
        a.get_barebone.assert_called_once_with(
            "system",
            [],
            parent_hierarchy=["A"],
            suppress_init_output=True,
        )
        summarizer.assert_called_once_with(
            barebone_summary,
            a.message_history,
            prompt_kind="what_remains",
            write_to_history=False,
        )
        cli.agent_response.assert_called_once()

        a.get_barebone.reset_mock()
        assert a._maybe_extend_simple_run("system", messages, ["A"], 6, extension_count, cli) == 1
        a.get_barebone.assert_not_called()

    def test_finalize_simple_after_max_steps_prefers_agent_end_text(self):
        a = _minimal_agent(maxsteps=5)
        a.get_barebone = MagicMock()
        cli = MagicMock()
        cli.get_step.return_value = 4
        summarizer = MagicMock()

        with patch.dict(
            a._finalize_simple_after_max_steps.__func__.__globals__,
            {"run_summarization": summarizer},
        ):
            final_response, summary = a._finalize_simple_after_max_steps(
                "system",
                ["A"],
                cli,
                last_agent_end_text="already final",
                last_content="last content",
            )

        assert final_response == "already final"
        assert summary == "already final"
        a.get_barebone.assert_not_called()
        summarizer.assert_not_called()
        cli.agent_response.assert_called_once_with(
            "A",
            "Reached max steps (5). Returning last content.\n\nalready final",
            ["A"],
            step=4,
            is_final=True,
        )

    def test_finalize_simple_after_max_steps_forces_or_falls_back_to_last_content(self):
        a = _minimal_agent(maxsteps=5)
        barebone_final = SimpleNamespace()
        a.get_barebone = MagicMock(return_value=barebone_final)
        cli = MagicMock()
        cli.get_step.return_value = None
        summarizer = MagicMock(return_value="  forced final  ")

        with patch.dict(
            a._finalize_simple_after_max_steps.__func__.__globals__,
            {"run_summarization": summarizer},
        ):
            final_response, summary = a._finalize_simple_after_max_steps(
                "system",
                ["A"],
                cli,
                last_agent_end_text=None,
                last_content="last content",
            )

        assert final_response == "forced final"
        assert summary == "forced final"
        a.get_barebone.assert_called_once_with(
            "system",
            [],
            parent_hierarchy=["A"],
            suppress_init_output=True,
        )
        summarizer.assert_called_once_with(
            barebone_final,
            a.message_history,
            prompt_kind="force_answer",
            write_to_history=False,
        )
        cli.agent_response.assert_called_once_with(
            "A",
            "Reached max steps (5). Returning last content.\n\nforced final",
            ["A"],
            step=5,
            is_final=True,
        )

        summarizer.return_value = "   "
        final_response, summary = a._finalize_simple_after_max_steps(
            "system",
            ["A"],
            cli,
            last_agent_end_text=None,
            last_content="fallback content",
        )
        assert final_response == "fallback content"
        assert summary == "fallback content"

    def test_simple_loop_guard_only_breaks_after_empty_non_initial_turn(self):
        a = _minimal_agent()
        a.logger = MagicMock()
        empty_turn = self._turn()

        assert a._simple_turn_has_no_progress(empty_turn, step_num=0) is False
        assert a._simple_turn_has_no_progress(empty_turn, step_num=1) is True
        a.logger.log_action.assert_called_once_with(
            "CRITICAL: Infinite loop detected (no tools, no content). Breaking."
        )

        assert a._simple_turn_has_no_progress(self._turn(last_content="progress"), step_num=1) is False
        assert a._simple_turn_has_no_progress(self._turn(tool_calls=[{"name": "tool"}]), step_num=1) is False
        assert a._simple_turn_has_no_progress(
            self._turn(executed_tool_calls=[{"function": {"name": "tool"}}]),
            step_num=1,
        ) is False

    def test_parse_simple_control_promotes_agent_end_exception_and_reports_parse_errors(self):
        a = _minimal_agent()
        a.parse_control_calls = MagicMock(return_value=(None, False, None))
        turn = self._turn(agent_end_exception=True, content_before_tools="before")

        agent_end_called, agent_end_text = a._parse_simple_control(turn, MagicMock(), ["A"], current_step=3)

        assert agent_end_called is True
        assert agent_end_text is None
        a.parse_control_calls.assert_called_once_with([], None, response_content="before")

        cli = MagicMock()
        a.parse_control_calls = MagicMock(side_effect=ValueError("bad control"))
        with pytest.raises(ValueError, match="bad control"):
            a._parse_simple_control(self._turn(), cli, ["A"], current_step=4)
        cli.agent_response.assert_called_once_with(
            "A",
            "ERROR: bad control",
            ["A"],
            step=4,
            is_final=False,
        )

    def test_finish_simple_agent_end_emits_final_or_reports_empty_result(self):
        a = _minimal_agent()
        a.logger = MagicMock()
        cli = MagicMock()
        turn = self._turn(
            last_content="last",
            content_before_tools="before",
            tool_calls=[{"function": {"name": "agent_end"}}],
            response={"usage": {"total_tokens": 1}, "cost": {"total_cost": 0.1}},
        )

        final_response, summary = a._finish_simple_agent_end(
            "final",
            turn,
            cli,
            ["A"],
            current_step=2,
            step_num=1,
            step_start=0.0,
        )

        assert final_response == "final"
        assert summary == "final"
        cli.agent_response.assert_called_once_with("A", "final", ["A"], step=2, is_final=True)
        a.logger.log_step.assert_called_once()

        bad_cli = MagicMock()
        with pytest.raises(ValueError, match="no output"):
            a._finish_simple_agent_end(
                None,
                self._turn(),
                bad_cli,
                ["A"],
                current_step=3,
                step_num=2,
                step_start=0.0,
            )
        bad_cli.agent_response.assert_called_once_with(
            "A",
            "ERROR: agent_end was called but no output was provided. "
            "The agent MUST provide a final answer when calling agent_end.",
            ["A"],
            step=3,
            is_final=False,
        )

    def test_simple_non_terminal_step_records_text_or_checkpoints_progress(self):
        a = _minimal_agent(maxsteps=3)
        a.logger = MagicMock()
        runtime = SimpleRuntimeContext(
            system_prompt="system",
            current_hierarchy=["A"],
            barebone_model=SimpleNamespace(),
            messages=[],
            tool_executors={},
            cli=MagicMock(),
        )
        text_turn = self._turn(last_content="intermediate text")

        step_num, extension_count, should_continue = a._finish_simple_non_terminal_step(
            text_turn,
            runtime.messages,
            runtime,
            current_step=1,
            step_num=0,
            step_start=0.0,
            extension_count=0,
        )

        assert (step_num, extension_count, should_continue) == (1, 0, True)
        assert runtime.messages[-1] == {"role": "assistant", "content": "intermediate text"}
        runtime.cli.agent_response.assert_called_once_with(
            "A",
            "intermediate text",
            ["A"],
            step=1,
            is_final=False,
        )

        a._save_agent_checkpoint = MagicMock()
        tool_turn = self._turn(
            last_content="tool result",
            tool_calls=[{"function": {"name": "search"}}],
        )
        step_num, extension_count, should_continue = a._finish_simple_non_terminal_step(
            tool_turn,
            runtime.messages,
            runtime,
            current_step=2,
            step_num=1,
            step_start=0.0,
            extension_count=0,
        )

        assert (step_num, extension_count, should_continue) == (2, 0, False)
        a._save_agent_checkpoint.assert_called_once_with(1, "tool result")

    def test_prepare_simple_runtime_builds_start_prompt_tools_and_context_budget(self):
        a = _minimal_agent(system_prompt="system", prompt="agent prompt")
        a.message_history["first_input"]["message"] = "history prompt"
        a.context_budget = 42
        barebone = SimpleNamespace()
        a.build_simple_tools = MagicMock(return_value=["dynamic_tool"])
        a.get_barebone = MagicMock(return_value=barebone)
        a.build_tool_executors = MagicMock(return_value={"tool": object()})
        cli = MagicMock()

        with patch.dict(
            a._prepare_simple_runtime.__func__.__globals__,
            {"get_cli_output": MagicMock(return_value=cli)},
        ):
            runtime = a._prepare_simple_runtime()

        assert runtime.system_prompt == "system"
        assert runtime.current_hierarchy == ["A"]
        assert runtime.barebone_model is barebone
        assert runtime.tool_executors
        assert runtime.cli is cli
        assert runtime.messages[0]["role"] == "user"
        assert "history prompt" in runtime.messages[0]["content"]
        assert "agent_end" in runtime.messages[0]["content"]
        assert barebone._tool_call_counts == {}
        assert barebone.context_budget == 42
        cli.set_step.assert_called_once_with("A", 1)
        a.get_barebone.assert_called_once_with(
            "system",
            ["dynamic_tool"],
            parent_hierarchy=["A"],
            content_prompt_override=runtime.messages[0]["content"],
        )
        a.build_tool_executors.assert_called_once_with(
            a.tools,
            memory_access=a.memory_access,
            subagents=a.subagents,
            parent_hierarchy=["A"],
        )

    def test_parse_and_finish_stage_control_handle_agent_end_exception_and_errors(self):
        stage = IkaStage("Review", "Review prompt", [])
        a = _minimal_agent(Stages=[stage])
        a.logger = MagicMock()
        a.parse_control_calls = MagicMock(return_value=(None, False, None))
        turn = self._turn(agent_end_exception=True, content_before_tools="before")

        target_stage, agent_end_called, agent_end_text = a._parse_stage_control(turn, stage, 0)

        assert target_stage is None
        assert agent_end_called is True
        assert agent_end_text is None
        a.parse_control_calls.assert_called_once_with([], stage, 0, response_content="before")

        a.parse_control_calls = MagicMock(side_effect=ValueError("bad stage control"))
        with pytest.raises(ValueError, match="bad stage control"):
            a._parse_stage_control(self._turn(), stage, 0)
        a.logger.log_action.assert_called_with("ERROR in agent_end: bad stage control")

        result = a._finish_stage_agent_end(
            stage_index=0,
            agent_end_text=None,
            turn=self._turn(last_content="last", content_before_tools="before"),
            used_steps=2,
        )
        assert tuple(result) == (0, "before", True, "before", 2)

        with pytest.raises(ValueError, match="no output"):
            a._finish_stage_agent_end(0, None, self._turn(), used_steps=1)
        a.logger.log_action.assert_called_with(
            "ERROR: agent_end was called but no output was provided. "
            "The agent MUST provide a final answer when calling agent_end."
        )

    @pytest.mark.parametrize(
        ("target_stage", "expected_next"),
        [
            ("next", 1),
            (0, 0),
        ],
    )
    def test_execute_stage_returns_stage_control_targets(self, target_stage, expected_next):
        stage = IkaStage("Review", "Review prompt", [])
        a = _minimal_agent(Stages=[stage])
        runtime = StageRuntimeContext(
            stage=stage,
            system_prompt="system",
            content_prompt="content",
            current_hierarchy=["A", "Stage 0: Review"],
            barebone_model=SimpleNamespace(),
            messages=[],
            tool_executors={},
            step_limit=3,
        )
        a._prepare_stage_runtime = MagicMock(return_value=runtime)
        a._enforce_rate_limit_model = MagicMock()
        a._run_chat_turn = MagicMock(return_value=self._turn(last_content="partial"))
        a._handle_stage_interrupt = MagicMock(return_value=None)
        a._parse_stage_control = MagicMock(return_value=(target_stage, False, None))

        result = a.execute_stage(0, remaining_steps=5)

        assert tuple(result) == (expected_next, "partial", False, None, 1)
        a._run_chat_turn.assert_called_once()
        assert a._run_chat_turn.call_args.kwargs["current_stage_index"] == 0
        assert a._run_chat_turn.call_args.kwargs["context_label"] == "execute_stage"

    def test_execute_stage_logs_checkpoints_and_forces_final_when_extensions_exhaust(self):
        stage = IkaStage("Review", "Review prompt", [])
        a = _minimal_agent(Stages=[stage], checkpoint=True)
        a.logger = MagicMock()
        messages = []
        runtime = StageRuntimeContext(
            stage=stage,
            system_prompt="system",
            content_prompt="content",
            current_hierarchy=["A", "Stage 0: Review"],
            barebone_model=SimpleNamespace(),
            messages=messages,
            tool_executors={},
            step_limit=1,
        )
        a._prepare_stage_runtime = MagicMock(return_value=runtime)
        a._enforce_rate_limit_model = MagicMock()
        a._run_chat_turn = MagicMock(
            return_value=self._turn(last_content="partial", tool_calls=[{"function": {"name": "search"}}])
        )
        a._handle_stage_interrupt = MagicMock(return_value=None)
        a._parse_stage_control = MagicMock(return_value=(None, False, None))
        a._save_stage_checkpoint = MagicMock()
        a._maybe_extend_stage = MagicMock(return_value=(1, 0, "forced answer", True))

        result = a.execute_stage(0, remaining_steps=5)

        assert tuple(result) == (1, "forced answer", False, None, 1)
        assert messages[-1] == {"role": "assistant", "content": "partial"}
        a._save_stage_checkpoint.assert_called_once_with(0, 4, "partial")
        a._maybe_extend_stage.assert_called_once()
        a.logger.log_step.assert_called_once()
        a.logger.log_stage_end.assert_called_once_with("Review", 1)

    def test_start_simple_step_initializes_first_step_and_rate_limits_each_step(self):
        a = _minimal_agent(maxsteps=3)
        a._enforce_rate_limit_model = MagicMock()
        cli = MagicMock()

        with patch.object(a._start_simple_step.__func__.__globals__["time"], "time", side_effect=[10.5, 11.5]):
            first_step, first_start = a._start_simple_step(cli, ["A"], step_num=0)
            second_step, second_start = a._start_simple_step(cli, ["A"], step_num=1)

        assert (first_step, first_start) == (1, 10.5)
        assert (second_step, second_start) == (2, 11.5)
        cli.agent_init.assert_called_once_with("A", ["A"], step=1, description="Step 1/3")
        assert cli.set_step.call_args_list[0].args == ("A", 1)
        assert cli.set_step.call_args_list[1].args == ("A", 2)
        assert a._enforce_rate_limit_model.call_count == 2

    def test_run_simple_returns_agent_end_and_finalizes_after_no_progress(self):
        a = _minimal_agent(maxsteps=2)
        runtime = SimpleRuntimeContext(
            system_prompt="system",
            current_hierarchy=["A"],
            barebone_model=SimpleNamespace(),
            messages=[],
            tool_executors={},
            cli=MagicMock(),
        )
        a._prepare_simple_runtime = MagicMock(return_value=runtime)
        a._start_simple_step = MagicMock(return_value=(1, 0.0))
        turn = self._turn(last_content="content", content_before_tools="before")
        a._run_chat_turn = MagicMock(return_value=turn)
        a._simple_turn_has_no_progress = MagicMock(return_value=False)
        a._parse_simple_control = MagicMock(return_value=(True, "final"))
        a._finish_simple_agent_end = MagicMock(return_value=("final", "final"))

        assert a.run_simple() == ("final", "final")
        a._finish_simple_agent_end.assert_called_once_with(
            "final",
            turn,
            runtime.cli,
            ["A"],
            1,
            0,
            0.0,
        )

        b = _minimal_agent(maxsteps=2)
        b._prepare_simple_runtime = MagicMock(return_value=runtime)
        b._start_simple_step = MagicMock(return_value=(1, 0.0))
        empty_turn = self._turn()
        b._run_chat_turn = MagicMock(return_value=empty_turn)
        b._simple_turn_has_no_progress = MagicMock(return_value=True)
        b._finalize_simple_after_max_steps = MagicMock(return_value=("fallback", "fallback"))

        assert b.run_simple() == ("fallback", "fallback")
        b._finalize_simple_after_max_steps.assert_called_once_with(
            "system",
            ["A"],
            runtime.cli,
            None,
            "",
        )

    def test_run_simple_extends_after_text_only_progress_at_step_limit(self):
        a = _minimal_agent(maxsteps=1, max_step_extensions=1, extend_steps_by=1)
        runtime = SimpleRuntimeContext(
            system_prompt="system",
            current_hierarchy=["A"],
            barebone_model=SimpleNamespace(),
            messages=[],
            tool_executors={},
            cli=MagicMock(),
        )
        a._prepare_simple_runtime = MagicMock(return_value=runtime)
        a._enforce_rate_limit_model = MagicMock()
        a.get_barebone = MagicMock(return_value=SimpleNamespace())
        a._run_chat_turn = MagicMock(
            side_effect=[
                self._turn(last_content="draft", content_before_tools="draft"),
                self._turn(last_content="final", content_before_tools="final"),
            ]
        )
        a._parse_simple_control = MagicMock(side_effect=[(False, None), (True, "final")])
        summarizer = MagicMock(return_value="remaining work")

        with patch.dict(a._maybe_extend_simple_run.__func__.__globals__, {"run_summarization": summarizer}):
            assert a.run_simple() == ("final", "final")

        assert a.maxsteps == 2
        assert runtime.messages[0] == {"role": "assistant", "content": "draft"}
        assert "remaining work" in runtime.messages[1]["content"]
        summarizer.assert_called_once()

    def test_interrupt_output_uses_last_content_or_question_and_runtime_totals(self):
        a = _minimal_agent()
        a._total_usage = {"total_tokens": 7}
        a._total_cost = {"total_cost": 0.5}

        with_last_content = a._interrupt_output({"last_content": "partial", "interrupt_data": {"question": "Q?"}})
        from_question = a._interrupt_output({"interrupt_data": {"question": "Q?"}})

        assert with_last_content["final_message"] == "partial"
        assert with_last_content["summary"] == "partial"
        assert with_last_content["usage"] == {"total_tokens": 7}
        assert with_last_content["cost"] == {"total_cost": 0.5}
        assert from_question["final_message"] == "Q?"
        assert from_question["status"] == "awaiting_user_input"

    def test_staged_dispatch_resumes_stage_checkpoint_and_builds_final_output(self):
        stage = IkaStage("Review", "Review prompt", [])
        a = _minimal_agent(Stages=[stage], maxsteps=10)
        resume_history = _minimal_agent().message_history
        resume_cp = {
            "scope": "stage",
            "stage_index": 0,
            "remaining_steps": 3,
            "message_history": resume_history,
            "last_content": "previous",
        }
        barebone = SimpleNamespace()
        a.execute_stage = MagicMock(return_value=StageExecutionResult(1, "last", True, ".", 2))
        a.get_barebone = MagicMock(return_value=barebone)
        a._build_final_output = MagicMock(return_value={"final_message": "last"})

        out = a._run_staged_execution(None, None, resume_cp, last_content="")

        assert out == {"final_message": "last"}
        assert a.message_history is resume_history
        a.execute_stage.assert_called_once_with(0, 3, resume_input=None)
        a._build_final_output.assert_called_once_with("last", barebone)

    def test_staged_dispatch_replays_hitl_without_input_or_resumes_with_input(self):
        stage = IkaStage("Review", "Review prompt", [])
        resume_history = _minimal_agent().message_history
        resume_cp = {
            "scope": "hitl",
            "stage_index": 0,
            "remaining_steps": 2,
            "message_history": resume_history,
            "last_content": "Need input",
            "interrupt_status": "awaiting_user_input",
            "interrupt_data": {"question": "Proceed?"},
        }
        a = _minimal_agent(Stages=[stage], maxsteps=10)
        a.execute_stage = MagicMock()

        replay = a._run_staged_execution("cp-1", "", resume_cp, last_content="")

        assert replay["status"] == "awaiting_user_input"
        assert replay["checkpoint_uid"] == "cp-1"
        assert replay["interrupt_data"]["question"] == "Proceed?"
        a.execute_stage.assert_not_called()

        b = _minimal_agent(Stages=[stage], maxsteps=10)
        b.execute_stage = MagicMock(return_value=StageExecutionResult(1, "done", True, "done", 1))
        b.get_barebone = MagicMock(return_value=SimpleNamespace())
        b._build_final_output = MagicMock(return_value={"final_message": "done"})

        out = b._run_staged_execution("cp-1", "human answer", resume_cp, last_content="")

        assert out == {"final_message": "done"}
        b.execute_stage.assert_called_once_with(0, 2, resume_input="human answer")

    def test_staged_dispatch_returns_current_interrupt_after_stage_execution(self):
        stage = IkaStage("Review", "Review prompt", [])
        a = _minimal_agent(Stages=[stage])
        a._last_interrupt = {"last_content": "Need input", "interrupt_data": {"question": "Proceed?"}}
        a.execute_stage = MagicMock(return_value=StageExecutionResult(0, "Need input", False, None, 1))

        out = a._run_staged_execution(None, None, None, last_content="")

        assert out["final_message"] == "Need input"
        a.execute_stage.assert_called_once_with(0, 100, resume_input=None)

    def test_next_agent_dispatch_uses_summary_or_final_message_as_next_input(self):
        next_agent = _minimal_agent(name="Next")
        next_agent.execution = MagicMock(return_value={"final_message": "next done"})
        a = _minimal_agent(next_agent=next_agent, summarize_final=True)
        a.run_simple = MagicMock(return_value=("final", "final"))
        barebone = SimpleNamespace()
        a.get_barebone = MagicMock(return_value=barebone)
        summarizer = MagicMock(return_value="short summary")

        with patch.dict(
            a._run_next_agent_execution.__func__.__globals__,
            {"summarise_message_history": summarizer},
        ):
            out = a._run_next_agent_execution()

        assert out == {"final_message": "next done"}
        assert next_agent.message_history["first_input"]["message"] == "Previous agent summary:\nshort summary"
        summarizer.assert_called_once_with(barebone, a.message_history)
        next_agent.execution.assert_called_once_with()

        other_next = _minimal_agent(name="OtherNext")
        other_next.execution = MagicMock(return_value={"final_message": "other done"})
        b = _minimal_agent(next_agent=other_next, summarize_final=False)
        b.run_simple = MagicMock(return_value=("plain final", "plain final"))

        assert b._run_next_agent_execution() == {"final_message": "other done"}
        assert other_next.message_history["first_input"]["message"] == "Previous agent summary:\nplain final"

    def test_simple_final_output_builds_summary_from_run_simple_result(self):
        a = _minimal_agent(system_prompt="system")
        a.run_simple = MagicMock(return_value=("final", "unused"))
        barebone = SimpleNamespace()
        a.get_barebone = MagicMock(return_value=barebone)
        a._build_final_output = MagicMock(return_value={"final_message": "final", "summary": "summary"})

        out = a._run_simple_final_output()

        assert out == {"final_message": "final", "summary": "summary"}
        a.get_barebone.assert_called_once_with(
            "system",
            [],
            parent_hierarchy=["A"],
            suppress_init_output=True,
        )
        a._build_final_output.assert_called_once_with("final", barebone)

    def test_validated_simple_execution_retries_with_feedback_until_check_passes(self):
        def requires_good(output) -> bool:
            """Return whether the output is good."""
            return output["final_message"] == "good"

        a = _minimal_agent(maxsteps=3, prompt="original prompt")
        a.final_answer_checks = [requires_good]
        a._run_simple_final_output = MagicMock(
            side_effect=[
                {"final_message": "bad"},
                {"final_message": "good"},
            ]
        )
        cli = MagicMock()

        with patch.dict(
            a._run_validated_simple_execution.__func__.__globals__,
            {"get_cli_output": MagicMock(return_value=cli)},
        ):
            out = a._run_validated_simple_execution()

        assert out == {"final_message": "good"}
        assert a._run_simple_final_output.call_count == 2
        assert a.maxsteps == 3
        assert a.prompt == "original prompt"
        cli.agent_response.assert_called_once()
        assert "requires_good" in cli.agent_response.call_args.args[1]

    def test_validated_simple_execution_returns_last_output_after_retry_budget(self):
        def always_fail(_output) -> bool:
            """Always reject the final answer."""
            return False

        a = _minimal_agent(maxsteps=3, prompt="original prompt")
        a.logger = MagicMock()
        a.final_answer_checks = [always_fail]
        final_output = {"final_message": "still bad"}
        a._run_simple_final_output = MagicMock(return_value=final_output)
        cli = MagicMock()

        with patch.dict(
            a._run_validated_simple_execution.__func__.__globals__,
            {"get_cli_output": MagicMock(return_value=cli)},
        ):
            out = a._run_validated_simple_execution()

        assert out is final_output
        assert a._run_simple_final_output.call_count == 4
        assert a.maxsteps == 3
        assert a.prompt == "original prompt"
        assert cli.agent_response.call_count == 3
        a.logger.log_action.assert_called_once()

    def test_execution_dispatches_stages_next_agent_checkpoint_and_validation_paths(self):
        staged = _minimal_agent(Stages=[IkaStage("S", "P", [])])
        staged._run_staged_execution = MagicMock(return_value={"final_message": "staged"})
        assert staged.execution() == {"final_message": "staged"}
        staged._run_staged_execution.assert_called_once_with(None, None, None, "")

        next_agent = _minimal_agent(name="Next")
        chained = _minimal_agent(next_agent=next_agent)
        chained._run_next_agent_execution = MagicMock(return_value={"final_message": "next"})
        assert chained.execution() == {"final_message": "next"}

        simple = _minimal_agent()
        resume_history = _minimal_agent().message_history
        simple._resume_checkpoint = {"scope": "agent", "remaining_steps": 0, "message_history": resume_history}
        simple._run_simple_final_output = MagicMock(return_value={"final_message": "simple"})
        assert simple.execution() == {"final_message": "simple"}
        assert simple.message_history is resume_history
        assert simple.maxsteps == 1

        validated = _minimal_agent()
        validated.final_answer_checks = [lambda _out: True]
        validated._run_validated_simple_execution = MagicMock(return_value={"final_message": "validated"})
        assert validated.execution() == {"final_message": "validated"}

    def test_simple_end_instruction_requires_submit_tool_first(self):
        a = _minimal_agent(tools=[SimpleNamespace(name="submit_discovery")])

        instruction = a._simple_end_instruction()

        assert "submit_discovery" in instruction
        assert "DO NOT call agent_end before" in instruction


class TestRuntimeHelpers:
    def test_prompt_hitl_question_logs_and_raises_payload(self):
        a = _minimal_agent()
        a.logger = MagicMock()
        human_input_required = a._prompt_hitl_question.__func__.__globals__["HumanInputRequired"]

        with pytest.raises(human_input_required) as exc_info:
            a._prompt_hitl_question("Review", "Proceed?", stage_index=2, remaining_steps=5)

        assert exc_info.value.payload["question"] == "Proceed?"
        assert exc_info.value.payload["stage_index"] == 2
        a.logger.log_hitl_question.assert_called_once_with("Review", "Proceed?")

    def test_rate_limit_sleeps_for_remaining_interval(self):
        a = _minimal_agent()
        a._last_tool_ts_search = 9.5

        with patch.object(a._enforce_rate_limit.__func__.__globals__["time"], "time", side_effect=[10.0, 10.1]):
            with patch.object(a._enforce_rate_limit.__func__.__globals__["time"], "sleep") as sleep:
                a._enforce_rate_limit(60, "_last_tool_ts_search")

        sleep.assert_called_once_with(0.5)
        assert a._last_tool_ts_search == 10.1

    def test_get_chat_client_returns_async_client_without_constructing_httpx(self):
        client = object()
        a = _minimal_agent(use_async=True)
        a.client = client

        assert a._get_chat_client() is client


class TestCheckpointHelpers:
    def test_stage_checkpoint_skips_when_store_missing_or_stage_disabled(self, tmp_path):
        stage = IkaStage("S", "Prompt", [], checkpoint=False)
        a = _minimal_agent(Stages=[stage])
        assert a._save_stage_checkpoint(0, remaining_steps=2, last_content="x") is None

        with_store = _minimal_agent(
            Stages=[stage],
            checkpoint=True,
            checkpoint_db_path=str(tmp_path / "checkpoints.db"),
        )
        assert with_store._save_stage_checkpoint(0, remaining_steps=2, last_content="x") is None

    def test_stage_checkpoint_persists_logs_and_emits(self, tmp_path):
        stage = IkaStage("S", "Prompt", [], checkpoint=True)
        a = _minimal_agent(
            Stages=[stage],
            checkpoint=True,
            checkpoint_db_path=str(tmp_path / "checkpoints.db"),
        )
        a.logger = MagicMock(level=2)
        cli = MagicMock()

        with patch.dict(
            a._emit_stage_checkpoint.__func__.__globals__,
            {"get_cli_output": MagicMock(return_value=cli)},
        ):
            uid = a._save_stage_checkpoint(
                0,
                remaining_steps=3,
                last_content="partial",
                extra_payload={"interrupt_data": {"question": "Proceed?"}},
            )

        assert uid
        saved = CheckpointStore(str(tmp_path / "checkpoints.db")).load_checkpoint(uid)
        assert saved["last_content"] == "partial"
        assert saved["interrupt_data"]["question"] == "Proceed?"
        a.logger.log_json.assert_called_once()
        cli.emit.assert_called_once()

    def test_agent_checkpoint_and_load_checkpoint_round_trip(self, tmp_path):
        a = _minimal_agent(checkpoint=True, checkpoint_db_path=str(tmp_path / "checkpoints.db"))

        uid = a._save_agent_checkpoint(remaining_steps=4, last_content="state")
        loaded = a.load_checkpoint(uid)

        assert loaded["scope"] == "agent"
        assert loaded["last_content"] == "state"
        assert a._resume_checkpoint == loaded

    def test_checkpoint_helpers_return_none_without_store(self):
        a = _minimal_agent()

        assert a._save_agent_checkpoint(remaining_steps=4, last_content="state") is None
        assert a.load_checkpoint("missing") is None


class TestParseControlCalls:
    def test_extract_json_from_text_handles_direct_json_code_blocks_and_invalid_text(self):
        a = _minimal_agent()

        assert a._extract_json_from_text('{"ok": true}') == '{"ok": true}'
        assert a._extract_json_from_text('prefix ```json\n{"ok": true}\n``` suffix') == '{"ok": true}'
        assert a._extract_json_from_text("   ") is None
        assert a._extract_json_from_text("bad {not-json}") is None
        assert a._extract_json_from_text("no json here") is None

    def test_parse_tool_arguments_rejects_malformed_and_non_dict_payloads(self):
        a = _minimal_agent()

        assert a._parse_tool_arguments('{"value": 1}') == {"value": 1}
        assert a._parse_tool_arguments("{bad json") == {}
        assert a._parse_tool_arguments("[1, 2]") == {}
        assert a._parse_tool_arguments({"value": 1}) == {"value": 1}
        assert a._parse_tool_arguments(["not", "dict"]) == {}

    def test_agent_end_detected(self):
        a = _minimal_agent()
        calls = [{"function": {"name": "agent_end", "arguments": '{"input": "Final answer"}'}}]
        target, ended, text = a.parse_control_calls(calls, None)
        assert ended is True
        assert "Final answer" in (text or "")

    def test_agent_end_uses_alternate_meaningful_argument(self):
        a = _minimal_agent()
        calls = [{"function": {"name": "agent_end", "arguments": '{"input": ".", "result": "Meaningful final answer"}'}}]

        _, ended, text = a.parse_control_calls(calls, None)

        assert ended is True
        assert text == "Meaningful final answer"

    def test_agent_end_uses_json_string_or_scalar_fallback_arguments(self):
        a = _minimal_agent()

        _, _, json_text = a.parse_control_calls(
            [{"function": {"name": "agent_end", "arguments": {"input": ".", "payload": '{"answer": 1}'}}}],
            None,
        )
        _, _, scalar_text = a.parse_control_calls(
            [{"function": {"name": "agent_end", "arguments": {"input": ".", "code": 42}}}],
            None,
        )

        assert json_text == '{"answer": 1}'
        assert scalar_text == "42"

    def test_agent_end_without_arguments_or_response_content_is_rejected(self):
        a = _minimal_agent()

        with pytest.raises(ValueError, match="empty or invalid arguments"):
            a.parse_control_calls(
                [{"function": {"name": "agent_end", "arguments": {"input": "."}}}],
                None,
            )

    def test_agent_end_extracts_json_from_response_content_when_arguments_empty(self):
        a = _minimal_agent()
        calls = [{"function": {"name": "agent_end", "arguments": '{"input": "."}'}}]

        _, ended, text = a.parse_control_calls(
            calls,
            None,
            response_content='Here is the result:\n```json\n{"answer": 42}\n```',
        )

        assert ended is True
        assert text == '{"answer": 42}'

    @pytest.mark.parametrize("bad_input", ["{}", "[]", "null", '""', "''"])
    def test_agent_end_rejects_empty_json_like_values(self, bad_input):
        a = _minimal_agent()

        with pytest.raises(ValueError, match="invalid/empty content"):
            a.parse_control_calls(
                [{"function": {"name": "agent_end", "arguments": {"input": bad_input}}}],
                None,
            )

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
        assert target == 0

    def test_change_stage_ignores_forward_or_unallowed_targets(self):
        stage = IkaStage("S", "P", [], allowed_back_to=[0])
        a = _minimal_agent(Stages=[stage, IkaStage("S1", "P1", []), IkaStage("S2", "P2", [])])

        forward_target, _, _ = a.parse_control_calls(
            [{"function": {"name": "change_stage", "arguments": '{"stage_index": 2}'}}],
            stage,
            current_stage_idx=1,
        )
        unallowed_target, _, _ = a.parse_control_calls(
            [{"function": {"name": "change_stage", "arguments": '{"stage_index": 1}'}}],
            stage,
            current_stage_idx=2,
        )
        malformed_target, _, _ = a.parse_control_calls(
            [{"function": {"name": "change_stage", "arguments": '{"stage_index": "not-int"}'}}],
            stage,
            current_stage_idx=2,
        )

        assert forward_target is None
        assert unallowed_target is None
        assert malformed_target is None

    def test_memory_control_calls_are_logged_and_ignored(self):
        a = _minimal_agent()
        a.logger = MagicMock()

        target, ended, text = a.parse_control_calls(
            [{"function": {"name": "short_term_save", "arguments": '{"data": "x"}'}}],
            None,
        )

        assert target is None
        assert ended is False
        assert text is None
        a.logger.log_action.assert_called_once_with("short_term_save called")

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

    def test_build_final_output_logs_and_emits_summary(self):
        a = _minimal_agent(summarize_final=True)
        a.logger = MagicMock()
        a._parent_hierarchy = ["Parent"]
        mock_bm = MagicMock(model_id="gpt-4o")
        cli = MagicMock()
        cli.get_step.return_value = 3

        with patch.dict(
            a._build_final_output.__func__.__globals__,
            {
                "summarise_message_history": MagicMock(return_value="summary text"),
                "get_cli_output": MagicMock(return_value=cli),
            },
        ):
            out = a._build_final_output("", mock_bm)

        assert out["final_message"] == "summary text"
        assert out["summary"] == "summary text"
        a.logger.log_summary.assert_called_once_with("summary text")
        cli.summarization.assert_called_once_with("A", "summary text", ["Parent", "A"], step=3)


class TestChatWrapper:
    def test_chat_wrapper_uses_sync_chat_and_accumulates_totals(self):
        a = _minimal_agent()
        barebone = MagicMock()
        response = {
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3, "input_cached_tokens": 4},
            "cost": {"input_cost": 0.1, "output_cost": 0.2, "total_cost": 0.3},
        }

        chat_mock = MagicMock(return_value=response)
        with patch.dict(a._run_chat_round.__func__.__globals__, {"chat": chat_mock}):
            out = a.chat_wrapper(
                barebone,
                [{"role": "user", "content": "task"}],
                tool_executors={},
                max_tool_calls=7,
                current_stage_index=1,
                total_stages=2,
            )

        assert out is response
        assert a._total_usage["total_tokens"] == 3
        assert a._total_usage["input_cached_tokens"] == 4
        assert a._total_cost["total_cost"] == 0.3
        chat_mock.assert_called_once()
        assert chat_mock.call_args.kwargs["max_tool_calls"] == 7

    def test_chat_wrapper_accumulates_totals_from_agent_end_exception(self):
        a = _minimal_agent()
        response = {
            "usage": {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
            "cost": {"input_cost": 0.5, "output_cost": 0.6, "total_cost": 1.1},
        }
        agent_end_exception = a.chat_wrapper.__func__.__globals__["AgentEndException"]
        a._run_chat_round = MagicMock(side_effect=agent_end_exception(response=response))

        with pytest.raises(agent_end_exception):
            a.chat_wrapper(MagicMock(), [{"role": "user", "content": "task"}])

        assert a._total_usage["total_tokens"] == 11
        assert a._total_cost["total_cost"] == 1.1

    def test_run_chat_round_uses_async_chat_when_enabled(self):
        a = _minimal_agent(use_async=True)
        response = {"usage": {}, "cost": {}}
        calls = []

        async def fake_async_chat(*args, **kwargs):
            calls.append((args, kwargs))
            return response

        with patch.dict(a._run_chat_round.__func__.__globals__, {"async_chat": fake_async_chat}):
            out = a._run_chat_round(
                MagicMock(),
                [{"role": "user", "content": "task"}],
                a.message_history,
                tool_executors={},
                logger=None,
                timeout=1,
                max_tool_rounds=2,
                max_tool_calls=3,
                current_stage_index=0,
                total_stages=1,
                client=None,
            )

        assert out is response
        assert calls[0][1]["max_tool_rounds"] == 2
