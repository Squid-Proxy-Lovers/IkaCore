# pyright: strict

"""Execution dispatch and resume mixins."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Optional, cast

from IkaCore.cli_output import get_cli_output
from IkaCore.execution_types import StagedExecutionState, StageExecutionResult
from IkaModel.chat_interface.chat_interface import (
    summarise_message_history as _summarise_message_history_untyped,  # pyright: ignore[reportUnknownVariableType]
)

from .agent_runtime_simple_execution import SimpleExecutionRuntimeMixin

JsonDict = dict[str, Any]
SummariseMessageHistory = Callable[..., str]
summarise_message_history = cast(SummariseMessageHistory, _summarise_message_history_untyped)


def _history_section(message_history: JsonDict, key: str) -> JsonDict:
    raw_section = message_history.get(key)
    if isinstance(raw_section, dict):
        return cast(JsonDict, raw_section)
    section: JsonDict = {}
    message_history[key] = section
    return section


def _history_message_text(message_history: JsonDict, key: str) -> str:
    value = _history_section(message_history, key).get("message", "")
    return value if isinstance(value, str) else str(value)


def _string_attr(obj: object, attr: str, default: str = "unknown") -> str:
    value = getattr(obj, attr, default)
    return value if isinstance(value, str) else default


class InterruptOutputMixin(SimpleExecutionRuntimeMixin):
    def _interrupt_output(self, interrupt: JsonDict) -> JsonDict:
        raw_interrupt_data = interrupt.get("interrupt_data")
        question = ""
        if isinstance(raw_interrupt_data, dict):
            raw_question = cast(JsonDict, raw_interrupt_data).get("question", "")
            question = raw_question if isinstance(raw_question, str) else str(raw_question)
        raw_content = interrupt.get("last_content") or question
        interrupt_content = raw_content if isinstance(raw_content, str) else str(raw_content)
        return {
            "status": interrupt.get("status", "awaiting_user_input"),
            "final_message": interrupt_content,
            "summary": interrupt_content,
            "usage": getattr(self, "_total_usage", {}),
            "cost": getattr(self, "_total_cost", {}),
            "model_id": getattr(self, "model_id", ""),
            **interrupt,
        }


class StagedResumeMixin(InterruptOutputMixin):
    def _initial_staged_execution_state(self, last_content: str) -> StagedExecutionState:
        return StagedExecutionState(stage_idx=0, remaining_steps=self.maxsteps, last_content=last_content)

    def _apply_stage_resume_checkpoint(
        self,
        state: StagedExecutionState,
        resume_cp: JsonDict,
    ) -> None:
        stage_index = resume_cp.get("stage_index", 0)
        remaining_steps = resume_cp.get("remaining_steps", state.remaining_steps)
        state.stage_idx = min(stage_index if isinstance(stage_index, int) else 0, len(self.Stages) - 1)
        state.remaining_steps = max(1, remaining_steps if isinstance(remaining_steps, int) else state.remaining_steps)
        raw_message_history = resume_cp.get("message_history")
        if isinstance(raw_message_history, dict):
            self.message_history = cast(JsonDict, raw_message_history)
        last_content = resume_cp.get("last_content", "")
        state.last_content = last_content if isinstance(last_content, str) else str(last_content)

    def _hitl_replay_interrupt_payload(
        self,
        checkpoint_uid: Optional[str],
        resume_cp: JsonDict,
        state: StagedExecutionState,
    ) -> JsonDict:
        return {
            "status": resume_cp.get("interrupt_status", "awaiting_user_input"),
            "stage_index": state.stage_idx,
            "stage_name": self.Stages[state.stage_idx].name if 0 <= state.stage_idx < len(self.Stages) else "unknown",
            "remaining_steps": state.remaining_steps,
            "checkpoint_uid": checkpoint_uid,
            "interrupt_data": deepcopy(resume_cp.get("interrupt_data", {})),
            "message_history": deepcopy(self.message_history),
            "last_content": state.last_content,
        }

    def _apply_hitl_resume_checkpoint(
        self,
        checkpoint_uid: Optional[str],
        resume_input: Optional[str],
        state: StagedExecutionState,
        resume_cp: JsonDict,
    ) -> Optional[JsonDict]:
        self._apply_stage_resume_checkpoint(state, resume_cp)
        state.resumed_stage_index = state.stage_idx
        if resume_input is not None and resume_input.strip():
            state.resume_stage_input = resume_input
            return None
        return self._hitl_replay_interrupt_payload(checkpoint_uid, resume_cp, state)

    def _apply_staged_resume_checkpoint(
        self,
        checkpoint_uid: Optional[str],
        resume_input: Optional[str],
        resume_cp: Optional[JsonDict],
        state: StagedExecutionState,
    ) -> Optional[JsonDict]:
        if not resume_cp:
            return None
        if resume_cp.get("scope") == "stage":
            self._apply_stage_resume_checkpoint(state, resume_cp)
            return None
        if resume_cp.get("scope") == "hitl":
            return self._apply_hitl_resume_checkpoint(checkpoint_uid, resume_input, state, resume_cp)
        return None


class StagedExecutionStateMixin(StagedResumeMixin):
    def _current_stage_resume_input(self, state: StagedExecutionState) -> Optional[str]:
        if state.resumed_stage_index is not None and state.stage_idx == state.resumed_stage_index:
            return state.resume_stage_input
        return None

    def _apply_staged_result(
        self,
        state: StagedExecutionState,
        stage_result: StageExecutionResult,
        current_resume_input: Optional[str],
    ) -> bool:
        state.stage_idx = stage_result.next_stage_index
        state.last_content = stage_result.last_content
        if current_resume_input is not None:
            state.resume_stage_input = None
        state.remaining_steps -= stage_result.used_steps
        if stage_result.agent_end_called:
            state.agent_end_text = stage_result.end_text
            return False
        return True

    def _finalize_staged_execution(self, state: StagedExecutionState) -> JsonDict:
        final_message = (
            state.agent_end_text
            if state.agent_end_text and state.agent_end_text.strip() not in [".", ""]
            else state.last_content
        )
        current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
        system_message = _history_message_text(self.message_history, "system")
        barebone_model = self.get_barebone(
            system_message,
            [],
            parent_hierarchy=current_hierarchy,
            suppress_init_output=True,
        )
        return self._build_final_output(final_message, barebone_model)


class StagedExecutionDispatchMixin(StagedExecutionStateMixin):
    def _run_staged_execution(
        self,
        checkpoint_uid: Optional[str],
        resume_input: Optional[str],
        resume_cp: Optional[JsonDict],
        last_content: str,
    ) -> JsonDict:
        state = self._initial_staged_execution_state(last_content)
        interrupt_payload = self._apply_staged_resume_checkpoint(checkpoint_uid, resume_input, resume_cp, state)
        if interrupt_payload:
            self._last_interrupt = interrupt_payload
            return self._interrupt_output(interrupt_payload)

        while 0 <= state.stage_idx < len(self.Stages):
            current_resume_input = self._current_stage_resume_input(state)
            stage_result = self.execute_stage(
                state.stage_idx,
                state.remaining_steps,
                resume_input=current_resume_input,
            )
            should_continue = self._apply_staged_result(state, stage_result, current_resume_input)
            if self._last_interrupt:
                return self._interrupt_output(self._last_interrupt)
            if not should_continue:
                break

        return self._finalize_staged_execution(state)


class SimpleExecutionDispatchMixin(StagedExecutionDispatchMixin):
    def _run_next_agent_execution(self) -> JsonDict:
        next_agent = self.next_agent
        if next_agent is None:
            raise RuntimeError("next_agent execution requested without a next_agent")

        final_message, _ = self.run_simple()
        if self.summarize_final:
            current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
            barebone_model = self.get_barebone(
                self.system_prompt or self.description or self.prompt,
                [],
                parent_hierarchy=current_hierarchy,
                suppress_init_output=True,
            )
            summary = summarise_message_history(barebone_model, self.message_history) or final_message
        else:
            summary = final_message
        next_agent.message_history = {
            "system": {"message": next_agent.system_prompt or "", "tokens": 0},
            "first_input": {"message": f"Previous agent summary:\n{summary}", "tokens": 0},
            "summary": {"message": "", "tokens": 0},
            "messages": {},
        }
        return next_agent.execution()

    def _run_simple_final_output(self) -> JsonDict:
        final_message, _ = self.run_simple()
        current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
        barebone_model = self.get_barebone(
            self.system_prompt or self.description or self.prompt,
            [],
            parent_hierarchy=current_hierarchy,
            suppress_init_output=True,
        )
        return self._build_final_output(final_message, barebone_model)


class ValidatedSimpleExecutionMixin(SimpleExecutionDispatchMixin):
    def _run_validated_simple_execution(self) -> JsonDict:
        original_maxsteps = self.maxsteps
        original_prompt = self.prompt
        max_retries = 3
        retry_count = 0

        while True:
            final_output = self._run_simple_final_output()
            failed_checks: list[str] = []
            for check in self.final_answer_checks:
                if not check(final_output):
                    func_name = _string_attr(check, '__name__')
                    failed_checks.append(func_name)

            if not failed_checks:
                self.maxsteps = original_maxsteps
                self.prompt = original_prompt
                return final_output

            failed_check_names = ", ".join(failed_checks)
            if retry_count >= max_retries:
                self.maxsteps = original_maxsteps
                self.prompt = original_prompt
                if self.logger:
                    self.logger.log_action(
                        "Final answer validation failed after "
                        f"{max_retries} retries. Returning last output despite failed checks: {failed_check_names}"
                    )
                return final_output

            self.maxsteps = original_maxsteps + 10
            check_descriptions: list[str] = []
            for check in self.final_answer_checks:
                func_name = _string_attr(check, '__name__')
                if func_name in failed_checks:
                    raw_doc = getattr(check, '__doc__', None)
                    doc = raw_doc.strip() if isinstance(raw_doc, str) and raw_doc.strip() else "No description available"
                    check_descriptions.append(f"{func_name}: {doc}")

            feedback_msg = (
                f"Your previous answer failed validation checks: {failed_check_names}.\n"
                f"Failed check requirements:\n" + "\n".join(f"- {desc}" for desc in check_descriptions) + "\n"
                f"Please review the requirements and provide an improved answer. "
                f"You have {self.maxsteps} steps to complete this task."
            )
            self.prompt = f"{original_prompt}\n\n[FEEDBACK]: {feedback_msg}"
            message_history = self.message_history
            _history_section(message_history, "first_input")["message"] = self.prompt
            message_history["messages"] = {}
            retry_count += 1
            cli = get_cli_output()
            cli.agent_response(
                self.name,
                f"Validation failed for checks: {failed_check_names}. Retrying (attempt {retry_count}/{max_retries})...",
                [self.name],
                step=0,
            )


class ExecutionDispatchRuntimeMixin(ValidatedSimpleExecutionMixin):
    def execution(self, checkpoint_uid: Optional[str] = None, resume_input: Optional[str] = None) -> JsonDict:
        last_content = ""
        self._last_interrupt = None

        if checkpoint_uid:
            self.load_checkpoint(checkpoint_uid)

        raw_resume_cp = getattr(self, "_resume_checkpoint", None)
        resume_cp = cast(JsonDict, raw_resume_cp) if isinstance(raw_resume_cp, dict) else None

        if self.Stages:
            return self._run_staged_execution(checkpoint_uid, resume_input, resume_cp, last_content)

        if self.next_agent:
            return self._run_next_agent_execution()

        if resume_cp and resume_cp.get("scope") == "agent":
            raw_message_history = resume_cp.get("message_history")
            if isinstance(raw_message_history, dict):
                self.message_history = cast(JsonDict, raw_message_history)
            remaining_steps = resume_cp.get("remaining_steps", self.maxsteps)
            self.maxsteps = max(1, remaining_steps if isinstance(remaining_steps, int) else self.maxsteps)

        if not self.final_answer_checks:
            return self._run_simple_final_output()

        return self._run_validated_simple_execution()
