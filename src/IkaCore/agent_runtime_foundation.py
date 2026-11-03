# pyright: strict

"""Shared execution-runtime state and preparation mixins."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, cast

from IkaCore.agent_execution_helpers import (
    build_stage_content_prompt,
    resolve_stage_model_overrides,
    stage_step_limit,
)
from IkaCore.agent_runtime_payloads import JsonDict, history_message_text, history_section
from IkaCore.cli_output import get_cli_output
from IkaCore.execution_types import AgentChatTurn, SimpleRuntimeContext, StageExecutionResult, StageRuntimeContext
from IkaCore.prompts import AGENT_END_INSTRUCTION
from IkaModel.base import AgentEndException
from IkaModel.chat_interface.chat_interface import (
    run_summarization as _run_summarization_untyped,  # pyright: ignore[reportUnknownVariableType]
)

ToolExecutor = Callable[..., Any]
RunSummarization = Callable[..., str]
run_summarization = cast(RunSummarization, _run_summarization_untyped)


class _RuntimeValueState:
    name: str
    description: str
    prompt: str
    system_prompt: Optional[str]
    tools: list[Any]
    Stages: list[Any]
    subagents: list[Any]
    next_agent: Any | None
    maxsteps: int
    step_timeout: int
    max_tool_rounds: int
    max_tool_calls: int
    max_step_extensions: int
    extend_steps_by: int
    max_stage_extensions: int
    extend_stage_steps_by: int
    checkpoint: bool
    memory_access: dict[str, bool]
    summarize_final: bool
    final_answer_checks: list[Callable[[JsonDict], bool]]
    logger: Any
    message_history: JsonDict
    _tool_call_counts: dict[str, int]
    _last_interrupt: Optional[JsonDict]
    context_budget: Optional[int]


class _RuntimeChatServices:
    def chat_wrapper(
        self,
        barebone_model: Any,
        messages: list[JsonDict],
        tool_executors: Optional[dict[str, ToolExecutor]] = None,
        logger: Optional[Any] = None,
        timeout: float = 900.0,
        max_tool_rounds: int = 5,
        max_tool_calls: Optional[int] = None,
        current_stage_index: Optional[int] = None,
        total_stages: int = 0,
        client: Optional[Any] = None,
    ) -> JsonDict:
        ...

    def _get_chat_client(self) -> Optional[Any]:
        ...

    def _save_stage_checkpoint(
        self,
        stage_index: int,
        remaining_steps: int,
        last_content: str,
        scope: str = "stage",
        extra_payload: Optional[JsonDict] = None,
    ) -> Any:
        ...

    def _save_agent_checkpoint(self, remaining_steps: int, last_content: str) -> Any:
        ...

    def _reset_tool_call_counts(self) -> None:
        ...

    def _enforce_rate_limit_model(self) -> None:
        ...


class _RuntimeBuildServices:
    def get_barebone(
        self,
        system_prompt: str,
        agent_tools: list[Any],
        parent_hierarchy: Optional[list[str]] = None,
        suppress_init_output: bool = False,
        model_overrides: Optional[JsonDict] = None,
        content_prompt_override: Optional[str] = None,
    ) -> Any:
        ...

    def build_stage(self, stage: Any) -> list[Any]:
        ...

    def build_simple_tools(self) -> list[Any]:
        ...

    def build_tool_executors(
        self,
        tools: list[Any],
        memory_access: Optional[dict[str, bool]] = None,
        long_term_filter: Optional[Callable[..., Any]] = None,
        subagents: Optional[list[Any]] = None,
        parent_hierarchy: Optional[list[str]] = None,
        stage: Optional[Any] = None,
        stage_index: Optional[int] = None,
        remaining_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        ...

    def final_prompt(self, stage: Any) -> str:
        ...


class _RuntimeControlServices:
    def parse_control_calls(
        self,
        tool_calls: list[JsonDict],
        stage: Optional[Any],
        current_stage_idx: int = 0,
        response_content: Optional[str] = None,
    ) -> tuple[Any, bool, Optional[str]]:
        ...

    def _fallback_final_content(
        self,
        agent_end_text: Optional[str],
        content_before_tools: str,
        last_content: str,
    ) -> str:
        ...

    def _build_final_output(self, final_message: str, barebone_model: Any) -> JsonDict:
        ...

    def load_checkpoint(self, uid: str) -> None:
        ...


if TYPE_CHECKING:

    class _RuntimeState(
        _RuntimeValueState,
        _RuntimeChatServices,
        _RuntimeBuildServices,
        _RuntimeControlServices,
    ):
        pass

else:

    class _RuntimeState:
        pass


class AgentChatTurnMixin(_RuntimeState):
    def _run_chat_turn(
        self,
        barebone_model: Any,
        messages: list[JsonDict],
        tool_executors: dict[str, Any],
        current_stage_index: Optional[int],
        total_stages: int,
        context_label: str,
    ) -> AgentChatTurn:
        agent_end_exception = False
        response: JsonDict
        try:
            response = self.chat_wrapper(
                barebone_model,
                messages,
                tool_executors=tool_executors,
                logger=self.logger,
                timeout=self.step_timeout,
                max_tool_rounds=self.max_tool_rounds,
                max_tool_calls=self.max_tool_calls,
                current_stage_index=current_stage_index,
                total_stages=total_stages,
                client=self._get_chat_client(),
            )
        except AgentEndException as exc:
            if self.logger:
                self.logger.log_action(f"Caught AgentEndException in {context_label}: {exc}")
            response = exc.response
            agent_end_exception = True

        raw_message_history = response.get("message_history")
        if isinstance(raw_message_history, dict):
            self.message_history = cast(JsonDict, raw_message_history)
        raw_tool_call_counts = getattr(barebone_model, '_tool_call_counts', None)
        if isinstance(raw_tool_call_counts, dict):
            self._tool_call_counts.update(cast(dict[str, int], raw_tool_call_counts))

        return AgentChatTurn(
            response=response,
            agent_end_exception=agent_end_exception,
            last_content=str(response.get("content", "")),
            content_before_tools=str(response.get("content_before_tools", "")),
            tool_calls=cast(list[JsonDict], response.get("tool_calls", []) or []),
            executed_tool_calls=cast(list[JsonDict], response.get("executed_tool_calls", []) or []),
        )


class StageInterruptMixin(AgentChatTurnMixin):
    def _handle_stage_interrupt(
        self,
        stage: Any,
        stage_index: int,
        remaining_steps: int,
        used_steps: int,
        turn: AgentChatTurn,
    ) -> Optional[StageExecutionResult]:
        if not turn.response.get("interrupted"):
            return None

        raw_interrupt_data = turn.response.get("interrupt_data", {})
        interrupt_data = cast(JsonDict, raw_interrupt_data) if isinstance(raw_interrupt_data, dict) else {}
        remaining_after = max(0, remaining_steps - used_steps)
        checkpoint_uid = None
        if self.logger:
            self.logger.log_hitl_prompt(stage.name)
        if self.checkpoint:
            checkpoint_uid = self._save_stage_checkpoint(
                stage_index,
                remaining_after,
                turn.last_content,
                scope="hitl",
                extra_payload={
                    "interrupt_data": deepcopy(interrupt_data),
                    "interrupt_status": turn.response.get("status", "awaiting_user_input"),
                },
            )
        self._last_interrupt = {
            "status": turn.response.get("status", "awaiting_user_input"),
            "stage_index": stage_index,
            "stage_name": stage.name,
            "remaining_steps": remaining_after,
            "checkpoint_uid": checkpoint_uid,
            "interrupt_data": deepcopy(interrupt_data),
            "message_history": deepcopy(self.message_history),
            "last_content": turn.last_content,
        }
        return StageExecutionResult(stage_index, turn.last_content, False, None, used_steps)


class StageExtensionMixin(StageInterruptMixin):
    def _maybe_extend_stage(
        self,
        barebone_model: Any,
        messages: list[JsonDict],
        step_limit: int,
        extension_count: int,
        last_content: str,
    ) -> tuple[int, int, str, bool]:
        message_history = self.message_history
        if extension_count < self.max_stage_extensions:
            what_remains = run_summarization(
                barebone_model,
                message_history,
                prompt_kind="what_remains",
                write_to_history=False,
            ) or "(no summary)"
            redirect = (
                "CRITICAL: You have used all allocated steps for this stage. "
                "Summary of what remains:\n\n"
                f"{what_remains}\n\n"
                "Complete the stage or call stage_end / agent_end. "
                f"You have {self.extend_stage_steps_by} additional steps."
            )
            messages.append({"role": "user", "content": redirect})
            return step_limit + self.extend_stage_steps_by, extension_count + 1, last_content, False

        force_answer = run_summarization(
            barebone_model,
            message_history,
            prompt_kind="force_answer",
            write_to_history=False,
        )
        if force_answer and force_answer.strip():
            last_content = force_answer.strip()
        return step_limit, extension_count, last_content, True


class StageRuntimeSupportMixin(StageExtensionMixin):
    pass


class SimpleInstructionMixin(StageRuntimeSupportMixin):
    def _simple_end_instruction(self) -> str:
        submit_tools = [
            getattr(t, "name", "") for t in (self.tools or [])
            if getattr(t, "name", "").startswith("submit_")
        ]
        if not submit_tools:
            return AGENT_END_INSTRUCTION
        return (
            "CRITICAL: This task uses a structured submission tool. You MUST "
            f"call the correct submit_* tool first ({', '.join(submit_tools)} "
            "are available in this task). Only after that submit_* tool returns "
            "success may you call agent_end with a short final summary.\n\n"
            "DO NOT call agent_end before the structured submit_* tool. That is "
            "a premature finish and fails the task."
        )

    def _log_simple_step(
        self,
        step_idx: int,
        output: str,
        tool_calls: list[JsonDict],
        response: JsonDict,
        elapsed: float,
    ) -> None:
        if self.logger:
            self.logger.log_step(
                stage_name="simple",
                step_idx=step_idx,
                output=output,
                tool_calls=tool_calls,
                usage=response.get("usage", {}),
                cost=response.get("cost", {}),
                elapsed=elapsed,
            )


class SimpleRunExtensionMixin(SimpleInstructionMixin):
    def _maybe_extend_simple_run(
        self,
        system_prompt: str,
        messages: list[JsonDict],
        current_hierarchy: list[str],
        current_step: int,
        extension_count: int,
        cli: Any,
    ) -> int:
        if extension_count >= self.max_step_extensions:
            return extension_count

        barebone_summary = self.get_barebone(
            system_prompt,
            [],
            parent_hierarchy=current_hierarchy,
            suppress_init_output=True,
        )
        message_history = self.message_history
        summary = run_summarization(
            barebone_summary,
            message_history,
            prompt_kind="what_remains",
            write_to_history=False,
        ) or "(no summary)"
        redirect = (
            "CRITICAL: You have used all allocated steps without completing the task. "
            "You MUST refer back to your original prompt and complete the original goal. "
            "Do NOT repeat the same tool calls. Summary of the conversation so far:\n\n"
            f"{summary}\n\n"
            "Complete the task now: use submit_discovery with your findings if you have not already, "
            "then call agent_end with your final answer. "
            f"You have {self.extend_steps_by} additional steps."
        )
        messages.append({"role": "user", "content": redirect})
        self.maxsteps += self.extend_steps_by
        cli.agent_response(
            self.name,
            f"Max steps reached. Injected redirect and extended by {self.extend_steps_by} steps. "
            "Complete the original goal.",
            current_hierarchy,
            step=current_step,
            is_final=False,
        )
        return extension_count + 1


class SimpleStepExtensionMixin(SimpleRunExtensionMixin):
    def _finalize_simple_after_max_steps(
        self,
        system_prompt: str,
        current_hierarchy: list[str],
        cli: Any,
        last_agent_end_text: Optional[str],
        last_content: str,
    ) -> tuple[str, str]:
        final_step = cli.get_step(self.name) or self.maxsteps
        if last_agent_end_text:
            final_response = last_agent_end_text
        else:
            barebone_final = self.get_barebone(
                system_prompt,
                [],
                parent_hierarchy=current_hierarchy,
                suppress_init_output=True,
            )
            message_history = self.message_history
            force_answer = run_summarization(
                barebone_final,
                message_history,
                prompt_kind="force_answer",
                write_to_history=False,
            )
            final_response = force_answer.strip() if force_answer else last_content
            if not final_response:
                final_response = last_content
        cli.agent_response(
            self.name,
            f"Reached max steps ({self.maxsteps}). Returning last content.\n\n{final_response}",
            current_hierarchy,
            step=final_step,
            is_final=True,
        )
        return final_response, final_response


class SimpleRuntimeSupportMixin(SimpleStepExtensionMixin):
    pass


class StageRuntimeModelBuildMixin(SimpleRuntimeSupportMixin):
    def _stage_system_prompt(self, stage: Any) -> str:
        base_system = self.final_prompt(stage)
        return (self.system_prompt + "\n\n" + base_system) if self.system_prompt else base_system

    def _stage_current_hierarchy(self, stage: Any, stage_index: int) -> list[str]:
        return getattr(self, "_parent_hierarchy", []) + [self.name, f"Stage {stage_index}: {stage.name}"]

    def _build_stage_barebone_model(
        self,
        stage: Any,
        stage_index: int,
        system_prompt: str,
        agent_tools: list[Any],
        current_hierarchy: list[str],
        content_prompt: str,
    ) -> Any:
        model_overrides = resolve_stage_model_overrides(self, stage)
        barebone_model = self.get_barebone(
            system_prompt,
            agent_tools,
            parent_hierarchy=current_hierarchy,
            suppress_init_output=(stage_index != 0),
            model_overrides=model_overrides if model_overrides else None,
            content_prompt_override=content_prompt,
        )
        barebone_model._tool_call_counts = {}
        if getattr(self, 'context_budget', None):
            barebone_model.context_budget = self.context_budget
        return barebone_model


class StageRuntimeBuildMixin(StageRuntimeModelBuildMixin):
    def _stage_initial_messages(
        self,
        stage: Any,
        resume_input: Optional[str],
        message_history: JsonDict,
        content_prompt: str,
    ) -> list[JsonDict]:
        if resume_input is None:
            return [{"role": "user", "content": content_prompt}]
        if self.logger:
            self.logger.log_hitl_input(stage.name, resume_input)
        if resume_input.strip():
            messages_bucket = history_section(message_history, "messages")
            messages_bucket[str(uuid.uuid4())] = {
                "message": resume_input,
                "tokens": 0,
                "type": "hitl_input",
            }
        return [{"role": "user", "content": resume_input}]

    def _stage_tool_executors(
        self,
        stage: Any,
        stage_index: int,
        remaining_steps: int,
        current_hierarchy: list[str],
    ) -> dict[str, Any]:
        self._reset_tool_call_counts()
        stage_memory_access = getattr(stage, "memory_access", None) or self.memory_access
        return self.build_tool_executors(
            stage.tools,
            memory_access=stage_memory_access,
            long_term_filter=getattr(stage, "long_term_filter", None),
            subagents=getattr(stage, "subagents", None),
            parent_hierarchy=current_hierarchy,
            stage=stage,
            stage_index=stage_index,
            remaining_steps=remaining_steps,
        )


class StageRuntimePreparationMixin(StageRuntimeBuildMixin):
    def _prepare_stage_runtime(
        self,
        stage_index: int,
        remaining_steps: int,
        resume_input: Optional[str],
    ) -> StageRuntimeContext:
        message_history = self.message_history
        stage = self.Stages[stage_index]
        system_prompt = self._stage_system_prompt(stage)
        history_section(message_history, "system")["message"] = system_prompt

        agent_tools = self.build_stage(stage)
        current_hierarchy = self._stage_current_hierarchy(stage, stage_index)
        content_prompt = build_stage_content_prompt(self.prompt, self.Stages, stage, stage_index)

        first_input = history_section(message_history, "first_input")
        first_input["message"] = content_prompt
        first_input["tokens"] = 0
        barebone_model = self._build_stage_barebone_model(
            stage,
            stage_index,
            system_prompt,
            agent_tools,
            current_hierarchy,
            content_prompt,
        )
        messages = self._stage_initial_messages(stage, resume_input, message_history, content_prompt)
        tool_executors = self._stage_tool_executors(stage, stage_index, remaining_steps, current_hierarchy)
        step_limit = stage_step_limit(stage, remaining_steps)
        if self.logger:
            self.logger.log_stage_start(stage.name, getattr(stage, "hitl", False), remaining_steps, step_limit)

        return StageRuntimeContext(
            stage=stage,
            system_prompt=system_prompt,
            content_prompt=content_prompt,
            current_hierarchy=current_hierarchy,
            barebone_model=barebone_model,
            messages=messages,
            tool_executors=tool_executors,
            step_limit=step_limit,
        )


class SimpleRuntimePreparationMixin(StageRuntimePreparationMixin):
    def _prepare_simple_runtime(self) -> SimpleRuntimeContext:
        dynamic_tools = self.build_simple_tools()
        system_prompt = self.system_prompt or ""
        current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
        cli = get_cli_output()
        cli.set_step(self.name, 1)

        message_history = self.message_history
        first_msg = history_message_text(message_history, "first_input")
        start_prompt = (first_msg or self.prompt or "") + "\n\n" + self._simple_end_instruction()
        barebone_model = self.get_barebone(
            system_prompt,
            dynamic_tools,
            parent_hierarchy=current_hierarchy,
            content_prompt_override=start_prompt,
        )
        barebone_model._tool_call_counts = {}
        if getattr(self, 'context_budget', None):
            barebone_model.context_budget = self.context_budget
        messages: list[JsonDict] = [{"role": "user", "content": start_prompt}]
        tool_executors = self.build_tool_executors(
            self.tools,
            memory_access=self.memory_access,
            subagents=self.subagents,
            parent_hierarchy=current_hierarchy,
        )
        return SimpleRuntimeContext(
            system_prompt=system_prompt,
            current_hierarchy=current_hierarchy,
            barebone_model=barebone_model,
            messages=messages,
            tool_executors=tool_executors,
            cli=cli,
        )


class RuntimePreparationMixin(SimpleRuntimePreparationMixin):
    pass
