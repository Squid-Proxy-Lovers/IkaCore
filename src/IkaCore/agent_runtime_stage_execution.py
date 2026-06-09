# pyright: strict

"""Staged execution-loop mixins."""

from __future__ import annotations

import time
from typing import Any, Optional

from IkaCore.execution_types import AgentChatTurn, StageExecutionResult, StageRuntimeContext

from .agent_runtime_foundation import RuntimePreparationMixin


class StageControlMixin(RuntimePreparationMixin):
    def _parse_stage_control(
        self,
        turn: AgentChatTurn,
        stage: Any,
        stage_index: int,
    ) -> tuple[Any, bool, Optional[str]]:
        try:
            target_stage, agent_end_called, agent_end_text = self.parse_control_calls(
                turn.executed_tool_calls if turn.executed_tool_calls else turn.tool_calls,
                stage,
                stage_index,
                response_content=turn.content_before_tools,
            )
        except ValueError as e:
            error_msg = str(e)
            if self.logger:
                self.logger.log_action(f"ERROR in agent_end: {error_msg}")
            raise

        if turn.agent_end_exception:
            agent_end_called = True
        return target_stage, agent_end_called, agent_end_text

    def _finish_stage_agent_end(
        self,
        stage_index: int,
        agent_end_text: Optional[str],
        turn: AgentChatTurn,
        used_steps: int,
    ) -> StageExecutionResult:
        try:
            final_content = self._fallback_final_content(
                agent_end_text,
                turn.content_before_tools,
                turn.last_content,
            )
        except ValueError as e:
            if self.logger:
                self.logger.log_action(f"ERROR: {str(e)}")
            raise
        return StageExecutionResult(stage_index, final_content, True, final_content, used_steps)


class StageExecutionStepMixin(StageControlMixin):
    def _run_stage_turn(
        self,
        runtime: StageRuntimeContext,
        stage_index: int,
    ) -> tuple[AgentChatTurn, float]:
        if self.logger:
            self.logger.log_action(f"stage_start:{runtime.stage.name}")
        self._enforce_rate_limit_model()
        step_start = time.time()
        turn = self._run_chat_turn(
            runtime.barebone_model,
            runtime.messages,
            runtime.tool_executors,
            current_stage_index=stage_index,
            total_stages=len(self.Stages) if self.Stages else 0,
            context_label="execute_stage",
        )
        return turn, step_start

    def _stage_terminal_result(
        self,
        stage_index: int,
        remaining_steps: int,
        used_steps: int,
        turn: AgentChatTurn,
    ) -> Optional[StageExecutionResult]:
        stage = self.Stages[stage_index]
        interrupt_result = self._handle_stage_interrupt(stage, stage_index, remaining_steps, used_steps, turn)
        if interrupt_result:
            return interrupt_result

        target_stage, agent_end_called, agent_end_text = self._parse_stage_control(turn, stage, stage_index)
        if agent_end_called:
            return self._finish_stage_agent_end(stage_index, agent_end_text, turn, used_steps)
        if target_stage == "next":
            return StageExecutionResult(stage_index + 1, turn.last_content, False, None, used_steps)
        if isinstance(target_stage, int):
            return StageExecutionResult(target_stage, turn.last_content, False, None, used_steps)
        return None

    def _record_stage_turn_progress(
        self,
        runtime: StageRuntimeContext,
        stage_index: int,
        remaining_steps: int,
        used_steps: int,
        turn: AgentChatTurn,
        step_start: float,
    ) -> None:
        runtime.messages.append({"role": "assistant", "content": turn.last_content})
        if self.logger:
            self.logger.log_step(
                stage_name=runtime.stage.name,
                step_idx=used_steps,
                output=turn.last_content,
                tool_calls=turn.tool_calls,
                usage=turn.response.get("usage", {}),
                cost=turn.response.get("cost", {}),
                elapsed=time.time() - step_start,
            )
        remaining_after = max(0, remaining_steps - used_steps)
        if self.checkpoint:
            self._save_stage_checkpoint(stage_index, remaining_after, turn.last_content)


class StageExecutionRuntimeMixin(StageExecutionStepMixin):
    def execute_stage(
        self,
        stage_index: int,
        remaining_steps: int,
        resume_input: Optional[str] = None,
    ) -> StageExecutionResult:
        runtime = self._prepare_stage_runtime(stage_index, remaining_steps, resume_input)
        last_content = ""
        used_steps = 0
        step_limit = runtime.step_limit
        extension_count = 0
        while used_steps < step_limit:
            turn, step_start = self._run_stage_turn(runtime, stage_index)
            last_content = turn.last_content
            used_steps += 1

            terminal_result = self._stage_terminal_result(stage_index, remaining_steps, used_steps, turn)
            if terminal_result:
                return terminal_result
            self._record_stage_turn_progress(runtime, stage_index, remaining_steps, used_steps, turn, step_start)

            if used_steps >= step_limit:
                step_limit, extension_count, last_content, should_break = self._maybe_extend_stage(
                    runtime.barebone_model,
                    runtime.messages,
                    step_limit,
                    extension_count,
                    last_content,
                )
                if should_break:
                    break

        if self.logger:
            self.logger.log_stage_end(runtime.stage.name, used_steps)
        return StageExecutionResult(stage_index + 1, last_content, False, None, used_steps)
