# pyright: strict

"""Simple agent execution-loop mixins."""

from __future__ import annotations

import time
from typing import Any, Optional

from IkaCore.execution_types import AgentChatTurn, SimpleRuntimeContext

from .agent_runtime_stage_execution import StageExecutionRuntimeMixin

JsonDict = dict[str, Any]


class SimpleLoopGuardMixin(StageExecutionRuntimeMixin):
    def _completion_contract_error(self) -> str:
        """Return an actionable error when a caller-defined contract is unmet."""
        check = getattr(self, "completion_check", None)
        if not callable(check):
            return ""
        try:
            result = check()
        except Exception as exc:  # The contract itself must fail closed.
            return f"completion contract check raised {type(exc).__name__}: {exc}"
        if isinstance(result, tuple):
            valid = bool(result[0]) if result else False
            detail = str(result[1]) if len(result) > 1 and result[1] else ""
        else:
            valid = bool(result)
            detail = ""
        if valid:
            return ""
        return detail or str(
            getattr(self, "completion_feedback", "Required structured submission is missing or incomplete.")
        )

    def _inject_completion_repair(
        self,
        messages: list[JsonDict],
        barebone_model: Any,
        error: str,
        step_num: int,
    ) -> int:
        """Continue the live conversation once so the model can repair submission."""
        feedback = (
            "STRUCTURED COMPLETION REQUIRED: Your attempted completion was rejected because "
            f"{error} Use the task-specific submit_* tool with a complete payload, then call "
            "agent_end again. Preserve your analysis and prior tool evidence; do not restart it."
        )
        messages.append({"role": "user", "content": feedback})
        self._tool_call_counts.pop("agent_end", None)
        model_counts = getattr(barebone_model, "_tool_call_counts", None)
        if isinstance(model_counts, dict):
            model_counts.pop("agent_end", None)
        repair_steps = max(1, int(getattr(self, "completion_repair_steps", 8)))
        self.maxsteps = max(self.maxsteps, step_num + repair_steps)
        if self.logger:
            self.logger.log_action(feedback)
        return step_num

    def _simple_turn_has_no_progress(self, turn: AgentChatTurn, step_num: int) -> bool:
        no_progress = not turn.tool_calls and not turn.last_content and not turn.executed_tool_calls
        if no_progress and step_num > 0:
            if self.logger:
                self.logger.log_action("CRITICAL: Infinite loop detected (no tools, no content). Breaking.")
            return True
        return False

    def _parse_simple_control(
        self,
        turn: AgentChatTurn,
        cli: Any,
        current_hierarchy: list[str],
        current_step: int,
    ) -> tuple[bool, Optional[str]]:
        try:
            _, agent_end_called, agent_end_text = self.parse_control_calls(
                turn.executed_tool_calls if turn.executed_tool_calls else turn.tool_calls,
                None,
                response_content=turn.content_before_tools,
            )
            if turn.agent_end_exception:
                agent_end_called = True
            return agent_end_called, agent_end_text
        except ValueError as e:
            error_msg = str(e)
            cli.agent_response(
                self.name,
                f"ERROR: {error_msg}",
                current_hierarchy,
                step=current_step,
                is_final=False,
            )
            raise


class SimpleAgentEndMixin(SimpleLoopGuardMixin):
    def _finish_simple_agent_end(
        self,
        agent_end_text: Optional[str],
        turn: AgentChatTurn,
        cli: Any,
        current_hierarchy: list[str],
        current_step: int,
        step_num: int,
        step_start: float,
    ) -> tuple[str, str]:
        try:
            final_content = self._fallback_final_content(
                agent_end_text,
                turn.content_before_tools,
                turn.last_content,
            )
        except ValueError as e:
            error_msg = str(e)
            cli.agent_response(
                self.name,
                f"ERROR: {error_msg}",
                current_hierarchy,
                step=current_step,
                is_final=False,
            )
            raise
        cli.agent_response(
            self.name,
            final_content,
            current_hierarchy,
            step=current_step,
            is_final=True,
        )
        self._log_simple_step(
            step_num,
            turn.last_content,
            turn.tool_calls,
            turn.response,
            time.time() - step_start,
        )
        return final_content, final_content


class SimpleProgressMixin(SimpleAgentEndMixin):
    def _record_simple_text_response(
        self,
        turn: AgentChatTurn,
        messages: list[JsonDict],
        cli: Any,
        current_hierarchy: list[str],
        current_step: int,
        step_num: int,
        step_start: float,
    ) -> bool:
        if turn.tool_calls or not turn.last_content:
            return False
        messages.append({"role": "assistant", "content": turn.last_content})
        cli.agent_response(
            self.name,
            turn.last_content,
            current_hierarchy,
            step=current_step,
            is_final=False,
        )
        self._log_simple_step(
            step_num,
            turn.last_content,
            [],
            turn.response,
            time.time() - step_start,
        )
        return True

    def _checkpoint_simple_progress(self, turn: AgentChatTurn, step_num: int, step_start: float) -> None:
        self._log_simple_step(
            step_num,
            turn.last_content,
            turn.tool_calls,
            turn.response,
            time.time() - step_start,
        )
        remaining_after = max(0, int(self.maxsteps) - (int(step_num) + 1))
        self._save_agent_checkpoint(remaining_after, turn.last_content)


class SimpleIterationMixin(SimpleProgressMixin):
    def _start_simple_step(
        self,
        cli: Any,
        current_hierarchy: list[str],
        step_num: int,
    ) -> tuple[int, float]:
        current_step = step_num + 1
        cli.set_step(self.name, current_step)
        if step_num == 0:
            cli.agent_init(
                self.name,
                current_hierarchy,
                step=current_step,
                description=f"Step {current_step}/{self.maxsteps}",
            )
        self._enforce_rate_limit_model()
        return current_step, time.time()

    def _finish_simple_non_terminal_step(
        self,
        turn: AgentChatTurn,
        messages: list[JsonDict],
        runtime: SimpleRuntimeContext,
        current_step: int,
        step_num: int,
        step_start: float,
        extension_count: int,
    ) -> tuple[int, int, bool]:
        recorded_text = self._record_simple_text_response(
            turn,
            messages,
            runtime.cli,
            runtime.current_hierarchy,
            current_step,
            step_num,
            step_start,
        )
        if not recorded_text:
            self._checkpoint_simple_progress(turn, step_num, step_start)

        step_num += 1
        if step_num >= self.maxsteps and extension_count < self.max_step_extensions:
            extension_count = self._maybe_extend_simple_run(
                runtime.system_prompt,
                messages,
                runtime.current_hierarchy,
                current_step,
                extension_count,
                runtime.cli,
            )
        return step_num, extension_count, recorded_text


class SimpleExecutionRuntimeMixin(SimpleIterationMixin):
    def run_simple(self) -> tuple[str, str]:
        runtime = self._prepare_simple_runtime()
        barebone_model = runtime.barebone_model
        messages = runtime.messages
        tool_executors = runtime.tool_executors
        last_content = ""
        last_agent_end_text = None
        step_num = 0
        extension_count = 0
        completion_repairs = 0
        completion_repair_limit = max(0, int(getattr(self, "completion_repair_limit", 1)))

        while True:
            while step_num < self.maxsteps:
                current_step, step_start = self._start_simple_step(
                    runtime.cli,
                    runtime.current_hierarchy,
                    step_num,
                )

                turn = self._run_chat_turn(
                    barebone_model,
                    messages,
                    tool_executors,
                    current_stage_index=None,
                    total_stages=0,
                    context_label="run_simple",
                )
                last_content = turn.last_content

                if self._simple_turn_has_no_progress(turn, step_num):
                    break

                agent_end_called, agent_end_text = self._parse_simple_control(
                    turn,
                    runtime.cli,
                    runtime.current_hierarchy,
                    current_step,
                )
                if agent_end_called and agent_end_text:
                    last_agent_end_text = agent_end_text
                if agent_end_called:
                    contract_error = self._completion_contract_error()
                    if not contract_error:
                        return self._finish_simple_agent_end(
                            agent_end_text,
                            turn,
                            runtime.cli,
                            runtime.current_hierarchy,
                            current_step,
                            step_num,
                            step_start,
                        )
                    if completion_repairs >= completion_repair_limit:
                        raise RuntimeError(f"Structured completion contract failed: {contract_error}")
                    self._log_simple_step(
                        step_num,
                        turn.last_content,
                        turn.tool_calls,
                        turn.response,
                        time.time() - step_start,
                    )
                    completion_repairs += 1
                    step_num += 1
                    self._inject_completion_repair(messages, barebone_model, contract_error, step_num)
                    continue

                step_num, extension_count, should_continue = self._finish_simple_non_terminal_step(
                    turn,
                    messages,
                    runtime,
                    current_step,
                    step_num,
                    step_start,
                    extension_count,
                )
                if should_continue:
                    continue

            contract_error = self._completion_contract_error()
            if contract_error:
                if completion_repairs >= completion_repair_limit:
                    raise RuntimeError(f"Structured completion contract failed: {contract_error}")
                completion_repairs += 1
                self._inject_completion_repair(messages, barebone_model, contract_error, step_num)
                continue

            return self._finalize_simple_after_max_steps(
                runtime.system_prompt,
                runtime.current_hierarchy,
                runtime.cli,
                last_agent_end_text,
                last_content,
            )
