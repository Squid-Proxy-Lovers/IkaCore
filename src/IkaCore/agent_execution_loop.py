# pyright: strict, reportPrivateUsage=false
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from IkaModel.execution_hooks import emit_boundary

from .execution_types import (
    AgentChatTurn,
    ExecutionLoopState,
    SimpleRuntimeContext,
    StageExecutionResult,
    StageRuntimeContext,
)
from .snapshot_agent_state import restore_loop_state

"AgentTurnLoop component; owns one execution responsibility."
RuntimeContext = SimpleRuntimeContext | StageRuntimeContext
if TYPE_CHECKING:
    from .agents import IkaBaseAgent


class AgentTurnLoop:
    def __init__(self, agent: IkaBaseAgent):
        self.agent = agent

    def _start_execution_step(self, runtime: RuntimeContext, state: ExecutionLoopState, remaining_steps: int) -> float:
        agent = self.agent
        staged = isinstance(runtime, StageRuntimeContext)
        remaining = remaining_steps if staged else agent.maxsteps
        emit_boundary("step_entry", {"remaining_steps": remaining - state.used_steps})
        if isinstance(runtime, SimpleRuntimeContext):
            step = state.used_steps + 1
            runtime.cli.set_step(agent.name, step)
            if state.used_steps == 0:
                runtime.cli.agent_init(
                    agent.name, runtime.current_hierarchy, step=step, description=f"Step {step}/{agent.maxsteps}"
                )
        elif agent.logger:
            agent.logger.log_action(f"stage_start:{runtime.stage.name}")
        agent.chat._enforce_rate_limit_model()
        return time.time()

    def _report_execution_error(
        self, runtime: RuntimeContext, state: ExecutionLoopState, error: ValueError, *, parsing: bool = False
    ) -> None:
        agent = self.agent
        if isinstance(runtime, SimpleRuntimeContext):
            runtime.cli.agent_response(
                agent.name, f"ERROR: {error}", runtime.current_hierarchy, step=state.used_steps, is_final=False
            )
        elif agent.logger:
            prefix = "ERROR in agent_end" if parsing else "ERROR"
            agent.logger.log_action(f"{prefix}: {error}")

    def _parse_turn_control(
        self, runtime: RuntimeContext, state: ExecutionLoopState, turn: AgentChatTurn
    ) -> tuple[Any, bool, Optional[str]]:
        agent = self.agent
        stage = runtime.stage if isinstance(runtime, StageRuntimeContext) else None
        index = runtime.stage_index if isinstance(runtime, StageRuntimeContext) else 0
        try:
            target, ended, text = agent.parse_control_calls(
                turn.executed_tool_calls or turn.tool_calls, stage, index, response_content=turn.content_before_tools
            )
        except ValueError as error:
            agent.turns._report_execution_error(runtime, state, error, parsing=True)
            raise
        return (target, ended or turn.agent_end_exception, text)

    def _log_execution_turn(
        self, runtime: RuntimeContext, state: ExecutionLoopState, turn: AgentChatTurn, started: float
    ) -> None:
        agent = self.agent
        if agent.logger:
            staged = isinstance(runtime, StageRuntimeContext)
            agent.logger.log_step(
                stage_name=runtime.stage.name if isinstance(runtime, StageRuntimeContext) else "simple",
                step_idx=state.used_steps if staged else state.used_steps - 1,
                output=turn.last_content,
                tool_calls=turn.tool_calls,
                usage=turn.response.get("usage", {}),
                cost=turn.response.get("cost", {}),
                elapsed=time.time() - started,
            )

    def _finish_execution_turn(
        self,
        runtime: RuntimeContext,
        state: ExecutionLoopState,
        turn: AgentChatTurn,
        text: Optional[str],
        started: float,
    ) -> StageExecutionResult:
        agent = self.agent
        try:
            content = agent.chat._fallback_final_content(text, turn.content_before_tools, turn.last_content)
        except ValueError as error:
            agent.turns._report_execution_error(runtime, state, error)
            raise
        if isinstance(runtime, SimpleRuntimeContext):
            runtime.cli.agent_response(
                agent.name, content, runtime.current_hierarchy, step=state.used_steps, is_final=True
            )
            agent.turns._log_execution_turn(runtime, state, turn, started)
        index = runtime.stage_index if isinstance(runtime, StageRuntimeContext) else 0
        return StageExecutionResult(index, content, True, content, state.used_steps)

    def _terminal_turn_result(
        self,
        runtime: RuntimeContext,
        state: ExecutionLoopState,
        turn: AgentChatTurn,
        remaining_steps: int,
        started: float,
    ) -> Optional[StageExecutionResult]:
        agent = self.agent
        if isinstance(runtime, StageRuntimeContext):
            interrupted = agent.preparation._handle_stage_interrupt(
                runtime.stage, runtime.stage_index, remaining_steps, state.used_steps, turn
            )
            if interrupted is not None:
                return interrupted
        target, ended, text = agent.turns._parse_turn_control(runtime, state, turn)
        if ended:
            return agent.turns._finish_execution_turn(runtime, state, turn, text, started)
        if isinstance(runtime, StageRuntimeContext):
            if target == "next":
                return StageExecutionResult(runtime.stage_index + 1, turn.last_content, False, None, state.used_steps)
            if isinstance(target, int):
                return StageExecutionResult(target, turn.last_content, False, None, state.used_steps)
        return None

    def _record_execution_progress(
        self, runtime: RuntimeContext, state: ExecutionLoopState, turn: AgentChatTurn, started: float
    ) -> None:
        agent = self.agent
        if isinstance(runtime, StageRuntimeContext) or (not turn.tool_calls and turn.last_content):
            runtime.messages.append({"role": "assistant", "content": turn.last_content})
            if isinstance(runtime, SimpleRuntimeContext):
                runtime.cli.agent_response(
                    agent.name, turn.last_content, runtime.current_hierarchy, step=state.used_steps, is_final=False
                )
        agent.turns._log_execution_turn(runtime, state, turn, started)

    def _extend_execution(self, runtime: RuntimeContext, state: ExecutionLoopState) -> bool:
        agent = self.agent
        if isinstance(runtime, StageRuntimeContext):
            if state.used_steps < state.step_limit:
                return False
            state.step_limit, state.extension_count, state.last_content, stop = (
                agent.policy_runtime._maybe_extend_stage(
                    runtime.barebone_model,
                    runtime.messages,
                    state.step_limit,
                    state.extension_count,
                    state.last_content,
                )
            )
            return stop
        if state.used_steps >= agent.maxsteps and state.extension_count < agent.max_step_extensions:
            state.extension_count = agent.policy_runtime._maybe_extend_simple_run(
                runtime.system_prompt,
                runtime.messages,
                runtime.current_hierarchy,
                state.used_steps,
                state.extension_count,
                runtime.cli,
            )
        state.step_limit = agent.maxsteps
        return False

    def _simple_turn_has_no_progress(self, turn: AgentChatTurn, step_num: int) -> bool:
        agent = self.agent
        no_progress = not turn.tool_calls and (not turn.last_content) and (not turn.executed_tool_calls)
        if no_progress and step_num > 0:
            if agent.logger:
                agent.logger.log_action("CRITICAL: Infinite loop detected (no tools, no content). Breaking.")
            return True
        return False

    def _run_execution_loop(self, runtime: RuntimeContext, remaining_steps: int) -> StageExecutionResult:
        agent = self.agent
        staged = isinstance(runtime, StageRuntimeContext)
        state = restore_loop_state(
            agent, runtime.step_limit if isinstance(runtime, StageRuntimeContext) else agent.maxsteps, remaining_steps
        )
        remaining_steps = state.remaining_steps
        if state.used_steps >= state.step_limit:
            agent.turns._extend_execution(runtime, state)
        while state.used_steps < state.step_limit:
            started = agent.turns._start_execution_step(runtime, state, remaining_steps)
            turn = agent.preparation._run_chat_turn(
                runtime.barebone_model,
                runtime.messages,
                runtime.tool_executors,
                current_stage_index=runtime.stage_index if isinstance(runtime, StageRuntimeContext) else None,
                total_stages=len(agent.Stages) if staged else 0,
                context_label="execute_stage" if staged else "run_simple",
            )
            state.used_steps += 1
            state.last_content = turn.last_content
            if not staged and agent.turns._simple_turn_has_no_progress(turn, state.used_steps - 1):
                break
            terminal = agent.turns._terminal_turn_result(runtime, state, turn, remaining_steps, started)
            if terminal is not None:
                return terminal
            agent.turns._record_execution_progress(runtime, state, turn, started)
            if agent.turns._extend_execution(runtime, state):
                break
        if isinstance(runtime, StageRuntimeContext) and agent.logger:
            agent.logger.log_stage_end(runtime.stage.name, state.used_steps)
        next_stage = runtime.stage_index + 1 if isinstance(runtime, StageRuntimeContext) else 0
        return StageExecutionResult(next_stage, state.last_content, False, None, state.used_steps)

    def run_simple(self) -> tuple[str, str]:
        agent = self.agent
        runtime = agent.preparation._prepare_simple_runtime()
        result = agent.turns._run_execution_loop(runtime, agent.maxsteps)
        if result.agent_end_called:
            return (result.last_content, result.last_content)
        return agent.policy_runtime._finalize_simple_after_max_steps(
            runtime.system_prompt, runtime.current_hierarchy, runtime.cli, None, result.last_content
        )

    def execute_stage(
        self, stage_index: int, remaining_steps: int, resume_input: Optional[str] = None
    ) -> StageExecutionResult:
        agent = self.agent
        runtime = agent.preparation._prepare_stage_runtime(stage_index, remaining_steps, resume_input)
        return agent.turns._run_execution_loop(runtime, remaining_steps)
