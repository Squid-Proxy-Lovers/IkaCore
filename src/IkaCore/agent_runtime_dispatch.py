# pyright: strict, reportPrivateUsage=false
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, cast

from IkaCore.execution_types import StagedExecutionState, StageExecutionResult
from IkaModel.chat_interface.chat_interface import summarise_message_history as _summarise_message_history_untyped

from .agent_runtime_payloads import history_section

"AgentExecution component; owns one execution responsibility."
JsonDict = dict[str, Any]
SummariseMessageHistory = Callable[..., str]
summarise_message_history: SummariseMessageHistory = cast(SummariseMessageHistory, _summarise_message_history_untyped)
if TYPE_CHECKING:
    from .agents import IkaBaseAgent


class AgentExecution:
    def __init__(self, agent: IkaBaseAgent):
        self.agent = agent

    def _interrupt_output(self, interrupt: JsonDict) -> JsonDict:
        agent = self.agent
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
            "usage": getattr(agent, "_total_usage", {}),
            "cost": getattr(agent, "_total_cost", {}),
            "model_id": getattr(agent, "model_id", ""),
            "validation": {"status": "not_run", "attempts": 0, "failed_checks": []},
            **interrupt,
        }

    def _initial_staged_execution_state(self, last_content: str) -> StagedExecutionState:
        agent = self.agent
        return StagedExecutionState(stage_idx=0, remaining_steps=agent.maxsteps, last_content=last_content)

    def _apply_stage_resume_state(self, state: StagedExecutionState, resume_cp: JsonDict) -> None:
        agent = self.agent
        stage_index = resume_cp.get("stage_index", 0)
        remaining_steps = resume_cp.get("remaining_steps", state.remaining_steps)
        if "loop_state" in resume_cp:
            remaining_steps = resume_cp["loop_state"]["remaining_steps"]
        state.stage_idx = min(stage_index if isinstance(stage_index, int) else 0, len(agent.stages) - 1)
        state.remaining_steps = max(0, remaining_steps if isinstance(remaining_steps, int) else state.remaining_steps)
        raw_message_history = resume_cp.get("message_history")
        if isinstance(raw_message_history, dict):
            agent.message_history = cast(JsonDict, raw_message_history)
        last_content = resume_cp.get("last_content", "")
        state.last_content = last_content if isinstance(last_content, str) else str(last_content)

    def _hitl_replay_interrupt_payload(self, resume_cp: JsonDict, state: StagedExecutionState) -> JsonDict:
        agent = self.agent
        return {
            "status": resume_cp.get("interrupt_status", "awaiting_user_input"),
            "stage_index": state.stage_idx,
            "stage_name": agent.stages[state.stage_idx].name if 0 <= state.stage_idx < len(agent.stages) else "unknown",
            "remaining_steps": state.remaining_steps,
            "interrupt_data": deepcopy(resume_cp.get("interrupt_data", {})),
            "message_history": deepcopy(agent.message_history),
            "last_content": state.last_content,
        }

    def _apply_hitl_resume_state(
        self, resume_input: Optional[str], state: StagedExecutionState, resume_cp: JsonDict
    ) -> Optional[JsonDict]:
        agent = self.agent
        agent.execution_runtime._apply_stage_resume_state(state, resume_cp)
        state.resumed_stage_index = state.stage_idx
        if resume_input is not None and resume_input.strip():
            state.resume_stage_input = resume_input
            return None
        return agent.execution_runtime._hitl_replay_interrupt_payload(resume_cp, state)

    def _apply_staged_resume_state(
        self, resume_input: Optional[str], resume_cp: Optional[JsonDict], state: StagedExecutionState
    ) -> Optional[JsonDict]:
        agent = self.agent
        if not resume_cp:
            return None
        if resume_cp.get("scope") == "stage":
            agent.execution_runtime._apply_stage_resume_state(state, resume_cp)
            return None
        if resume_cp.get("scope") == "hitl":
            return agent.execution_runtime._apply_hitl_resume_state(resume_input, state, resume_cp)
        return None

    def _current_stage_resume_input(self, state: StagedExecutionState) -> Optional[str]:
        if state.resumed_stage_index is not None and state.stage_idx == state.resumed_stage_index:
            return state.resume_stage_input
        return None

    def _apply_staged_result(
        self, state: StagedExecutionState, stage_result: StageExecutionResult, current_resume_input: Optional[str]
    ) -> bool:
        state.stage_idx = stage_result.next_stage_index
        state.last_content = stage_result.last_content
        if current_resume_input is not None:
            state.resume_stage_input = None
        state.remaining_steps = (stage_result.remaining_steps if stage_result.remaining_steps is not None
                                 else max(0, state.remaining_steps - stage_result.used_steps))
        if stage_result.agent_end_called:
            state.agent_end_text = stage_result.end_text
            return False
        return True

    def _finalize_staged_execution(self, state: StagedExecutionState) -> JsonDict:
        agent = self.agent
        final_message = (
            state.agent_end_text
            if state.agent_end_text and state.agent_end_text.strip() not in [".", ""]
            else state.last_content
        )
        current_hierarchy = getattr(agent, "_parent_hierarchy", []) + [agent.name]
        system_message = str(history_section(agent.message_history, "system").get("message", ""))
        barebone_model = agent.get_barebone(
            system_message, [], parent_hierarchy=current_hierarchy, suppress_init_output=True
        )
        return agent.chat._build_final_output(final_message, barebone_model)

    def _run_staged_execution(
        self, resume_input: Optional[str], resume_cp: Optional[JsonDict], last_content: str
    ) -> JsonDict:
        agent = self.agent
        state = agent.execution_runtime._initial_staged_execution_state(last_content)
        interrupt_payload = agent.execution_runtime._apply_staged_resume_state(resume_input, resume_cp, state)
        if interrupt_payload:
            agent._last_interrupt = interrupt_payload
            return agent.execution_runtime._interrupt_output(interrupt_payload)
        while 0 <= state.stage_idx < len(agent.stages):
            current_resume_input = agent.execution_runtime._current_stage_resume_input(state)
            stage_result = agent.execute_stage(
                state.stage_idx, state.remaining_steps, resume_input=current_resume_input
            )
            should_continue = agent.execution_runtime._apply_staged_result(state, stage_result, current_resume_input)
            if agent._last_interrupt:
                return agent.execution_runtime._interrupt_output(agent._last_interrupt)
            if not should_continue or state.remaining_steps <= 0:
                break
        return agent.execution_runtime._finalize_staged_execution(state)

    def _run_next_agent_execution(self) -> JsonDict:
        agent = self.agent
        next_agent = agent.next_agent
        if next_agent is None:
            raise RuntimeError("next_agent execution requested without a next_agent")
        final_message, _ = agent.run_simple()
        if agent.summarize_final:
            current_hierarchy = getattr(agent, "_parent_hierarchy", []) + [agent.name]
            barebone_model = agent.get_barebone(
                agent.system_prompt or agent.description or agent.prompt,
                [],
                parent_hierarchy=current_hierarchy,
                suppress_init_output=True,
            )
            summary = summarise_message_history(barebone_model, agent.message_history) or final_message
        else:
            summary = final_message
        downstream = next_agent.clone_for_run()
        try:
            downstream.inject_workflow_context(summary, label="Previous agent summary:")
            return downstream.execution()
        finally:
            downstream.shutdown()

    def _run_simple_final_output(self) -> JsonDict:
        agent = self.agent
        final_message, _ = agent.run_simple()
        if agent._last_interrupt:
            return agent.execution_runtime._interrupt_output(agent._last_interrupt)
        current_hierarchy = getattr(agent, "_parent_hierarchy", []) + [agent.name]
        barebone_model = agent.get_barebone(
            agent.system_prompt or agent.description or agent.prompt,
            [],
            parent_hierarchy=current_hierarchy,
            suppress_init_output=True,
        )
        return agent.chat._build_final_output(final_message, barebone_model)

    def execution(self, resume_input: Optional[str] = None) -> JsonDict:
        agent = self.agent
        last_content = ""
        agent._last_interrupt = None
        raw_resume_cp = getattr(agent, "_resume_state", None)
        resume_cp = cast(JsonDict, raw_resume_cp) if isinstance(raw_resume_cp, dict) else None
        if resume_cp and resume_cp.get("interrupt_data", {}).get("delegated"):
            from .snapshot_agent_state import resume_delegation

            interrupted = resume_delegation(agent, resume_cp, resume_input)
            if interrupted is not None:
                agent._last_interrupt = interrupted
                return agent.execution_runtime._interrupt_output(interrupted)
            resume_input = None
        if agent.stages:
            def run() -> JsonDict:
                return agent.execution_runtime._run_staged_execution(resume_input, resume_cp, last_content)
            return agent.policy_runtime.run_validated(run) if agent.final_answer_checks else run()
        if agent.next_agent:
            return agent.execution_runtime._run_next_agent_execution()
        if resume_cp and resume_cp.get("scope") == "agent":
            raw_message_history = resume_cp.get("message_history")
            if isinstance(raw_message_history, dict):
                agent.message_history = cast(JsonDict, raw_message_history)
            budget = "maxsteps" if "loop_state" in resume_cp else "remaining_steps"
            maxsteps = resume_cp.get(budget, agent.maxsteps)
            agent.maxsteps = max(0, maxsteps if isinstance(maxsteps, int) else agent.maxsteps)
        if not agent.final_answer_checks:
            return agent.execution_runtime._run_simple_final_output()
        return agent.policy_runtime._run_validated_simple_execution()
