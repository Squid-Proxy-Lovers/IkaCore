# pyright: strict, reportPrivateUsage=false
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, cast

from IkaCore.agent_execution_helpers import build_stage_content_prompt, resolve_stage_model_overrides, stage_step_limit
from IkaCore.agent_runtime_payloads import JsonDict, history_message_text, history_section
from IkaCore.events import current_output as get_cli_output
from IkaCore.execution_types import AgentChatTurn, SimpleRuntimeContext, StageExecutionResult, StageRuntimeContext
from IkaCore.prompts import AGENT_END_INSTRUCTION
from IkaModel.base import AgentEndException
from IkaModel.chat_interface.chat_interface import run_summarization as _run_summarization_untyped

"AgentPreparation component; owns one execution responsibility."
ToolExecutor = Callable[..., Any]
RunSummarization = Callable[..., str]
run_summarization = cast(RunSummarization, _run_summarization_untyped)
if TYPE_CHECKING:
    from .agents import IkaBaseAgent


class AgentPreparation:
    def __init__(self, agent: IkaBaseAgent):
        self.agent = agent

    def _run_chat_turn(
        self,
        barebone_model: Any,
        messages: list[JsonDict],
        tool_executors: dict[str, Any],
        current_stage_index: Optional[int],
        total_stages: int,
        context_label: str,
    ) -> AgentChatTurn:
        agent = self.agent
        agent_end_exception = False
        response: JsonDict
        try:
            response = agent.chat_wrapper(
                barebone_model,
                messages,
                tool_executors=tool_executors,
                logger=agent.logger,
                timeout=agent.step_timeout,
                max_tool_rounds=agent.max_tool_rounds,
                max_tool_calls=agent.max_tool_calls,
                current_stage_index=current_stage_index,
                total_stages=total_stages,
                client=agent.chat._get_chat_client(),
            )
        except AgentEndException as exc:
            if agent.logger:
                agent.logger.log_action(f"Caught AgentEndException in {context_label}: {exc}")
            response = exc.response
            agent_end_exception = True
        raw_message_history = response.get("message_history")
        if isinstance(raw_message_history, dict):
            agent.message_history = cast(JsonDict, raw_message_history)
        raw_tool_call_counts = getattr(barebone_model, "_tool_call_counts", None)
        if isinstance(raw_tool_call_counts, dict):
            agent._tool_call_counts.update(cast(dict[str, int], raw_tool_call_counts))
        return AgentChatTurn(
            response=response,
            agent_end_exception=agent_end_exception,
            last_content=str(response.get("content", "")),
            content_before_tools=str(response.get("content_before_tools", "")),
            tool_calls=cast(list[JsonDict], response.get("tool_calls", []) or []),
            executed_tool_calls=cast(list[JsonDict], response.get("executed_tool_calls", []) or []),
        )

    def _handle_stage_interrupt(
        self, stage: Any, stage_index: int, remaining_steps: int, used_steps: int, turn: AgentChatTurn
    ) -> Optional[StageExecutionResult]:
        agent = self.agent
        if not turn.response.get("interrupted"):
            return None
        raw_interrupt_data = turn.response.get("interrupt_data", {})
        interrupt_data = cast(JsonDict, raw_interrupt_data) if isinstance(raw_interrupt_data, dict) else {}
        remaining_after = max(0, remaining_steps - used_steps)
        if agent.logger:
            agent.logger.log_hitl_prompt(stage.name)
        agent._last_interrupt = {
            "status": turn.response.get("status", "awaiting_user_input"),
            "stage_index": stage_index,
            "stage_name": stage.name,
            "remaining_steps": remaining_after,
            "interrupt_data": deepcopy(interrupt_data),
            "message_history": deepcopy(agent.message_history),
            "last_content": turn.last_content,
        }
        return StageExecutionResult(stage_index, turn.last_content, False, None, used_steps)

    def _stage_system_prompt(self, stage: Any) -> str:
        agent = self.agent
        base_system = agent.final_prompt(stage)
        return agent.system_prompt + "\n\n" + base_system if agent.system_prompt else base_system

    def _stage_current_hierarchy(self, stage: Any, stage_index: int) -> list[str]:
        agent = self.agent
        return getattr(agent, "_parent_hierarchy", []) + [agent.name, f"Stage {stage_index}: {stage.name}"]

    def _build_stage_barebone_model(
        self,
        stage: Any,
        stage_index: int,
        system_prompt: str,
        agent_tools: list[Any],
        current_hierarchy: list[str],
        content_prompt: str,
    ) -> Any:
        agent = self.agent
        model_overrides = resolve_stage_model_overrides(agent, stage)
        barebone_model = agent.get_barebone(
            system_prompt,
            agent_tools,
            parent_hierarchy=current_hierarchy,
            suppress_init_output=stage_index != 0,
            model_overrides=model_overrides if model_overrides else None,
            content_prompt_override=content_prompt,
        )
        barebone_model._tool_call_counts = {}
        if getattr(agent, "context_budget", None):
            barebone_model.context_budget = agent.context_budget
        return barebone_model

    def _stage_initial_messages(
        self, stage: Any, resume_input: Optional[str], message_history: JsonDict, content_prompt: str
    ) -> list[JsonDict]:
        agent = self.agent
        from .snapshot_agent_state import stage_initial_messages

        return stage_initial_messages(agent, stage, resume_input, message_history, content_prompt)

    def _stage_tool_executors(
        self, stage: Any, stage_index: int, remaining_steps: int, current_hierarchy: list[str]
    ) -> dict[str, Any]:
        agent = self.agent
        agent.chat._reset_tool_call_counts()
        return agent.build_tool_executors(
            stage.tools,
            subagents=getattr(stage, "subagents", None),
            parent_hierarchy=current_hierarchy,
            stage=stage,
            stage_index=stage_index,
            remaining_steps=remaining_steps,
        )

    def _prepare_stage_runtime(
        self, stage_index: int, remaining_steps: int, resume_input: Optional[str]
    ) -> StageRuntimeContext:
        agent = self.agent
        message_history = agent.message_history
        stage = agent.Stages[stage_index]
        agent.state.turn_message = None
        system_prompt = agent.preparation._stage_system_prompt(stage)
        history_section(message_history, "system")["message"] = system_prompt
        agent_tools = agent.build_stage(stage)
        current_hierarchy = agent.preparation._stage_current_hierarchy(stage, stage_index)
        content_prompt = build_stage_content_prompt(agent.prompt, agent.Stages, stage, stage_index)
        from .workflow_prompt_context import stage_context

        content_prompt = stage_context(agent, content_prompt)
        first_input = history_section(message_history, "first_input")
        first_input["message"] = content_prompt
        first_input["tokens"] = 0
        barebone_model = agent.preparation._build_stage_barebone_model(
            stage, stage_index, system_prompt, agent_tools, current_hierarchy, content_prompt
        )
        messages = agent.preparation._stage_initial_messages(stage, resume_input, message_history, content_prompt)
        from .snapshot_agent_state import stage_boundary

        stage_boundary(stage_index, remaining_steps, stage.name, messages)
        tool_executors = agent.preparation._stage_tool_executors(stage, stage_index, remaining_steps, current_hierarchy)
        step_limit = stage_step_limit(stage, remaining_steps)
        if agent.logger:
            agent.logger.log_stage_start(stage.name, getattr(stage, "hitl", False), remaining_steps, step_limit)
        return StageRuntimeContext(
            stage=stage,
            system_prompt=system_prompt,
            content_prompt=content_prompt,
            current_hierarchy=current_hierarchy,
            barebone_model=barebone_model,
            messages=messages,
            tool_executors=tool_executors,
            step_limit=step_limit,
            stage_index=stage_index,
        )

    def _prepare_simple_runtime(self) -> SimpleRuntimeContext:
        agent = self.agent
        correcting = agent.state.validation_attempts > 0
        dynamic_tools = ([agent.tool_runtime._simple_agent_end_tool([])]
                         if correcting else agent.build_simple_tools())
        system_prompt = agent.system_prompt or ""
        current_hierarchy = getattr(agent, "_parent_hierarchy", []) + [agent.name]
        cli = get_cli_output()
        cli.set_step(agent.name, 1)
        message_history = agent.message_history
        first_msg = history_message_text(message_history, "first_input")
        turn_message = agent.state.turn_message
        instruction = AGENT_END_INSTRUCTION if correcting else agent.policy_runtime._simple_end_instruction()
        base_prompt = turn_message or first_msg or agent.prompt or ""
        start_prompt = (base_prompt if instruction.strip() in base_prompt
                        else base_prompt + "\n\n" + instruction)
        initial = not history_section(message_history, "messages")
        if not first_msg or initial:
            history_section(message_history, "first_input")["message"] = start_prompt
        if turn_message or not first_msg or initial:
            from uuid import uuid4
            history_section(message_history, "messages")[str(uuid4())] = {
                "type": "user_input", "message": start_prompt, "tokens": max(1, len(start_prompt) // 4)}
        agent.state.turn_message = None
        barebone_model = agent.get_barebone(
            system_prompt, dynamic_tools, parent_hierarchy=current_hierarchy, content_prompt_override=start_prompt
        )
        barebone_model._tool_call_counts = {}
        if getattr(agent, "context_budget", None):
            barebone_model.context_budget = agent.context_budget
        messages: list[JsonDict] = [{"role": "user", "content": start_prompt}]
        from .snapshot_agent_state import restore_messages

        messages = restore_messages(agent, messages)
        tool_executors = agent.build_tool_executors(
            [] if correcting else agent.tools, subagents=[] if correcting else agent.subagents,
            parent_hierarchy=current_hierarchy
        )
        return SimpleRuntimeContext(
            system_prompt=system_prompt,
            current_hierarchy=current_hierarchy,
            barebone_model=barebone_model,
            messages=messages,
            tool_executors=tool_executors,
            cli=cli,
        )
