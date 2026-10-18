# pyright: strict, reportPrivateUsage=false
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict, history_section
from IkaCore.events import current_output as get_cli_output
from IkaCore.prompts import AGENT_END_INSTRUCTION
from IkaModel.chat_interface.chat_interface import run_summarization as _run_summarization_untyped


def _string_attr(obj: object, attr: str, default: str = "unknown") -> str:
    value = getattr(obj, attr, default)
    return value if isinstance(value, str) else default


"AgentPolicy component; owns one execution responsibility."
ToolExecutor = Callable[..., Any]
RunSummarization = Callable[..., str]
run_summarization = cast(RunSummarization, _run_summarization_untyped)
if TYPE_CHECKING:
    from .agents import IkaBaseAgent


class AgentPolicy:
    def __init__(self, agent: IkaBaseAgent):
        self.agent = agent

    def _maybe_extend_stage(
        self, barebone_model: Any, messages: list[JsonDict], step_limit: int, extension_count: int, last_content: str
    ) -> tuple[int, int, str, bool]:
        agent = self.agent
        message_history = agent.message_history
        if extension_count < agent.config.policy.max_stage_extensions:
            what_remains = (
                run_summarization(barebone_model, message_history, prompt_kind="what_remains", write_to_history=False)
                or "(no summary)"
            )
            redirect = f"CRITICAL: You have used all allocated steps for this stage. Summary of what remains:\n\n{what_remains}\n\nComplete the stage or call stage_end / agent_end. You have {agent.config.policy.extend_stage_steps_by} additional steps."
            messages.append({"role": "user", "content": redirect})
            return (step_limit + agent.config.policy.extend_stage_steps_by, extension_count + 1, last_content, False)
        force_answer = run_summarization(
            barebone_model, message_history, prompt_kind="force_answer", write_to_history=False
        )
        if force_answer and force_answer.strip():
            last_content = force_answer.strip()
        return (step_limit, extension_count, last_content, True)

    def _simple_end_instruction(self) -> str:
        agent = self.agent
        submit_tools = [
            getattr(t, "name", "") for t in agent.tools or [] if getattr(t, "name", "").startswith("submit_")
        ]
        if not submit_tools:
            return AGENT_END_INSTRUCTION
        return f"CRITICAL: This task uses a structured submission tool. You MUST call the correct submit_* tool first ({', '.join(submit_tools)} are available in this task). Only after that submit_* tool returns success may you call agent_end with a short final summary.\n\nDO NOT call agent_end before the structured submit_* tool. That is a premature finish and fails the task."

    def _maybe_extend_simple_run(
        self,
        system_prompt: str,
        messages: list[JsonDict],
        current_hierarchy: list[str],
        current_step: int,
        extension_count: int,
        cli: Any,
    ) -> int:
        agent = self.agent
        if extension_count >= agent.config.policy.max_step_extensions:
            return extension_count
        barebone_summary = agent.get_barebone(
            system_prompt, [], parent_hierarchy=current_hierarchy, suppress_init_output=True
        )
        message_history = agent.message_history
        summary = (
            run_summarization(barebone_summary, message_history, prompt_kind="what_remains", write_to_history=False)
            or "(no summary)"
        )
        redirect = f"CRITICAL: You have used all allocated steps without completing the task. You MUST refer back to your original prompt and complete the original goal. Do NOT repeat the same tool calls. Summary of the conversation so far:\n\n{summary}\n\nComplete the task now: use submit_discovery with your findings if you have not already, then call agent_end with your final answer. You have {agent.config.policy.extend_steps_by} additional steps."
        messages.append({"role": "user", "content": redirect})
        agent.maxsteps += agent.config.policy.extend_steps_by
        cli.agent_response(
            agent.name,
            f"Max steps reached. Injected redirect and extended by {agent.config.policy.extend_steps_by} steps. Complete the original goal.",
            current_hierarchy,
            step=current_step,
            is_final=False,
        )
        return extension_count + 1

    def _finalize_simple_after_max_steps(
        self,
        system_prompt: str,
        current_hierarchy: list[str],
        cli: Any,
        last_agent_end_text: Optional[str],
        last_content: str,
    ) -> tuple[str, str]:
        agent = self.agent
        final_step = cli.get_step(agent.name) or agent.maxsteps
        if last_agent_end_text:
            final_response = last_agent_end_text
        else:
            barebone_final = agent.get_barebone(
                system_prompt, [], parent_hierarchy=current_hierarchy, suppress_init_output=True
            )
            message_history = agent.message_history
            force_answer = run_summarization(
                barebone_final, message_history, prompt_kind="force_answer", write_to_history=False
            )
            final_response = force_answer.strip() if force_answer else last_content
            if not final_response:
                final_response = last_content
        cli.agent_response(
            agent.name,
            f"Reached max steps ({agent.maxsteps}). Returning last content.\n\n{final_response}",
            current_hierarchy,
            step=final_step,
            is_final=True,
        )
        return (final_response, final_response)

    def _run_validated_simple_execution(self) -> JsonDict:
        agent = self.agent
        original_maxsteps = agent.maxsteps
        original_prompt = agent.prompt
        max_retries = agent.config.policy.validation_retries
        retry_count = 0
        while True:
            final_output = agent.execution_runtime._run_simple_final_output()
            failed_checks: list[str] = []
            for check in agent.final_answer_checks:
                if not check(final_output):
                    func_name = _string_attr(check, "__name__")
                    failed_checks.append(func_name)
            if not failed_checks:
                agent.maxsteps = original_maxsteps
                agent.prompt = original_prompt
                return final_output
            failed_check_names = ", ".join(failed_checks)
            if retry_count >= max_retries:
                agent.maxsteps = original_maxsteps
                agent.prompt = original_prompt
                if agent.logger:
                    agent.logger.log_action(
                        f"Final answer validation failed after {max_retries} retries. Returning last output despite failed checks: {failed_check_names}"
                    )
                return final_output
            agent.maxsteps = original_maxsteps + agent.config.policy.validation_extra_steps
            check_descriptions: list[str] = []
            for check in agent.final_answer_checks:
                func_name = _string_attr(check, "__name__")
                if func_name in failed_checks:
                    raw_doc = getattr(check, "__doc__", None)
                    doc = (
                        raw_doc.strip() if isinstance(raw_doc, str) and raw_doc.strip() else "No description available"
                    )
                    check_descriptions.append(f"{func_name}: {doc}")
            feedback_msg = (
                f"Your previous answer failed validation checks: {failed_check_names}.\nFailed check requirements:\n"
                + "\n".join((f"- {desc}" for desc in check_descriptions))
                + f"\nPlease review the requirements and provide an improved answer. You have {agent.maxsteps} steps to complete this task."
            )
            agent.prompt = f"{original_prompt}\n\n[FEEDBACK]: {feedback_msg}"
            message_history = agent.message_history
            history_section(message_history, "first_input")["message"] = agent.prompt
            message_history["messages"] = {}
            retry_count += 1
            cli = get_cli_output()
            cli.agent_response(
                agent.name,
                f"Validation failed for checks: {failed_check_names}. Retrying (attempt {retry_count}/{max_retries})...",
                [agent.name],
                step=0,
            )
