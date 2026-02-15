import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from IkaCore.tools import IkaTools
from IkaCore.stages import IkaStage
from IkaCore.logging_utils import IkaLogger
from IkaCore.checkpoint import CheckpointStore
from IkaCore.cli_output import get_cli_output
from IkaCore.prompts import *

src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_dir / "IkaMem")) 
from IkaMem import STMemory, LTMemory  # type: ignore

from IkaModel.base import *


from IkaModel.chat_interface import run_summarization, summarise_message_history

from IkaCore.agent_memory import AgentMemoryMixin
from IkaCore.agent_tools import AgentToolsMixin
from IkaCore.agent_execution import AgentExecutionMixin
from IkaCore.agent_helpers import AgentHelpersMixin


class IkaBaseAgent(AgentMemoryMixin, AgentToolsMixin, AgentExecutionMixin, AgentHelpersMixin):
    def __init__(
        self,
        name: str,
        description: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        start_prompt: Optional[str] = None,
        end_prompt: Optional[str] = None,
        tools: Optional[List[IkaTools]] = None,
        model_id: str = "",
        api_key: str = "",
        api_url: Optional[str] = None,
        max_tokens: int = 50000,
        temperature: float = 0.0,
        checkpoint: bool = False,
        Batch: bool = False,
        BatchMax: int = 3,
        Stages: Optional[List[IkaStage]] = None,
        subagents: Optional[List["IkaBaseAgent"]] = None,
        next_agent: Optional["IkaBaseAgent"] = None,
        feedback_agent: Optional["IkaBaseAgent"] = None,
        maxsteps: int = 100,
        step_timeout: int = 900,
        rate_limit_per_min: Optional[float] = None,
        per_tool_rate_limit: Optional[Dict[str, float]] = None,
        memory: bool = False,
        memory_access: Optional[Dict[str, bool]] = None,
        final_answer_check: Optional[List[Callable]] = None,
        logging_level: int = 0,
        logging_file: str = "logs.txt",
        show_usage_level0: bool = True,
        checkpoint_db_path: str = "checkpoints.db",
        summarize_final: bool = False,
        use_async: bool = False,
        max_tool_rounds: Optional[int] = None,
        max_step_extensions: int = 2,
        extend_steps_by: int = 3,
        max_stage_extensions: int = 2,
        extend_stage_steps_by: int = 3,
        reasoning_effort: Optional[str] = None,
    ):
        tools = tools or []
        Stages = Stages or []
        subagents = subagents or []

        self._validate_required_fields(
            {
                "name": name,
                "description": description,
                "prompt": prompt,
                "model_id": model_id,
                "api_key": api_key,
            }
        )

        self.name = name
        self.description = description
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.start_prompt = start_prompt
        self.end_prompt = end_prompt
        self.tools = tools
        self.model_id = model_id
        self.api_key = api_key
        self.api_url = api_url if api_url else self.geturl(model_id)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.checkpoint = checkpoint
        self.Batch = Batch
        self.BatchMax = BatchMax
        self.Stages = Stages
        self.subagents = subagents
        self.next_agent = next_agent
        self.feedback_agent = feedback_agent
        if self.Stages:
            if self.subagents or self.next_agent or self.feedback_agent:
                raise ValueError("Subagents/next/feedback agents are not allowed when stages are defined.")
        else:
            if self.subagents and self.next_agent:
                raise ValueError("Only one of subagents or next_agent may be set when no stages are provided.")
        self.maxsteps = maxsteps
        self.step_timeout = step_timeout
        self.memory = memory
        self.memory_access = self._build_memory_access_defaults(memory, memory_access)
        self.message_history = self._initial_message_history(self.system_prompt)
        self.logging_level = logging_level
        self.logging_file = logging_file
        self.logger = IkaLogger(logging_level, logging_file, show_usage_level0=show_usage_level0)
        
        if logging_level == 3:
            import logging
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logging.getLogger("IkaModel.chat_interface").setLevel(logging.DEBUG)
        self.rate_limit_per_min = rate_limit_per_min
        self.per_tool_rate_limit = per_tool_rate_limit or {}
        self._last_api_call_ts: float = 0.0
        self.checkpoint_store = CheckpointStore(checkpoint_db_path) if checkpoint else None
        self._resume_checkpoint: Optional[dict] = None
        self.short_term_memory: Optional[STMemory] = None
        self.long_term_memory: Optional[LTMemory] = get_global_long_term_memory()
        self.summarize_final = summarize_final
        self.use_async = use_async
        self.max_tool_rounds = max_tool_rounds if max_tool_rounds is not None else 5
        self.final_answer_checks = self._validate_final_answer_checks(final_answer_check)
        self._tool_call_counts: Dict[str, int] = {}
        self._total_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "input_cached_tokens": 0}
        self._total_cost: Dict[str, float] = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
        self.client = None
        self.max_step_extensions = max_step_extensions
        self.extend_steps_by = extend_steps_by
        self.max_stage_extensions = max_stage_extensions
        self.extend_stage_steps_by = extend_stage_steps_by

    def shutdown(self) -> None:
        if self.logger:
            self.logger.shutdown()

    def execute_stage(self, stage_index: int, remaining_steps: int) -> tuple[int, str, bool, Optional[str], int]:

        stage = self.Stages[stage_index]
        base_system = self.final_prompt(stage)
        system_prompt = (self.system_prompt + "\n\n" + base_system) if self.system_prompt else base_system
        self.message_history["system"]["message"] = system_prompt

        agent_tools = self.build_stage(stage)
        current_hierarchy = getattr(self, "_parent_hierarchy", []) + [self.name, f"Stage {stage_index}: {stage.name}"]

        parts = []
        if self.prompt:
            parts.append(self.prompt)
        if self.Stages:
            stages_block = "\n\n".join(
                f"Stage {i} ({s.name}):\n{s.prompt or ''}" for i, s in enumerate(self.Stages)
            )
            parts.append("STAGES:\n\n" + stages_block)

        parts.append("CURRENT STAGE IS:\n\n" + (stage.prompt or ""))
        parts.append(STAGE_MOVEMENT_INSTRUCTION)  # noqa: F405
        
        if getattr(stage, "hitl", False):
            parts.append(HITL_INSTRUCTION)  # noqa: F405
            
        if stage_index == len(self.Stages) - 1:
            parts.append(AGENT_END_INSTRUCTION)  # noqa: F405
        content_prompt = "\n\n".join(parts)



        # we use parent values if not set on the stage
        model_overrides: Dict[str, Any] = {}
        if getattr(stage, "model_id", None) is not None:
            model_overrides["model_id"] = stage.model_id
        if getattr(stage, "api_key", None) is not None:
            model_overrides["api_key"] = stage.api_key
        if getattr(stage, "api_url", None) is not None:
            model_overrides["api_url"] = stage.api_url
        elif model_overrides.get("model_id") is not None:
            model_overrides["api_url"] = self.geturl(model_overrides["model_id"])
        if getattr(stage, "max_tokens", None) is not None:
            model_overrides["max_tokens"] = stage.max_tokens
        if getattr(stage, "temperature", None) is not None:
            model_overrides["temperature"] = stage.temperature

        suppress_init_output = (stage_index != 0)
        barebone_model = self.get_barebone(system_prompt, agent_tools, parent_hierarchy=current_hierarchy, suppress_init_output=suppress_init_output, model_overrides=model_overrides if model_overrides else None, content_prompt_override=content_prompt)
        # Give each stage its own fresh tool call counter (don't share across stages)
        barebone_model._tool_call_counts = {}
        # Propagate context_budget so chat() can trigger summarization earlier
        if getattr(self, 'context_budget', None):
            barebone_model.context_budget = self.context_budget

        messages: List[dict] = [{"role": "user", "content": content_prompt}]
        last_content = ""

        used_steps = 0
        stage_max = getattr(stage, "stage_max_step", 1)
        if stage_max == 0:
            step_limit = max(1, remaining_steps)
        else:
            step_limit = min(stage_max, max(1, remaining_steps))

        self._reset_tool_call_counts()

        stage_memory_access = getattr(stage, "memory_access", None) or self.memory_access

        tool_executors = self.build_tool_executors(
            stage.tools,
            memory_access=stage_memory_access,
            long_term_filter=getattr(stage, "long_term_filter", None),
            subagents=getattr(stage, "subagents", None),
            parent_hierarchy=current_hierarchy,
            stage=stage,
        )

        if self.logger:
            self.logger.log_stage_start(stage.name, getattr(stage, "hitl", False), remaining_steps, step_limit)

        extension_count = 0
        while used_steps < step_limit:
            if self.logger:
                self.logger.log_action(f"stage_start:{stage.name}")
            self._enforce_rate_limit_model()
            step_start = time.time()

            agent_end_exception = False
            try:
                response = self.chat_wrapper(
                    barebone_model,
                    messages,
                    tool_executors=tool_executors,
                    logger=self.logger,
                    timeout=self.step_timeout,
                    max_tool_rounds=self.max_tool_rounds,
                    max_tool_calls=self.maxsteps,
                    current_stage_index=stage_index,
                    total_stages=len(self.Stages) if self.Stages else 0,
                    client=self.client,
                )
            except AgentEndException as exc:
                if self.logger:
                    self.logger.log_action(f"Caught AgentEndException in execute_stage: {exc}")
                response = exc.response or {}
                agent_end_exception = True

            self.message_history = response.get("message_history", self.message_history)
            last_content = response.get("content", "")
            content_before_tools = response.get("content_before_tools", "")
            tool_calls = response.get("tool_calls", []) or []
            executed_tool_calls = response.get("executed_tool_calls", []) or []
            used_steps += 1
            if hasattr(barebone_model, '_tool_call_counts'):
                self._tool_call_counts.update(barebone_model._tool_call_counts)

            try:
                target_stage, agent_end_called, agent_end_text = self.parse_control_calls(
                    executed_tool_calls if executed_tool_calls else tool_calls,
                    stage,
                    stage_index,
                    response_content=content_before_tools,
                )
            except ValueError as e:
                error_msg = str(e)
                if self.logger:
                    self.logger.log_action(f"ERROR in agent_end: {error_msg}")
                raise

            if agent_end_exception:
                agent_end_called = True
            if agent_end_called:
                try:
                    final_content = self._fallback_final_content(
                        agent_end_text,
                        content_before_tools,
                        last_content,
                    )
                except ValueError as e:
                    if self.logger:
                        self.logger.log_action(f"ERROR: {str(e)}")
                    raise
                return stage_index, final_content, True, final_content, used_steps
            if target_stage == "next":
                return stage_index + 1, last_content, False, None, used_steps
            if isinstance(target_stage, int):
                return target_stage, last_content, False, None, used_steps

            messages.append({"role": "assistant", "content": last_content})

            if self.logger:
                self.logger.log_step(
                    stage_name=stage.name,
                    step_idx=used_steps,
                    output=last_content,
                    tool_calls=tool_calls,
                    usage=response.get("usage", {}),
                    cost=response.get("cost", {}),
                    elapsed=time.time() - step_start,
                )
            remaining_after = max(0, remaining_steps - used_steps)
            if self.checkpoint:
                self._save_stage_checkpoint(stage_index, remaining_after, last_content)

            if used_steps >= step_limit:
                if extension_count < self.max_stage_extensions:
                    what_remains = run_summarization(
                        barebone_model,
                        self.message_history,
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
                    step_limit += self.extend_stage_steps_by
                    extension_count += 1
                else:
                    force_answer = run_summarization(
                        barebone_model,
                        self.message_history,
                        prompt_kind="force_answer",
                        write_to_history=False,
                    )
                    if force_answer and force_answer.strip():
                        last_content = force_answer.strip()
                    break

        if self.logger:
            self.logger.log_stage_end(stage.name, used_steps)
        return stage_index + 1, last_content, False, None, used_steps

    def run_simple(self) -> tuple[str, str]:
        dynamic_tools = self.build_simple_tools()
        system_prompt = self.system_prompt
        current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
        cli = get_cli_output()
        cli.set_step(self.name, 1)

        first_msg = (self.message_history.get("first_input") or {}).get("message") or ""
        start_prompt = (first_msg or self.prompt or "") + "\n\n" + AGENT_END_INSTRUCTION
        barebone_model = self.get_barebone(
            system_prompt,
            dynamic_tools,
            parent_hierarchy=current_hierarchy,
            content_prompt_override=start_prompt,
        )
        # Give each agent its own fresh tool call counter (don't share with parent)
        barebone_model._tool_call_counts = {}
        # Propagate context_budget so chat() can trigger summarization earlier
        if getattr(self, 'context_budget', None):
            barebone_model.context_budget = self.context_budget
        messages: List[dict] = [{"role": "user", "content": start_prompt}]
        last_content = ""
        last_agent_end_text = None
        current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
        tool_executors = self.build_tool_executors(self.tools, memory_access=self.memory_access, subagents=self.subagents, parent_hierarchy=current_hierarchy)
        step_num = 0
        extension_count = 0

        while step_num < self.maxsteps:
            current_step = step_num + 1
            cli.set_step(self.name, current_step)
            if step_num == 0:
                cli.agent_init(
                    self.name,
                    current_hierarchy,
                    step=current_step,
                    description=f"Step {current_step}/{self.maxsteps}"
                )
            self._enforce_rate_limit_model()
            step_start = time.time()

            agent_end_exception = False
            try:
                response = self.chat_wrapper(
                    barebone_model,
                    messages,
                    tool_executors=tool_executors,
                    logger=self.logger,
                    timeout=self.step_timeout,
                    max_tool_rounds=self.max_tool_rounds,
                    max_tool_calls=self.maxsteps,
                    current_stage_index=None,
                    total_stages=0,
                    client=self.client,
                )
            except AgentEndException as exc:
                if self.logger:
                    self.logger.log_action(f"Caught AgentEndException in run_simple: {exc}")
                response = exc.response or {}
                agent_end_exception = True

            self.message_history = response.get("message_history", self.message_history)
            last_content = response.get("content", "")
            content_before_tools = response.get("content_before_tools", "")
            tool_calls = response.get("tool_calls", []) or []
            executed_tool_calls = response.get("executed_tool_calls", []) or []

            if response.get("hijacked"):
                pass

            if hasattr(barebone_model, '_tool_call_counts'):
                self._tool_call_counts.update(barebone_model._tool_call_counts)

            if self.logger:
                self.logger.log_action(f"DEBUG run_simple: step={step_num} tools={len(tool_calls)} exec={len(executed_tool_calls)} content='{last_content}'")
                if executed_tool_calls:
                     names = [tc.get('function', {}).get('name') or tc.get('name') for tc in executed_tool_calls]
                     self.logger.log_action(f"DEBUG run_simple: executed tool names: {names}")

            # Safety break for infinite loop
            if not tool_calls and not last_content and not executed_tool_calls and step_num > 0:
                 if self.logger:
                      self.logger.log_action("CRITICAL: Infinite loop detected (no tools, no content). Breaking.")
                 break

            try:
                _, agent_end_called, agent_end_text = self.parse_control_calls(
                    executed_tool_calls if executed_tool_calls else tool_calls,
                    None,
                    response_content=content_before_tools,
                )
                
                if self.logger and agent_end_called:
                    self.logger.log_action(f"DEBUG run_simple: agent_end detected with text: {agent_end_text}")

                if agent_end_called and agent_end_text:
                    last_agent_end_text = agent_end_text
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

            if agent_end_exception:
                agent_end_called = True
            if agent_end_called:
                try:
                    final_content = self._fallback_final_content(
                        agent_end_text,
                        content_before_tools,
                        last_content,
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
                    is_final=True
                )
                if self.logger:
                    self.logger.log_step(
                        stage_name="simple",
                        step_idx=step_num,
                        output=last_content,
                        tool_calls=tool_calls,
                        usage=response.get("usage", {}),
                        cost=response.get("cost", {}),
                        elapsed=time.time() - step_start,
                    )
                return final_content, final_content

            if not tool_calls and last_content:
                messages.append({"role": "assistant", "content": last_content})
                cli.agent_response(
                    self.name,
                    last_content,
                    current_hierarchy,
                    step=current_step,
                    is_final=False
                )
                if self.logger:
                    self.logger.log_step(
                        stage_name="simple",
                        step_idx=step_num,
                        output=last_content,
                        tool_calls=[],
                        usage=response.get("usage", {}),
                        cost=response.get("cost", {}),
                        elapsed=time.time() - step_start,
                    )
                step_num += 1
                continue

            if self.logger:
                self.logger.log_step(
                    stage_name="simple",
                    step_idx=step_num,
                    output=last_content,
                    tool_calls=tool_calls,
                    usage=response.get("usage", {}),
                    cost=response.get("cost", {}),
                    elapsed=time.time() - step_start,
                )
            max_steps_val = int(self.maxsteps)
            current_step_val = int(step_num) if step_num is not None else 0
            remaining_after = max(0, max_steps_val - (current_step_val + 1))
            self._save_agent_checkpoint(remaining_after, last_content)
            step_num += 1

            if step_num >= self.maxsteps and extension_count < self.max_step_extensions:
                barebone_summary = self.get_barebone(
                    system_prompt,
                    [],
                    parent_hierarchy=current_hierarchy,
                    suppress_init_output=True,
                )
                summary = run_summarization(
                    barebone_summary,
                    self.message_history,
                    prompt_kind="what_remains",
                    write_to_history=False,
                ) or "(no summary)"
                redirect = (
                    "CRITICAL: You have used all allocated steps without completing the task. "
                    "You MUST refer back to your original prompt and complete the original goal. "
                    "Do NOT repeat the same tool calls. Summary of the conversation so far:\n\n"
                    f"{summary}\n\n"
                    "Complete the task now: use submit_discovery with your findings if you have not already, then call agent_end with your final answer. "
                    f"You have {self.extend_steps_by} additional steps."
                )
                messages.append({"role": "user", "content": redirect})
                self.maxsteps += self.extend_steps_by
                extension_count += 1
                cli.agent_response(
                    self.name,
                    f"Max steps reached. Injected redirect and extended by {self.extend_steps_by} steps. Complete the original goal.",
                    current_hierarchy,
                    step=current_step,
                    is_final=False,
                )

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
            force_answer = run_summarization(
                barebone_final,
                self.message_history,
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
            is_final=True
        )
        return final_response, final_response

    def execution(self, checkpoint_uid: Optional[str] = None) -> Dict[str, str]:
        last_content = ""

        if checkpoint_uid:
            self.load_checkpoint(checkpoint_uid)

        resume_cp = getattr(self, "_resume_checkpoint", None)

        if self.Stages:
            stage_idx = 0
            agent_end_text = None
            remaining_steps = self.maxsteps
            if resume_cp and resume_cp.get("scope") == "stage":
                stage_idx = min(resume_cp.get("stage_index", 0), len(self.Stages) - 1)
                remaining_steps = max(1, resume_cp.get("remaining_steps", remaining_steps))
                self.message_history = resume_cp.get("message_history", self.message_history)
                last_content = resume_cp.get("last_content", "")
            while 0 <= stage_idx < len(self.Stages):
                stage_idx, last_content, agent_end_called, end_text, used = self.execute_stage(stage_idx, remaining_steps)
                remaining_steps -= used
                if agent_end_called:
                    agent_end_text = end_text
                    break
            final_message = agent_end_text if agent_end_text and agent_end_text.strip() not in [".", ""] else last_content
            current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
            barebone_model = self.get_barebone(self.message_history["system"]["message"], [], parent_hierarchy=current_hierarchy, suppress_init_output=True)
            return self._build_final_output(final_message, barebone_model)

        if self.next_agent:
            final_message, _ = self.run_simple()
            if self.summarize_final:
                current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
                barebone_model = self.get_barebone(self.system_prompt or self.description or self.prompt, [], parent_hierarchy=current_hierarchy, suppress_init_output=True)
                summary = summarise_message_history(barebone_model, self.message_history) or final_message
            else:
                summary = final_message
            self.next_agent.message_history = {
                "system": {"message": self.next_agent.system_prompt or "", "tokens": 0},
                "first_input": {"message": f"Previous agent summary:\n{summary}", "tokens": 0},
                "summary": {"message": "", "tokens": 0},
                "messages": {},
            }
            return self.next_agent.execution()

        if resume_cp and resume_cp.get("scope") == "agent":
            self.message_history = resume_cp.get("message_history", self.message_history)
            self.maxsteps = max(1, resume_cp.get("remaining_steps", self.maxsteps))

        if not self.final_answer_checks:
            final_message, _ = self.run_simple()
            current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
            barebone_model = self.get_barebone(self.system_prompt or self.description or self.prompt, [], parent_hierarchy=current_hierarchy, suppress_init_output=True)
            return self._build_final_output(final_message, barebone_model)

        original_maxsteps = self.maxsteps
        original_prompt = self.prompt
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            final_message, _ = self.run_simple()
            current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
            barebone_model = self.get_barebone(self.system_prompt or self.description or self.prompt, [], parent_hierarchy=current_hierarchy, suppress_init_output=True)
            final_output = self._build_final_output(final_message, barebone_model)

            failed_checks = []
            for check in self.final_answer_checks:
                if not check(final_output):
                    func_name = getattr(check, '__name__', 'unknown')
                    failed_checks.append(func_name)

            if not failed_checks:
                self.maxsteps = original_maxsteps
                self.prompt = original_prompt
                return final_output

            if retry_count < max_retries:
                self.maxsteps = original_maxsteps + 10
                failed_check_names = ", ".join(failed_checks)
                check_descriptions = []
                for check in self.final_answer_checks:
                    func_name = getattr(check, '__name__', 'unknown')
                    if func_name in failed_checks:
                        doc = getattr(check, '__doc__', 'No description available').strip()
                        check_descriptions.append(f"{func_name}: {doc}")
                
                feedback_msg = (
                    f"Your previous answer failed validation checks: {failed_check_names}.\n"
                    f"Failed check requirements:\n" + "\n".join(f"- {desc}" for desc in check_descriptions) + "\n"
                    f"Please review the requirements and provide an improved answer. "
                    f"You have {self.maxsteps} steps to complete this task."
                )
                self.prompt = f"{original_prompt}\n\n[FEEDBACK]: {feedback_msg}"
                self.message_history["first_input"]["message"] = self.prompt
                self.message_history["messages"] = {}
                retry_count += 1
                cli = get_cli_output()
                cli.agent_response(
                    self.name,
                    f"Validation failed for checks: {failed_check_names}. Retrying (attempt {retry_count}/{max_retries})...",
                    [self.name],
                    step=0
                )
            else:
                self.maxsteps = original_maxsteps
                self.prompt = original_prompt
                failed_check_names = ", ".join(failed_checks)
                if self.logger:
                    self.logger.log_action(f"Final answer validation failed after {max_retries} retries. Returning last output despite failed checks: {failed_check_names}")
                return final_output

        self.maxsteps = original_maxsteps
        self.prompt = original_prompt
        return final_output
