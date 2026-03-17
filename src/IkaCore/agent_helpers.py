from __future__ import annotations

import builtins
import json
import re
import time
import typing
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, get_type_hints
import asyncio

from IkaCore.tools import IkaTools
from IkaCore.stages import IkaStage
from IkaCore.logging_utils import IkaLogger
from IkaCore.checkpoint import CheckpointStore
from IkaCore.cli_output import get_cli_output, OutputType
from IkaCore.prompts import *

from IkaModel.base import BareBoneModel, AgentTool, ToolArgs, AgentEndException
from IkaModel.chat_interface.chat_interface import chat, async_chat, summarise_message_history

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent


class AgentHelpersMixin:
    
    name: str
    description: str
    prompt: str
    system_prompt: Optional[str]
    start_prompt: Optional[str]
    end_prompt: Optional[str]
    tools: List[IkaTools]
    model_id: str
    api_key: str
    api_url: Optional[str]
    max_tokens: int
    temperature: float
    checkpoint: bool
    Batch: bool
    BatchMax: int
    Stages: List[IkaStage]
    subagents: List["IkaBaseAgent"]
    next_agent: Optional["IkaBaseAgent"]
    feedback_agent: Optional["IkaBaseAgent"]
    maxsteps: int
    step_timeout: int
    memory: bool
    memory_access: Dict[str, bool]
    message_history: Dict[str, Dict[str, object]]
    logging_level: int
    logging_file: str
    logger: Optional[IkaLogger]
    rate_limit_per_min: Optional[float]
    per_tool_rate_limit: Dict[str, float]
    _last_api_call_ts: float
    checkpoint_store: Optional[CheckpointStore]
    _resume_checkpoint: Optional[dict]
    short_term_memory: Optional[Any]
    long_term_memory: Optional[Any]
    summarize_final: bool
    use_async: bool
    max_tool_rounds: int
    final_answer_checks: List[Callable]
    _tool_call_counts: Dict[str, int]
    client: Optional[Any]

    @staticmethod
    def _validate_required_fields(fields: Dict[str, object]) -> None:
        for field_name, value in fields.items():
            if not value:
                raise ValueError(f"{field_name} is required for the agent")

    @staticmethod
    def _build_memory_access_defaults(
        memory_enabled: bool, overrides: Optional[Dict[str, bool]] = None
    ) -> Dict[str, bool]:
        defaults = {
            "short_term_save": True,
            "short_term_search": True,
            "long_term_save": True,
            "long_term_search": True,
        }
        if not memory_enabled:
            defaults = {key: False for key in defaults}

        if overrides:
            defaults.update(overrides)
        return defaults

    @staticmethod
    def _initial_message_history(system_prompt: Optional[str]) -> Dict[str, Dict[str, object]]:
        return {
            "system": {"message": system_prompt or "", "tokens": 0},
            "first_input": {"message": "", "tokens": 0},
            "summary": {"message": "", "tokens": 0},
            "messages": {},
        }

    @staticmethod
    def geturl(model_id: str) -> str:
        """
        Get the API URL for a model.

        IMPORTANT: Check for OpenRouter format (/) BEFORE checking provider names
        to handle cases like "google/gemini-3-flash-preview" on OpenRouter.
        """
        model_id_lower = model_id.lower()

        # PRIORITY 1: Check for OpenRouter format (has slash) FIRST
        # This catches "google/gemini-pro", "anthropic/claude-sonnet", etc.
        if "/" in model_id_lower:
            return "https://openrouter.ai/api/v1/chat/completions"

        # PRIORITY 2: Then check for provider-specific patterns
        if "deepseek" in model_id_lower:
            return "https://api.deepseek.com/chat/completions"
        if "gpt" in model_id_lower or "o1" in model_id_lower or "o3" in model_id_lower:
            return "https://api.openai.com/v1/chat/completions"
        if "claude" in model_id_lower:
            return "https://api.anthropic.com/v1/messages"
        if "gemini" in model_id_lower:
            # Only reached if no slash (direct Google API)
            return f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"

        # Default to OpenAI-compatible
        return "https://api.openai.com/v1/chat/completions"

    def _validate_final_answer_checks(
        self,
        final_answer_check: Optional[List[Callable]],
    ) -> List[Callable]:
        
        if final_answer_check is None:
            return []

        for check in final_answer_check:
            if not callable(check):
                raise ValueError("final_answer_check must be a list of callable functions")

            func_name = getattr(check, "__name__", "unknown")
            docstring = (check.__doc__ or "").strip()
            if not docstring:
                raise ValueError(
                    f"final_answer_check function '{func_name}' must have a docstring explaining what it does"
                )

            try:
                hints = get_type_hints(check)
                return_type = hints.get("return")
            except (TypeError, AttributeError):
                return_type = None

            if return_type is not None:
                get_origin = getattr(typing, "get_origin", None)
                get_args = getattr(typing, "get_args", None)
                origin = get_origin(return_type) if get_origin else None
                args = get_args(return_type) if (origin and get_args) else ()
                is_bool = return_type is bool or (origin is not None and bool in args)
                if not is_bool:
                    raise ValueError(
                        f"final_answer_check function '{func_name}' must have return type annotation of bool, got {return_type}"
                    )

            result = check(self.message_history)
            if not isinstance(result, bool):
                raise ValueError(
                    f"final_answer_check function '{func_name}' must return a boolean, got {type(result).__name__}"
                )

        return final_answer_check

    def _reset_tool_call_counts(self):
        self._tool_call_counts = {}

    def _enforce_rate_limit(self, per_minute: Optional[float], last_ts_attr: str) -> None:
        if not per_minute or per_minute <= 0:
            return
        interval = 60.0 / per_minute
        now = time.time()
        last_ts = getattr(self, last_ts_attr, 0.0)
        elapsed = now - last_ts
        if elapsed < interval:
            time.sleep(interval - elapsed)
        setattr(self, last_ts_attr, time.time())

    def _enforce_rate_limit_model(self) -> None:
        self._enforce_rate_limit(self.rate_limit_per_min, "_last_api_call_ts")

    def _enforce_rate_limit_tool(self, tool_name: str) -> None:
        per_tool_limit = None
        if hasattr(self, "per_tool_rate_limit") and isinstance(self.per_tool_rate_limit, dict):
            per_tool_limit = self.per_tool_rate_limit.get(tool_name)
        effective = per_tool_limit if per_tool_limit is not None else self.rate_limit_per_min
        last_attr = f"_last_tool_ts_{tool_name}"
        self._enforce_rate_limit(effective, last_attr)

    def _prompt_hitl_question(self, stage_name: str, question: str) -> str:
        if self.logger:
            self.logger.log_hitl_question(stage_name, question)
        try:
            prompt = f"[HITL] {question}\nYour answer: "
            if self.logger and self.logger.use_colors and self.logger.level == 0:
                prompt = self.logger._color(prompt, "orange")
            user_text = builtins.input(prompt).strip()
            if self.logger and user_text:
                self.logger.log_hitl_answer(stage_name, user_text)
            return user_text or ""
        except EOFError:
            return ""

    def inject_workflow_context(self, context: str) -> None:
        if not context:
            return
        current_first_input = self.message_history.get("first_input", {}).get("message", "")
        if current_first_input:
            combined = f"{context}\n\n{current_first_input}"
        else:
            combined = context
        self.message_history["first_input"]["message"] = combined

    def apply_workflow_stage_wiring(self, stage_wiring: Dict[int, Dict[str, List["IkaBaseAgent"]]]) -> None:
        if not stage_wiring or not self.Stages:
            return
        for stage_idx, wiring in stage_wiring.items():
            if 0 <= stage_idx < len(self.Stages):
                stage = self.Stages[stage_idx]
                if "subagents" in wiring:
                    stage.subagents = wiring["subagents"]

    def _save_stage_checkpoint(self, stage_index: int, remaining_steps: int, last_content: str) -> Optional[str]:
        if not self.checkpoint_store:
            return None
        
        if stage_index < len(self.Stages):
            stage = self.Stages[stage_index]
            stage_checkpoint_enabled = getattr(stage, "checkpoint", False)
            if not stage_checkpoint_enabled:
                return None
        
        payload = {
            "scope": "stage",
            "agent_name": self.name,
            "stage_index": stage_index,
            "remaining_steps": remaining_steps,
            "last_content": last_content,
            "message_history": deepcopy(self.message_history),
            "maxsteps": self.maxsteps,
            "memory_access": deepcopy(self.memory_access),
            "timestamp": time.time(),
        }
        
        checkpoint_uid = self.checkpoint_store.save_checkpoint(scope="stage", payload=payload)
        
        if checkpoint_uid:
            stage_name = self.Stages[stage_index].name if stage_index < len(self.Stages) else "unknown"
            if self.logger:
                if self.logger.level == 2:
                    self.logger.log_json({
                        "event": "checkpoint_saved",
                        "scope": "stage",
                        "stage_index": stage_index,
                        "stage_name": stage_name,
                        "checkpoint_uid": checkpoint_uid,
                        "remaining_steps": remaining_steps,
                    })
                else:
                    self.logger.write_line(f"[CHECKPOINT] scope=stage stage_index={stage_index} stage_name={stage_name} uid={checkpoint_uid} remaining={remaining_steps}")
            cli = get_cli_output()
            current_hierarchy = getattr(self, "_parent_hierarchy", []) + [self.name, f"Stage {stage_index}: {stage_name}"]
            checkpoint_msg = f"Checkpoint saved at Stage {stage_index}: {stage_name}\nCheckpoint UID: {checkpoint_uid}\nRemaining steps: {remaining_steps}"
            cli.emit(
                OutputType.AGENT_INIT,
                checkpoint_msg,
                current_hierarchy,
                step=0,
            )
        
        return checkpoint_uid

    def _save_agent_checkpoint(self, remaining_steps: int, last_content: str) -> Optional[str]:
        if not self.checkpoint_store:
            return None
        payload = {
            "scope": "agent",
            "agent_name": self.name,
            "remaining_steps": remaining_steps,
            "last_content": last_content,
            "message_history": deepcopy(self.message_history),
            "maxsteps": self.maxsteps,
            "memory_access": deepcopy(self.memory_access),
            "timestamp": time.time(),
        }
        return self.checkpoint_store.save_checkpoint(scope="agent", payload=payload)

    def load_checkpoint(self, uid: str) -> Optional[dict]:
        if not self.checkpoint_store:
            return None
        checkpoint_data = self.checkpoint_store.load_checkpoint(uid)
        if checkpoint_data:
            self._resume_checkpoint = checkpoint_data
        else:
            self._resume_checkpoint = None
        return checkpoint_data

    def final_prompt(self, stage: IkaStage) -> str:
        parts = [self.start_prompt, stage.prompt, self.end_prompt]
        if not any(parts):
            parts = [self.description, stage.prompt]
        base_prompt = "\n\n".join([p for p in parts if p])
        return base_prompt

    def _build_memory_tools(
        self,
        memory_access: Dict[str, bool],
        memory_type: Literal["short_term", "long_term"],
    ) -> List[AgentTool]:
        tools: List[AgentTool] = []
        prefix = memory_type
        if memory_type == "short_term" and not self.short_term_memory:
            return tools
        if memory_type == "long_term" and not self.long_term_memory:
            return tools

        save_key = f"{prefix}_save"
        search_key = f"{prefix}_search"
        if memory_access.get(save_key, False):
            if memory_type == "short_term":
                tools.append(
                    AgentTool(
                        id=save_key,
                        name=save_key,
                        description="Saves information to short-term memory for this agent's current execution session. The data persists only during this agent's runtime and is lost when the agent completes. Use this tool when you need to remember temporary context, intermediate results, or insights that may be useful later in the current task. It should be called when you discover information worth retaining for future steps. The tool returns a confirmation message upon successful save.",
                        args=ToolArgs(type="input", description="The information, context, or insight to store in short-term memory. Be specific and include relevant details."),
                        required=False,
                    )
                )
            else:
                tools.append(
                    AgentTool(
                        id=save_key,
                        name=save_key,
                        description="Saves a task-output pair to long-term memory for permanent storage across all agent executions. The data persists beyond the current session and can be accessed by any future agent runs with long-term memory access. Use this tool when you complete a task that produces reusable knowledge or results that should be remembered for future executions. The input must follow the format 'task|output' where task describes what was done and output contains the result. The tool returns a confirmation message upon successful save.",
                        args=ToolArgs(type="input", description="The task and output to save permanently, formatted as 'task|output'. Example: 'Calculate sales total|$45,231.50'", data="task|output"),
                        required=False,
                    )
                )

        if memory_access.get(search_key, False):
            if memory_type == "short_term":
                desc = "Search parameters for querying short-term memory"
                tool_desc = "Searches the agent's short-term memory for information relevant to a given query using semantic similarity. Only searches data saved during the current agent execution session. Returns a list of matching memory entries sorted by relevance score, with the most relevant entries first. Use this tool when you need to recall information you previously saved to short-term memory. The tool will not search long-term memory or any external sources."
                prop_desc = "The search query string to find relevant memory entries"
            else:
                desc = "Search parameters for querying long-term memory"
                tool_desc = "Searches long-term memory for task-output pairs relevant to a given query using semantic similarity. Searches across all permanently stored data from previous agent executions, not just the current session. Returns a list of matching task-output pairs sorted by relevance score, with the most relevant entries first. Use this tool when you need to recall information or results from past executions that might help with the current task. The tool will not search short-term memory or any external sources."
                prop_desc = "The search query string to find relevant task-output pairs from past executions"
            tools.append(
                AgentTool(
                    id=search_key,
                    name=search_key,
                    description=tool_desc,
                    args=ToolArgs(
                        type="object",
                        description=desc,
                        properties={
                            "query": {"type": "string", "description": prop_desc},
                            "limit": {"type": "integer", "description": "Maximum number of results to return (default: 5)", "default": 5},
                            "score_threshold": {"type": "number", "description": "Minimum similarity score from 0.0 to 1.0, where 1.0 is perfect match (default: 0.6)", "default": 0.6},
                        },
                        required=["query"],
                    ),
                    required=False,
                )
            )
        return tools

    def _build_stage_memory_tools(self, stage_memory_access: Dict[str, bool]) -> List[AgentTool]:
        tools: List[AgentTool] = []
        tools.extend(self._build_memory_tools(stage_memory_access, "short_term"))
        tools.extend(self._build_memory_tools(stage_memory_access, "long_term"))
        return tools

    def build_stage(self, stage: IkaStage) -> List[AgentTool]:
        stage_tools = self._convert_tools_to_agent_tools(stage.tools)
        subagent_tools = self._convert_subagents_to_tools(getattr(stage, "subagents", None))

        is_last_stage = stage == self.Stages[-1]

        if is_last_stage:
            agent_end_tool = AgentTool(
                id="agent_end",
                name="agent_end",
                description="Terminates the agent execution and returns the final answer to the user or calling system. This tool must be called when you have completed the task specified in your initial prompt. The final answer should be comprehensive, addressing all requirements from the original task. It should be based on your initial prompt and any context you have gathered throughout execution. This tool will immediately end the agent loop, so ensure your answer is complete before calling it. The tool can only be called once per execution. agent_end must be the only tool call in that response and must never be batched with any other tool.",
                args=ToolArgs(type="input", description="Your complete final answer addressing the original task. This parameter is required and cannot be empty."),
                required=True,
                limit_calls=1,
            )
            stage_tools.append(agent_end_tool)

            stage_end_index = next((i for i, t in enumerate(stage_tools) if t.name == "stage_end"), None)
            if stage_end_index is not None:
                stage_tools.pop(stage_end_index)

        stage_memory_access = getattr(stage, "memory_access", None) or self.memory_access
        memory_tools = self._build_stage_memory_tools(stage_memory_access)

        return stage_tools + subagent_tools + memory_tools

    def build_simple_tools(self)-> List[AgentTool]:
        agent_end_tool = AgentTool(
            id="agent_end",
            name="agent_end",
            description="Terminates the agent execution and returns the final answer to the user or calling system. This tool must be called when you have completed the task specified in your initial prompt. The final answer should be comprehensive, addressing all requirements from the original task. It should be based on your initial prompt and any context you have gathered throughout execution. This tool will immediately end the agent loop, so ensure your answer is complete before calling it. The tool can only be called once per execution. agent_end must be the only tool call in that response and must never be batched with any other tool.",
            args=ToolArgs(type="input", description="Your complete final answer addressing the original task. This parameter is required and cannot be empty."),
            required=True,
            limit_calls=1,
        )
        memory_tools = self._build_stage_memory_tools(self.memory_access)
        return self._convert_tools_to_agent_tools(self.tools) + self._convert_subagents_to_tools() + memory_tools + [agent_end_tool]

    def get_barebone(self, system_prompt: str, agent_tools: List[AgentTool], parent_hierarchy: Optional[List[str]] = None, suppress_init_output: bool = False, model_overrides: Optional[Dict[str, Any]] = None, content_prompt_override: Optional[str] = None) -> BareBoneModel:
        """
        conversion to barebone model, this is used to create the model instance for the agent
        barebone model is used for conversion to apis and handling of the model instance.

        Args:
            system_prompt: System prompt for the agent
            agent_tools: List of AgentTool instances
            parent_hierarchy: Optional list of parent agent names for tracking hierarchy
            suppress_init_output: If True, suppress the initialization CLI output
            model_overrides: Optional dict with model_id, api_key, api_url, max_tokens, temperature.
                When a key is present and not None, it overrides the agent's value; otherwise the agent's value is used.
            content_prompt_override: If set, used as content_prompt instead of self.prompt (e.g. the stage-aware first user message in execute_stage).
        """
        o = model_overrides or {}
        model_id = o["model_id"] if "model_id" in o and o["model_id"] is not None else self.model_id
        api_key = o["api_key"] if "api_key" in o and o["api_key"] is not None else self.api_key
        api_url = o["api_url"] if "api_url" in o and o["api_url"] is not None else self.api_url
        max_tokens = o["max_tokens"] if "max_tokens" in o and o["max_tokens"] is not None else self.max_tokens
        temperature = o["temperature"] if "temperature" in o and o["temperature"] is not None else self.temperature
        reasoning_effort = o.get("reasoning_effort") or getattr(self, "reasoning_effort", None)

        parallel_tool_calls = True
        for tool in agent_tools:
            if hasattr(tool, 'parallel') and not tool.parallel:
                parallel_tool_calls = False
                break

        agent_hierarchy = parent_hierarchy if parent_hierarchy else [self.name]
        content_prompt = content_prompt_override if content_prompt_override is not None else self.prompt

        model = BareBoneModel(
            model_id=model_id,
            api_key=api_key,
            api_url=api_url,
            system_prompt=system_prompt,
            content_prompt=content_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            agent_name=self.name,
            agent_hierarchy=agent_hierarchy,
            suppress_init_output=suppress_init_output,
            reasoning_effort=reasoning_effort,
        )
        model.agent_tools = agent_tools
        model._current_step = 0
        return model

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        
        stripped = text.strip()
        
        try:
            json.loads(stripped)
            return stripped
        except Exception:
            pass
        
        patterns = [
            r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```',
            r'(\{.*?\}|\[.*?\])',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, stripped, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)
                    return match
                except Exception:
                    continue
        
        return None

    def _get_end_text(self, args: Dict) -> str:
        agent_end_text = args.get("input", "")
        if agent_end_text and agent_end_text.strip() not in [".", ""]:
            return agent_end_text

        for value in args.values():
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    return value
                if len(stripped) > 10 and stripped != ".":
                    return value
            elif value and not isinstance(value, (dict, list)) and str(value).strip() not in [".", ""]:
                return str(value)

        return ""

    def _validate_final(self, agent_end_text: str) -> str:
        if not agent_end_text or agent_end_text.strip() in [".", ""]:
            return (
                "agent_end was called with empty or invalid arguments. "
                "The agent MUST provide a final answer/output when calling agent_end. "
                "Use the 'input' parameter to pass your response."
            )
        stripped_text = agent_end_text.strip()
        if stripped_text in ["{}", "[]", "null", '""', "''"]:
            return (
                f"agent_end was called with invalid/empty content: '{stripped_text}'. "
                "You MUST provide a meaningful final answer."
            )
        if len(stripped_text) < 3:
            return (
                f"agent_end was called with content that is too short: '{stripped_text}'. "
                "You MUST provide a meaningful final answer."
            )
        return agent_end_text

    def _fallback_final_content(
        self,
        agent_end_text: Optional[str],
        content_before_tools: str,
        last_content: str,
    ) -> str:
        """
        Build final content when agent_end was called but tool arguments were empty.

        Fallback order:
        1. agent_end_text from tool arguments
        2. content_before_tools
        3. last_content
        """
        if agent_end_text and agent_end_text.strip() not in [".", ""]:
            return agent_end_text

        if content_before_tools and content_before_tools.strip() not in ["", "{}"]:
            return content_before_tools
        if last_content and last_content.strip() not in ["", "{}"]:
            return last_content

        raise ValueError(
            "agent_end was called but no output was provided. "
            "The agent MUST provide a final answer when calling agent_end."
        )

    def parse_control_calls(self, tool_calls: List[dict], stage: Optional[IkaStage], current_stage_idx: int = 0, response_content: Optional[str] = None) -> tuple[Optional[int], bool, Optional[str]]:
        target_stage = None
        agent_end_called = False
        agent_end_text = None

        allowed_back = getattr(stage, "allowed_back_to", []) if stage else []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name") or call.get("name")
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}

            if name in ["short_term_save", "short_term_search", "long_term_save", "long_term_search"]:
                self.logger.log_action(f"{name} called")
                continue

            if name == "agent_end":
                agent_end_called = True
                agent_end_text = self._get_end_text(args)

                if not agent_end_text or agent_end_text.strip() in [".", ""]:
                    if response_content and len(response_content.strip()) > 10:
                        agent_end_text = response_content.strip()

                agent_end_text = self._validate_final(agent_end_text or "")

                break
            if name == "stage_end" and stage is not None:
                target_stage = "next"
            if name == "change_stage" and stage is not None:
                stage_idx = args.get("stage_index") or args.get("stage") or args.get("to")
                try:
                    stage_idx = int(stage_idx)
                except Exception:
                    stage_idx = None
                if stage_idx is not None and stage_idx in allowed_back and stage_idx < current_stage_idx:
                    target_stage = stage_idx
        return target_stage, agent_end_called, agent_end_text

    def chat_wrapper(
        self,
        barebone_model: BareBoneModel,
        messages: List[dict],
        tool_executors: Optional[Dict[str, Callable]] = None,
        logger: Optional[Any] = None,
        timeout: float = 900.0,
        max_tool_rounds: int = 5,
        max_tool_calls: Optional[int] = None,
        current_stage_index: Optional[int] = None,
        total_stages: int = 0,
        client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        message_history = self.message_history
        try:
            if self.use_async:
                result = asyncio.run(
                    async_chat(
                        barebone_model,
                        messages,
                        message_history,
                        tool_executors=tool_executors,
                        logger=logger,
                        timeout=timeout,
                        client=client,
                        max_tool_rounds=max_tool_rounds,
                        max_tool_calls=max_tool_calls,
                        current_stage_index=current_stage_index,
                        total_stages=total_stages,
                    )
                )
            else:
                result = chat(
                    barebone_model,
                    messages,
                    message_history,
                    tool_executors=tool_executors,
                    logger=logger,
                    timeout=timeout,
                    max_tool_rounds=max_tool_rounds,
                    max_tool_calls=max_tool_calls,
                    current_stage_index=current_stage_index,
                    total_stages=total_stages,
                )
        except AgentEndException as exc:
            # AgentEndException carries usage/cost in .response — accumulate
            # before re-raising so _total_usage/_total_cost are accurate.
            resp = exc.response or {}
            resp_usage = resp.get("usage") or {}
            for key in ("input_tokens", "output_tokens", "total_tokens", "input_cached_tokens"):
                self._total_usage[key] = self._total_usage.get(key, 0) + ((resp_usage.get(key, 0)) or 0)
            resp_cost = resp.get("cost") or {}
            for key in ("input_cost", "output_cost", "total_cost"):
                self._total_cost[key] = self._total_cost.get(key, 0.0) + ((resp_cost.get(key, 0.0)) or 0.0)
            raise

        # Accumulate usage/cost from this chat round onto the agent totals
        resp_usage = result.get("usage") or {}
        for key in ("input_tokens", "output_tokens", "total_tokens", "input_cached_tokens"):
            self._total_usage[key] = self._total_usage.get(key, 0) + ((resp_usage.get(key, 0)) or 0)
        resp_cost = result.get("cost") or {}
        for key in ("input_cost", "output_cost", "total_cost"):
            self._total_cost[key] = self._total_cost.get(key, 0.0) + ((resp_cost.get(key, 0.0)) or 0.0)

        return result

    def _build_final_output(self, final_message: str, barebone_model: BareBoneModel) -> Dict[str, Any]:
        summary = ""
        if self.summarize_final:
            summary = summarise_message_history(barebone_model, self.message_history) or self.message_history.get("summary", {}).get("message", "")
            if summary:
                if self.logger:
                    self.logger.log_summary(summary)
                cli = get_cli_output()
                current_hierarchy = getattr(self, "_parent_hierarchy", []) + [self.name]
                step = cli.get_step(self.name) or 0
                cli.summarization(self.name, summary, current_hierarchy, step=step)

        if not summary or summary.strip() == "":
            summary = final_message

        if not final_message or final_message.strip() == "":
            final_message = summary

        return {
            "final_message": final_message,
            "summary": summary,
            "usage": getattr(self, "_total_usage", {}),
            "cost": getattr(self, "_total_cost", {}),
            "model_id": barebone_model.model_id,
        }
