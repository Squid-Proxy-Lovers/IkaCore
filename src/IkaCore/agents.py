import builtins
import json
import re
import sys
import time
import typing
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from IkaCore.tools import IkaTools
from IkaCore.stages import IkaStage
from IkaCore.logging_utils import IkaLogger
from IkaCore.checkpoint import CheckpointStore
from IkaCore.cli_output import get_cli_output, OutputType

src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_dir / "IkaMem"))
from IkaMem import STMemory, LTMemory  # type: ignore

from IkaModel.base import (
    BareBoneModel,
    AgentTool,
    ToolArgs,
    init_global_long_term_memory,
    get_global_long_term_memory,
)
from IkaModel.chat_interface import chat, summarise_message_history, execute_tool_calls, get_provider

from .agent_memory import AgentMemoryMixin
from .agent_tools import AgentToolsMixin
from .agent_execution import AgentExecutionMixin


# We add this instruction to all agent prompts to ensure proper task completion
AGENT_END_INSTRUCTION = """
CRITICAL: When you call the agent_end tool, you MUST provide your final answer/output in the tool arguments.
The agent_end tool REQUIRES a non-empty response. You must pass your final answer using the "input" parameter.

DO NOT call agent_end with empty arguments {}. This will cause an error.
Your final answer must be based on your initial prompt and any context you have gathered.
The output format is given by the rest of the prompt.
"""

STAGE_MOVEMENT_INSTRUCTION = """
STAGE MOVEMENT:
Use the stage_end tool when the current stage is complete to advance to the next stage.
If the change_stage tool is available, use it with stage_index (the stage to change to) and reason (why you need to go back). stage_index must be in that stage's allowed_back_to list.
"""

HITL_INSTRUCTION = """
HITL: You can use the ask_user tool when you need the user to answer something. Use it as often as needed. The question must be clear and direct.
"""


class IkaBaseAgent(AgentMemoryMixin, AgentToolsMixin, AgentExecutionMixin):
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
        max_tokens: int = 20000,
        temperature: float = 0.0,
        checkpoint: bool = False,
        Batch: bool = False,
        BatchMax: int = 3,
        Stages: Optional[List[IkaStage]] = None,
        subagents: Optional[List["IkaBaseAgent"]] = None,
        next_agent: Optional["IkaBaseAgent"] = None,
        feedback_agent: Optional["IkaBaseAgent"] = None,
        maxsteps: int = 10,
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
        self.final_answer_checks = self._validate_final_answer_checks(final_answer_check)
        self._tool_call_counts: Dict[str, int] = {}
    
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


    @staticmethod
    def geturl(model_id: str) -> str:
        model_id_lower = model_id.lower()
        if "deepseek" in model_id_lower:
            return "https://api.deepseek.com/chat/completions"
        if "gpt" in model_id_lower or "o1" in model_id_lower or "o3" in model_id_lower:
            return "https://api.openai.com/v1/chat/completions"
        if "claude" in model_id_lower:
            return "https://api.anthropic.com/v1/messages"
        if "gemini" in model_id_lower:
            return "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        return "https://api.openai.com/v1/chat/completions"


    def final_prompt(self, stage: IkaStage) -> str:
        parts = [self.start_prompt, stage.prompt, self.end_prompt]
        if not any(parts):
            parts = [self.description, stage.prompt]
        base_prompt = "\n\n".join([p for p in parts if p])
        return base_prompt

    def _build_short_term_memory_tools(self, memory_access: Dict[str, bool]) -> List[AgentTool]:
        tools: List[AgentTool] = []
        if not self.short_term_memory:
            return tools

        if memory_access.get("short_term_save", False):
            tools.append(
                AgentTool(
                    id="short_term_save",
                    name="short_term_save",
                    description="Save data to short-term memory for this agent. Use for temporary context or insights.",
                    args=ToolArgs(type="input", description="Data or insight to save temporarily."),
                    required=False,
                )
            )
            
        if memory_access.get("short_term_search", False):
            tools.append(
                AgentTool(
                    id="short_term_search",
                    name="short_term_search",
                    description="Search short-term memory for relevant information. Returns matching entries sorted by relevance.",
                    args=ToolArgs(
                        type="object",
                        description="Search parameters",
                        properties={
                            "query": {"type": "string", "description": "Search query to find relevant memory entries"},
                            "limit": {"type": "integer", "description": "Maximum number of results (default: 5)", "default": 5},
                            "score_threshold": {"type": "number", "description": "Minimum similarity score 0-1 (default: 0.6)", "default": 0.6},
                        },
                        required=["query"],
                    ),
                    required=False,
                )
            )
        return tools

    def _build_long_term_memory_tools(self, memory_access: Dict[str, bool]) -> List[AgentTool]:
        tools: List[AgentTool] = []
        if not self.long_term_memory:
            return tools

        if memory_access.get("long_term_save", False):
            tools.append(
                AgentTool(
                    id="long_term_save",
                    name="long_term_save",
                    description="Save task and output to long-term memory. Format: task|output",
                    args=ToolArgs(type="input", description="Data or insight to save forever.", data="task|output"),
                    required=False,
                )
            )
            
        if memory_access.get("long_term_search", False):
            tools.append(
                AgentTool(
                    id="long_term_search",
                    name="long_term_search",
                    description="Search long-term memory for relevant past tasks and outputs. Returns matching entries sorted by relevance.",
                    args=ToolArgs(
                        type="object",
                        description="Search parameters",
                        properties={
                            "query": {"type": "string", "description": "Search query to find relevant memory entries"},
                            "limit": {"type": "integer", "description": "Maximum number of results (default: 5)", "default": 5},
                            "score_threshold": {"type": "number", "description": "Minimum similarity score 0-1 (default: 0.6)", "default": 0.6},
                        },
                        required=["query"],
                    ),
                    required=False,
                )
            )
        return tools

    def _build_stage_memory_tools(self, stage_memory_access: Dict[str, bool]) -> List[AgentTool]:
        tools: List[AgentTool] = []
        tools.extend(self._build_short_term_memory_tools(stage_memory_access))
        tools.extend(self._build_long_term_memory_tools(stage_memory_access))
        return tools

    def build_stage(self, stage: IkaStage) -> List[AgentTool]:
        stage_tools = self._convert_tools_to_agent_tools(stage.tools)
        subagent_tools = self._convert_subagents_to_tools(getattr(stage, "subagents", None))

        is_last_stage = stage == self.Stages[-1]

        if is_last_stage:
            agent_end_tool = AgentTool(
                id="agent_end",
                name="agent_end",
                description="End the agent loop with a final answer. Use this tool when you have completed the task. Provide the final output as detailed as possible, this should be based on your initial prompt and any context you have gathered. CRITICAL: You MUST provide your final answer in the 'input' parameter. Do NOT call this tool with empty arguments.",
                args=ToolArgs(type="input", description="Final response content. This is REQUIRED - provide your complete final answer here."),
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
        )
        model.agent_tools = agent_tools
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

    def _get_agent_end_text_from_args(self, args: Dict) -> str:
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

    def _ensure_valid_agent_end_content(self, agent_end_text: str) -> None:
        if not agent_end_text or agent_end_text.strip() in [".", ""]:
            raise ValueError(
                "agent_end was called with empty or invalid arguments. "
                "The agent MUST provide a final answer/output when calling agent_end. "
                "Use the 'input' parameter to pass your response."
            )

        stripped_text = agent_end_text.strip()
        if stripped_text in ["{}", "[]", "null", '""', "''"]:
            raise ValueError(
                f"agent_end was called with invalid/empty content: '{stripped_text}'. "
                "You MUST provide a meaningful final answer."
            )

        if len(stripped_text) < 3:
            raise ValueError(
                f"agent_end was called with content that is too short: '{stripped_text}'. "
                "You MUST provide a meaningful final answer."
            )

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

            # Log memory tool usage (tools are executed by chat function via tool_executors)
            if name in ["short_term_save", "short_term_search", "long_term_save", "long_term_search"]:
                self.logger.log_action(f"{name} called")
                continue

            if name == "agent_end":
                agent_end_called = True
                agent_end_text = self._get_agent_end_text_from_args(args)

                if not agent_end_text or agent_end_text.strip() in [".", ""]:
                    if response_content and len(response_content.strip()) > 10:
                        agent_end_text = response_content.strip()
                
                self._ensure_valid_agent_end_content(agent_end_text)
                
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

    def execute_stage(self, stage_index: int, remaining_steps: int) -> tuple[int, str, bool, Optional[str], int]:
        from IkaModel.chat_interface import chat, async_chat
        import asyncio

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
        parts.append(STAGE_MOVEMENT_INSTRUCTION)
        if getattr(stage, "hitl", False):
            parts.append(HITL_INSTRUCTION)
        if stage_index == len(self.Stages) - 1:
            parts.append(AGENT_END_INSTRUCTION)
        content_prompt = "\n\n".join(parts)

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
        barebone_model._tool_call_counts = self._tool_call_counts

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

        for step_num in range(step_limit):
            if self.logger:
                self.logger.log_action(f"stage_start:{stage.name}")
            barebone_model._current_step = step_num + 1
            self._enforce_rate_limit_model()
            step_start = time.time()

            if self.use_async:
                response = asyncio.run(
                    async_chat(
                        barebone_model,
                        messages,
                        self.message_history,
                        tool_executors=tool_executors,
                        logger=self.logger,
                        timeout=self.step_timeout,
                    )
                )
            else:
                response = chat(
                    barebone_model,
                    messages,
                    self.message_history,
                    tool_executors=tool_executors,
                    logger=self.logger,
                    timeout=self.step_timeout,
                )
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

            messages = [{"role": "assistant", "content": last_content}]

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

        if self.logger:
            self.logger.log_stage_end(stage.name, used_steps)
        return stage_index + 1, last_content, False, None, used_steps

    def run_simple(self) -> tuple[str, str]:
        agent_end_tool = AgentTool(
            id="agent_end",
            name="agent_end",
            description="""End the agent loop with a final answer.
                Provide the final output and any key reasoning.
                CRITICAL: You MUST provide your final answer in the 'input' parameter. 
                Do NOT call this tool with empty arguments.""",
            args=ToolArgs(type="input", description="Final response content. This is REQUIRED - provide your complete final answer here."),
            required=True,
            limit_calls=1,
        )
        
        memory_tools = self._build_stage_memory_tools(self.memory_access)
        
        agent_tools = self._convert_tools_to_agent_tools(self.tools) + self._convert_subagents_to_tools() + memory_tools + [agent_end_tool]
        base_prompt = self.system_prompt or self.description or self.prompt
        system_prompt = base_prompt
        current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
        barebone_model = self.get_barebone(system_prompt, agent_tools, parent_hierarchy=current_hierarchy)

        first_msg = (self.message_history.get("first_input") or {}).get("message") or ""
        content_prompt = (first_msg or self.prompt or "") + "\n\n" + AGENT_END_INSTRUCTION
        messages: List[dict] = [{"role": "user", "content": content_prompt}]
        last_content = ""
        last_agent_end_text = None

        current_hierarchy = getattr(self, '_parent_hierarchy', []) + [self.name]
        tool_executors = self.build_tool_executors(self.tools, memory_access=self.memory_access, subagents=self.subagents, parent_hierarchy=current_hierarchy)
        
        cli = get_cli_output()

        from IkaModel.chat_interface import chat, async_chat
        import asyncio

        for step_num in range(self.maxsteps):
            cli.set_step(self.name, step_num + 1)
            cli.agent_init(
                self.name,
                current_hierarchy,
                step=step_num + 1,
                description=f"Step {step_num + 1}/{self.maxsteps}"
            )
            barebone_model._current_step = step_num + 1
            self._enforce_rate_limit_model()
            step_start = time.time()
            if self.use_async:
                response = asyncio.run(
                    async_chat(
                        barebone_model,
                        messages,
                        self.message_history,
                        tool_executors=tool_executors,
                        logger=self.logger,
                        timeout=self.step_timeout,
                    )
                )
            else:
                response = chat(
                    barebone_model,
                    messages,
                    self.message_history,
                    tool_executors=tool_executors,
                    logger=self.logger,
                    timeout=self.step_timeout,
                )
            self.message_history = response.get("message_history", self.message_history)
            last_content = response.get("content", "")
            content_before_tools = response.get("content_before_tools", "")
            tool_calls = response.get("tool_calls", []) or []
            executed_tool_calls = response.get("executed_tool_calls", []) or []
            if hasattr(barebone_model, '_tool_call_counts'):
                self._tool_call_counts.update(barebone_model._tool_call_counts)

            try:
                _, agent_end_called, agent_end_text = self.parse_control_calls(
                    executed_tool_calls if executed_tool_calls else tool_calls,
                    None,
                    response_content=content_before_tools,
                )
                if agent_end_called and agent_end_text:
                    last_agent_end_text = agent_end_text
            except ValueError as e:
                error_msg = str(e)
                cli.agent_response(
                    self.name,
                    f"ERROR: {error_msg}",
                    current_hierarchy,
                    step=step_num + 1,
                    is_final=False,
                )
                raise
            
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
                        step=step_num + 1,
                        is_final=False,
                    )
                    raise
                
                cli.agent_response(
                    self.name,
                    final_content,
                    current_hierarchy,
                    step=step_num + 1,
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

            # Handle pending tool calls that weren't executed by chat (e.g. chained calls)
            # But skip if agent_end was already called - we should have returned by now
            if tool_calls and not agent_end_called:
                provider = get_provider(barebone_model.model_id)
                try:
                    tool_msgs, tool_res = execute_tool_calls(
                        tool_calls, tool_executors, provider, self.step_timeout,
                        agent_hierarchy=current_hierarchy, step=step_num + 1
                    )
                    if self.logger:
                        self.logger.log_tool_results(tool_calls, tool_res)
                    messages.extend(tool_msgs)
                    
                    # Check again after tool execution if agent_end was called
                    _, agent_end_called_after, agent_end_text_after = self.parse_control_calls(
                        tool_calls,
                        None,
                        response_content=content_before_tools,
                    )
                    if agent_end_called_after:
                        try:
                            final_content = self._fallback_final_content(
                                agent_end_text_after,
                                content_before_tools,
                                last_content,
                            )
                        except ValueError:
                                final_content = agent_end_text_after or "Agent completed."
                        
                        cli.agent_response(
                            self.name,
                            final_content,
                            current_hierarchy,
                            step=step_num + 1,
                            is_final=True
                        )
                        return final_content, final_content
                except ValueError as e:
                    if "agent_end" in str(e).lower():
                        raise
                    tool_msgs = []
                    tool_res = []

            # Do NOT reset messages to preserve context
            # messages = [{"role": "assistant", "content": last_content}]

            if not tool_calls and last_content:
                # If the agent is just thinking or talking, we let it continue unless it's the last step
                # We append the message to history so it remembers what it said
                messages.append({"role": "assistant", "content": last_content})
                cli.agent_response(
                    self.name,
                    last_content,
                    current_hierarchy,
                    step=step_num + 1,
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
                # Continue to next step instead of returning
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
            # Ensure maxsteps is an integer and step_num is an integer
            max_steps_val = int(self.maxsteps)
            current_step_val = int(step_num) if step_num is not None else 0
            remaining_after = max(0, max_steps_val - (current_step_val + 1))
            self._save_agent_checkpoint(remaining_after, last_content)

        final_response = last_agent_end_text if last_agent_end_text else last_content
        cli.agent_response(
            self.name,
            f"Reached max steps ({self.maxsteps}). Returning last content.\n\n{final_response}",
            current_hierarchy,
            step=self.maxsteps,
            is_final=True
        )
        return final_response, final_response

    def _build_final_output(self, final_message: str, barebone_model: BareBoneModel) -> Dict[str, str]:
        summary = ""
        if self.summarize_final:
            summary = summarise_message_history(barebone_model, self.message_history) or self.message_history.get("summary", {}).get("message", "")
            if summary:
                if self.logger:
                    self.logger.log_summary(summary)
                # Also emit via CLI output with dedicated summarization color and no truncation
                cli = get_cli_output()
                current_hierarchy = getattr(self, "_parent_hierarchy", []) + [self.name]
                # Use final step number for summary box
                step = cli.get_step(self.name) or 0
                cli.summarization(self.name, summary, current_hierarchy, step=step)

        # Always use final_message as summary fallback when summary is empty
        if not summary or summary.strip() == "":
            summary = final_message

        if not final_message or final_message.strip() == "":
            final_message = summary

        return {"final_message": final_message, "summary": summary}

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
