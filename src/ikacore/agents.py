from typing import Optional, List, Dict, Callable, get_type_hints
import json
import sys
from pathlib import Path
import time
import builtins
from copy import deepcopy

from tools import SquidTools
from stages import SquidStage
from squidrag import SquidRAGSource
from memory import Memory
from logging_utils import ParaLogger
from checkpoint import CheckpointStore

# Import IkaMem for short-term and long-term memory
sys.path.insert(0, str(Path(__file__).parent.parent / "IkaMem"))
from IkaMem import STMemory, LTMemory, STMemItem, LTMemItem

# Dynamically load ika-model base and chat_interface to avoid package issues
import importlib.util

base_model_path = Path(__file__).parent.parent / "ika-model" / "base.py"
base_spec = importlib.util.spec_from_file_location("base", base_model_path)
base = importlib.util.module_from_spec(base_spec)
sys.modules["base"] = base
base_spec.loader.exec_module(base)

BareBoneModel = base.BareBoneModel
AgentTool = base.AgentTool
ToolArgs = base.ToolArgs
init_global_long_term_memory = base.init_global_long_term_memory
get_global_long_term_memory = base.get_global_long_term_memory

chat_interface_path = Path(__file__).parent.parent / "ika-model" / "chat_interface.py"
chat_spec = importlib.util.spec_from_file_location("chat_interface", chat_interface_path)
chat_interface = importlib.util.module_from_spec(chat_spec)
sys.modules["chat_interface"] = chat_interface
chat_spec.loader.exec_module(chat_interface)

chat = chat_interface.chat
summarise_message_history = chat_interface.summarise_message_history


class ParaBaseAgent:
    def __init__(
        self,
        name: str,
        description: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        start_prompt: Optional[str] = None,
        end_prompt: Optional[str] = None,
        role: str = "",
        tools: List[SquidTools] = None,
        model_id: str = "",
        api_key: str = "",
        api_url: Optional[str] = None,
        max_tokens: int = 20000,
        temperature: float = 0.0,
        checkpoint: bool = False,
        Batch: bool = False,
        BatchMax: int = 3,
        Stages: List[SquidStage] = None,
        subagents: Optional[List["ParaBaseAgent"]] = None,
        next_agent: Optional["ParaBaseAgent"] = None,
        feedback_agent: Optional["ParaBaseAgent"] = None,
        maxsteps: int = 10,
        step_timeout: int = 900,
        rate_limit_per_min: Optional[float] = None,
        per_tool_rate_limit: Optional[Dict[str, float]] = None,
        RAGSource: Optional[List[type[SquidRAGSource]]] = None,
        memory: bool = False,
        memory_finder: Optional["Memory"] = None,
        memory_access: Optional[Dict[str, bool]] = None,
        final_answer_check: Optional[List[Callable]] = None,
        logging_level: int = 0,
        logging_file: str = "logs.txt",
        show_usage_level0: bool = True,
        checkpoint_db_path: str = "checkpoints.db",
    ):
        tools = tools or []
        Stages = Stages or []

        if not name:
            raise ValueError("name is required for the agent")
        if not description:
            raise ValueError("description is required for the agent")
        if not prompt:
            raise ValueError("prompt is required for the agent")
        if not role:
            raise ValueError("role is required for the agent")
        if not tools and not Stages:
            raise ValueError("tools is required for the agent when no stages are provided")
        if not model_id:
            raise ValueError("model_id is required for the agent")
        if not api_key:
            raise ValueError("api_key is required for the agent")

        self.name = name
        self.description = description
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.start_prompt = start_prompt
        self.end_prompt = end_prompt
        self.role = role
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
        self.subagents = subagents or []
        self.next_agent = next_agent
        self.feedback_agent = feedback_agent

        self.maxsteps = maxsteps
        self.step_timeout = step_timeout
        self.RAGSource = RAGSource
        self.memory = memory
        self.memory_finder = memory_finder
        
        # Memory access control: defaults allow all if memory is enabled
        # Keys: "short_term_save", "short_term_search", "long_term_save", "long_term_search"
        default_memory_access = {
            "short_term_save": True,
            "short_term_search": True,
            "long_term_save": True,
            "long_term_search": True,
        } if memory else {
            "short_term_save": False,
            "short_term_search": False,
            "long_term_save": False,
            "long_term_search": False,
        }
        self.memory_access = {**default_memory_access, **(memory_access or {})}
        
        self.final_answer_check = final_answer_check
        self.message_history = {
            "system": {"message": self.system_prompt or "", "tokens": 0},
            "first_input": {"message": "", "tokens": 0},
            "summary": {"message": "", "tokens": 0},
            "messages": {},
        }

        self.logging_level = logging_level
        self.logging_file = logging_file
        self.logger = ParaLogger(logging_level, logging_file, show_usage_level0=show_usage_level0)
        self.rate_limit_per_min = rate_limit_per_min
        self.per_tool_rate_limit = per_tool_rate_limit or {}
        self._last_api_call_ts: float = 0.0
        self.checkpoint_store = CheckpointStore(checkpoint_db_path) if checkpoint else None
        self._resume_checkpoint: Optional[dict] = None
        
        self.short_term_memory: Optional[STMemory] = None
        self.long_term_memory: Optional[LTMemory] = get_global_long_term_memory()

        if final_answer_check is None:
            final_answer_check = []
        
        for check in final_answer_check:
            if not callable(check):
                raise ValueError("final_answer_check must be a list of callable functions")
            
            func_name = getattr(check, '__name__', 'unknown')
            
            if not check.__doc__ or not check.__doc__.strip():
                raise ValueError(f"final_answer_check function '{func_name}' must have a docstring explaining what it does")
            
            try:
                hints = get_type_hints(check)
                return_type = hints.get('return', None)
                if return_type is not None:
                    import typing
                    if hasattr(typing, 'get_origin'):
                        origin = typing.get_origin(return_type)
                        args = typing.get_args(return_type) if origin else ()
                        is_bool = (return_type is bool or (origin is not None and bool in args))
                    else:
                        is_bool = (return_type is bool)
                    
                    if not is_bool:
                        raise ValueError(f"final_answer_check function '{func_name}' must have return type annotation of bool, got {return_type}")
            except (TypeError, AttributeError):
                pass
            
            result = check(self.message_history)
            if not isinstance(result, bool):
                raise ValueError(f"final_answer_check function '{func_name}' must return a boolean, got {type(result).__name__}")
        
        self.final_answer_checks = final_answer_check

        if self.Stages:
            if self.subagents or self.next_agent or self.feedback_agent:
                raise ValueError("Subagents/next/feedback agents are not allowed when stages are defined.")
        else:
            if self.subagents and self.next_agent:
                raise ValueError("Only one of subagents or next_agent may be set when no stages are provided.")

    def _enforce_rate_limit(self, per_min: Optional[float], last_ts_attr: str) -> None:
        if not per_min or per_min <= 0:
            return
        interval = 60.0 / per_min
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

    def _prompt_hitl_input(self, stage_name: str, last_response: str = "") -> Optional[str]:
        if self.logger:
            self.logger.log_hitl_prompt(stage_name)
            if last_response:
                self.logger.write_line(self.logger._color(f"[HITL LAST] {last_response}", "yellow"))
        try:
            user_text = builtins.input(f"[HITL:{stage_name}] Enter message or 'stage_end' or '/end' to finish (empty to continue): ").strip()
            if self.logger and user_text:
                self.logger.log_hitl_input(stage_name, user_text)
            return user_text
        except EOFError:
            return None

    # --------------------
    # Workflow integration
    # --------------------
    def inject_workflow_context(self, context: str) -> None:
        if not context:
            return
        current_first_input = self.message_history.get("first_input", {}).get("message", "")
        if current_first_input:
            combined = f"{context}\n\n{current_first_input}"
        else:
            combined = context
        self.message_history["first_input"]["message"] = combined

    def apply_workflow_stage_wiring(self, stage_wiring: Dict[int, Dict[str, List["ParaBaseAgent"]]]) -> None:
        if not stage_wiring or not self.Stages:
            return
        for stage_idx, wiring in stage_wiring.items():
            if 0 <= stage_idx < len(self.Stages):
                stage = self.Stages[stage_idx]
                if "subagents" in wiring:
                    stage.subagents = wiring["subagents"]

    # --------------------
    # Checkpoint helpers
    # --------------------
    def _save_stage_checkpoint(self, stage_index: int, remaining_steps: int, last_content: str) -> Optional[str]:
        if not self.checkpoint_store:
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
        return self.checkpoint_store.save_checkpoint(scope="stage", payload=payload)

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
        cp = self.checkpoint_store.load_checkpoint(uid)
        if cp:
            self._resume_checkpoint = cp
        return cp

    def init_short_term_memory(self, embedder_config: dict) -> None:
        self.short_term_memory = STMemory(embedder_config=embedder_config)
        self.short_term_memory.agent = self.name
    
    def _save_to_short_term(self, data: str, metadata: Optional[Dict] = None) -> str:
        if not self.short_term_memory:
            return "error: short-term memory not initialized"
        
        try:
            item = STMemItem(data=data, agent=self.name, metadata=metadata or {})
            self.short_term_memory.storage.save(item.data, item.metadata)
            return f"saved to short-term memory: {data[:50]}..."
        except Exception as e:
            return f"error saving to short-term memory: {str(e)}"
    
    def _save_to_long_term(self, task: str, output: str) -> str:
        """internal: save to long-term memory."""
        if not self.long_term_memory:
            return "error: long-term memory not initialized"
        
        try:
            from datetime import datetime
            item = LTMemItem(
                agent=self.name,
                task=task,
                expected_output=output,
                datetime=datetime.now().isoformat(),
                quality=1.0,  # default quality
                metadata={}
            )
            self.long_term_memory.save(item)
            return f"saved to long-term memory - task: {task[:30]}..."
        except Exception as e:
            return f"error saving to long-term memory: {str(e)}"

    def _search_short_term(self, query: str, limit: int = 5, score_threshold: float = 0.6) -> dict:
        """internal: search short-term memory; returns structured results."""
        if not self.short_term_memory:
            return {"error": "short-term memory not initialized"}
        query = (query or "").strip()
        if not query:
            return {"error": "query is required"}
        limit = max(1, min(int(limit or 5), 50))
        score_threshold = max(0.0, min(float(score_threshold or 0.6), 1.0))
        try:
            results = self.short_term_memory.search(query=query, limit=limit, score_threshold=score_threshold)
            structured = []
            for result in results or []:
                if isinstance(result, dict):
                    structured.append({
                        "data": result.get("data") or result.get("content") or str(result),
                        "metadata": result.get("metadata", {}),
                    })
                else:
                    structured.append({"data": str(result), "metadata": {}})
            return {
                "query": query,
                "limit": limit,
                "score_threshold": score_threshold,
                "count": len(structured),
                "results": structured,
            }
        except Exception as e:
            return {"error": f"error searching short-term memory: {str(e)}"}

    def _search_long_term(self, query: str, limit: int = 5, score_threshold: float = 0.6, filter_func: Optional[Callable] = None) -> dict:
        """internal: search long-term memory; returns structured results."""
        if not self.long_term_memory:
            return {"error": "long-term memory not initialized"}
        query = (query or "").strip()
        if not query:
            return {"error": "query is required"}
        limit = max(1, min(int(limit or 5), 50))
        score_threshold = max(0.0, min(float(score_threshold or 0.6), 1.0))
        try:
            results = self.long_term_memory.search(query=query, limit=limit, score_threshold=score_threshold)
            if filter_func:
                try:
                    results = filter_func(results)
                except Exception:
                    pass
            structured = []
            for result in results or []:
                if hasattr(result, "task") and hasattr(result, "expected_output"):
                    structured.append({
                        "task": getattr(result, "task", "") or "",
                        "output": getattr(result, "expected_output", "") or "",
                        "metadata": getattr(result, "metadata", {}) if hasattr(result, "metadata") else {},
                    })
                elif isinstance(result, dict):
                    structured.append({
                        "task": result.get("task", result.get("content", "")),
                        "output": result.get("expected_output", result.get("output", "")),
                        "metadata": result.get("metadata", {}),
                    })
                else:
                    structured.append({"task": str(result), "output": "", "metadata": {}})
            return {
                "query": query,
                "limit": limit,
                "score_threshold": score_threshold,
                "count": len(structured),
                "results": structured,
            }
        except Exception as e:
            return {"error": f"error searching long-term memory: {str(e)}"}

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


    def _convert_subagents_to_tools(self, subagents: Optional[List["ParaBaseAgent"]] = None) -> List[AgentTool]:
        agent_tools: List[AgentTool] = []
        source_subagents = subagents if subagents is not None else self.subagents
        for subagent in source_subagents:
            tool_args = ToolArgs(
                type="input",
                description=f"Task request to subagent {getattr(subagent, 'name', 'subagent')}",
            )
            agent_tool = AgentTool(
                id=getattr(subagent, "name", "subagent"),
                name=getattr(subagent, "name", "subagent"),
                description=getattr(subagent, "description", "Subagent"),
                args=tool_args,
                required=True,
            )
            agent_tools.append(agent_tool)
        return agent_tools

    def _convert_tools_to_agent_tools(self, stage_tools: List[SquidTools]) -> List[AgentTool]:
        converted: List[AgentTool] = []
        for tool in stage_tools:
            if isinstance(tool, AgentTool):
                converted.append(tool)
                continue
            tool_args = ToolArgs(
                type="input",
                description=tool.description or "Tool input",
            )
            converted.append(
                AgentTool(
                    id=tool.id,
                    name=tool.name,
                    description=tool.description,
                    args=tool_args,
                    required=tool.required,
                )
            )
        return converted

    def _build_subagent_executor(self, subagent: "ParaBaseAgent") -> Callable:
        def subagent_executor(args: dict) -> str:
            task_input = args.get("input") or args.get("task") or ""
            if not task_input:
                return json.dumps({"error": "No input provided for subagent"})
            
            try:
                if self.logger:
                    self.logger.log_action(f"Calling subagent: {subagent.name}")
                
                subagent.message_history["first_input"]["message"] = task_input
                subagent.prompt = task_input
                
                result = subagent.execution()
                final_output = result.get("final_message", "")
                summary = result.get("summary", final_output)
                
                if self.logger:
                    self.logger.log_action(f"Subagent {subagent.name} completed")
                
                return summary or final_output
            except Exception as e:
                error_msg = f"Error executing subagent '{subagent.name}': {str(e)}"
                if self.logger:
                    self.logger.log_action(error_msg)
                return json.dumps({"error": error_msg})
        
        return subagent_executor

    def build_tool_executors(self, tools: List[SquidTools], memory_access: Optional[Dict[str, bool]] = None, long_term_filter: Optional[Callable] = None, subagents: Optional[List["ParaBaseAgent"]] = None) -> Dict[str, Callable]:
        tool_executors = {}
        for tool in tools:
            if hasattr(tool, 'execute_function') and tool.execute_function:
                tool_executors[tool.name] = tool.execute_function
        
        # Add subagent executors
        source_subagents = subagents if subagents is not None else self.subagents
        if source_subagents:
            for subagent in source_subagents:
                subagent_name = getattr(subagent, "name", "subagent")
                tool_executors[subagent_name] = self._build_subagent_executor(subagent)
        
        # Add memory tool executors based on access control
        effective_access = memory_access or self.memory_access
        if self.short_term_memory:
            if effective_access.get("short_term_save", False):
                def short_save_executor(args: dict) -> str:
                    self._enforce_rate_limit_tool("short_term_save")
                    data = args.get("input") or args.get("data") or ""
                    return self._save_to_short_term(data)
                tool_executors["short_term_save"] = short_save_executor
            
            if effective_access.get("short_term_search", False):
                def short_search_executor(args: dict) -> str:
                    self._enforce_rate_limit_tool("short_term_search")
                    query = args.get("query", "")
                    limit = args.get("limit", 5)
                    score_threshold = args.get("score_threshold", 0.6)
                    return self._search_short_term(query, limit=limit, score_threshold=score_threshold)
                tool_executors["short_term_search"] = short_search_executor
        
        if self.long_term_memory:
            if effective_access.get("long_term_save", False):
                def long_save_executor(args: dict) -> str:
                    self._enforce_rate_limit_tool("long_term_save")
                    data = args.get("data") or args.get("input") or ""
                    if "|" in data:
                        parts = data.split("|", 1)
                        task = parts[0].strip()
                        output = parts[1].strip()
                        return self._save_to_long_term(task, output)
                    return "error: long_term_save requires format 'task|output'"
                tool_executors["long_term_save"] = long_save_executor
            
            if effective_access.get("long_term_search", False):
                def long_search_executor(args: dict) -> str:
                    self._enforce_rate_limit_tool("long_term_search")
                    query = args.get("query", "")
                    limit = args.get("limit", 5)
                    score_threshold = args.get("score_threshold", 0.6)
                    return self._search_long_term(query, limit=limit, score_threshold=score_threshold, filter_func=long_term_filter)
                tool_executors["long_term_search"] = long_search_executor
        
        return tool_executors

    def final_pronpt(self, stage: SquidStage) -> str:
        parts = [self.start_prompt, stage.prompt, self.end_prompt]
        if not any(parts):
            parts = [self.description, stage.prompt]
        return "\n\n".join([p for p in parts if p])

    def build_stage(self, stage: SquidStage) -> List[AgentTool]:
        stage_tools = self._convert_tools_to_agent_tools(stage.tools)
        subagent_tools = self._convert_subagents_to_tools(getattr(stage, "subagents", None))
        
        agent_end_tool = AgentTool(
            id="agent_end",
            name="agent_end",
            description="End the agent loop with a final answer. Provide the final output and any key reasoning.",
            args=ToolArgs(type="input", description="Final response content."),
            required=False,
        )
        
        # Get stage-specific memory access (inherits from agent if not overridden)
        stage_memory_access = getattr(stage, "memory_access", None) or self.memory_access
        
        memory_tools = []
        
        # Short-term memory tools
        if self.short_term_memory:
            if stage_memory_access.get("short_term_save", False):
                short_save_tool = AgentTool(
                    id="short_term_save",
                    name="short_term_save",
                    description="Save data to short-term memory for this agent. Use for temporary context or insights.",
                    args=ToolArgs(type="input", description="Data or insight to save temporarily."),
                    required=False,
                )
                memory_tools.append(short_save_tool)
            
            if stage_memory_access.get("short_term_search", False):
                short_search_tool = AgentTool(
                    id="short_term_search",
                    name="short_term_search",
                    description="Search short-term memory for relevant information. Returns matching entries sorted by relevance.",
                    args=ToolArgs(
                        type="object",
                        description="Search parameters",
                        properties={
                            "query": {"type": "string", "description": "Search query to find relevant memory entries"},
                            "limit": {"type": "integer", "description": "Maximum number of results (default: 5)", "default": 5},
                            "score_threshold": {"type": "number", "description": "Minimum similarity score 0-1 (default: 0.6)", "default": 0.6}
                        },
                        required=["query"]
                    ),
                    required=False,
                )
                memory_tools.append(short_search_tool)
        
        # Long-term memory tools
        if self.long_term_memory:
            if stage_memory_access.get("long_term_save", False):
                long_save_tool = AgentTool(
                    id="long_term_save",
                    name="long_term_save",
                    description="Save task and output to long-term memory. Format: task|output",
                    args=ToolArgs(type="input", description="Data or insight to save forever.", data="task|output"),
                    required=False,
                )
                memory_tools.append(long_save_tool)
            
            if stage_memory_access.get("long_term_search", False):
                long_search_tool = AgentTool(
                    id="long_term_search",
                    name="long_term_search",
                    description="Search long-term memory for relevant past tasks and outputs. Returns matching entries sorted by relevance.",
                    args=ToolArgs(
                        type="object",
                        description="Search parameters",
                        properties={
                            "query": {"type": "string", "description": "Search query to find relevant memory entries"},
                            "limit": {"type": "integer", "description": "Maximum number of results (default: 5)", "default": 5},
                            "score_threshold": {"type": "number", "description": "Minimum similarity score 0-1 (default: 0.6)", "default": 0.6}
                        },
                        required=["query"]
                    ),
                    required=False,
                )
                memory_tools.append(long_search_tool)
        
        return stage_tools + subagent_tools + memory_tools + [agent_end_tool]

    def get_barebone(self, system_prompt: str, agent_tools: List[AgentTool]) -> BareBoneModel:
        model = BareBoneModel(
            model_id=self.model_id,
            api_key=self.api_key,
            api_url=self.api_url,
            system_prompt=system_prompt,
            content_prompt=self.prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        model.agent_tools = agent_tools
        return model

    def parse_control_calls(self, tool_calls: List[dict], stage: Optional[SquidStage], current_stage_idx: int = 0) -> tuple[Optional[int], bool, Optional[str]]:
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
                agent_end_text = args.get("input") or args.get("final") or args.get("message") or ""
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

    def excute_stage(self, stage_index: int, remaining_steps: int) -> tuple[int, str, bool, Optional[str], int]:
        stage = self.Stages[stage_index]
        system_prompt = self.final_pronpt(stage)
        self.message_history["system"]["message"] = system_prompt

        agent_tools = self.build_stage(stage)
        barebone_model = self.get_barebone(system_prompt, agent_tools)

        messages: List[dict] = [{"role": "user", "content": stage.prompt or self.prompt}]
        last_content = ""

        used_steps = 0
        step_limit = min(getattr(stage, "stage_max_step", 1), max(1, remaining_steps))
        if getattr(stage, "hitl", False):
            step_limit = max(1, remaining_steps)

        stage_memory_access = getattr(stage, "memory_access", None) or self.memory_access
        tool_executors = self.build_tool_executors(
            stage.tools, 
            memory_access=stage_memory_access, 
            long_term_filter=getattr(stage, "long_term_filter", None),
            subagents=getattr(stage, "subagents", None)
        )
        
        if self.logger:
            self.logger.log_stage_start(stage.name, getattr(stage, "hitl", False), remaining_steps, step_limit)

        for _ in range(step_limit): # run till we reach the step limit or we call agent_end
            if self.logger:
                self.logger.log_action(f"stage_start:{stage.name}")
            self._enforce_rate_limit_model()
            step_start = time.time()
            response = chat(
                barebone_model, 
                messages, 
                self.message_history, 
                tool_executors=tool_executors,
                logger=self.logger,
                timeout=self.step_timeout
            )
            self.message_history = response.get("message_history", self.message_history)
            last_content = response.get("content", "")
            tool_calls = response.get("tool_calls", []) or []
            used_steps += 1

            target_stage, agent_end_called, agent_end_text = self.parse_control_calls(tool_calls, stage, stage_index)
            if agent_end_called:
                return stage_index, agent_end_text or last_content, True, agent_end_text or last_content, used_steps
            if target_stage == "next":
                return stage_index + 1, last_content, False, None, used_steps
            if isinstance(target_stage, int):
                return target_stage, last_content, False, None, used_steps

            messages = [{"role": "assistant", "content": last_content}]

            if getattr(stage, "hitl", False):
                user_text = self._prompt_hitl_input(stage.name, last_response=last_content)
                if user_text is None:
                    continue
                lower_text = user_text.lower()
                if lower_text in ("stage_end", "/end", "end", "exit"):
                    if self.logger:
                        self.logger.log_stage_end(stage.name, used_steps)
                    return stage_index + 1, last_content, False, None, used_steps
                if user_text:
                    messages.append({"role": "user", "content": user_text})

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
            self._save_stage_checkpoint(stage_index, remaining_after, last_content)

        if getattr(stage, "hitl", False):
            if self.logger:
                self.logger.log_stage_end(stage.name, used_steps)
            return stage_index, last_content, False, None, used_steps
        if self.logger:
            self.logger.log_stage_end(stage.name, used_steps)
        return stage_index + 1, last_content, False, None, used_steps

    def run_simple(self) -> tuple[str, str]:
        # build and run agent without stages 
        agent_end_tool = AgentTool(
            id="agent_end",
            name="agent_end",
            description="End the agent loop with a final answer. Provide the final output and any key reasoning.",
            args=ToolArgs(type="input", description="Final response content."),
            required=False,
        )
        
        # Build memory tools based on agent-level access control
        memory_tools = []
        
        if self.short_term_memory:
            if self.memory_access.get("short_term_save", False):
                short_save_tool = AgentTool(
                    id="short_term_save",
                    name="short_term_save",
                    description="Save data to short-term memory for this agent. Use for temporary context or insights.",
                    args=ToolArgs(type="input", description="Data or insight to save temporarily."),
                    required=False,
                )
                memory_tools.append(short_save_tool)
            
            if self.memory_access.get("short_term_search", False):
                short_search_tool = AgentTool(
                    id="short_term_search",
                    name="short_term_search",
                    description="Search short-term memory for relevant information. Returns matching entries sorted by relevance.",
                    args=ToolArgs(
                        type="object",
                        description="Search parameters",
                        properties={
                            "query": {"type": "string", "description": "Search query to find relevant memory entries"},
                            "limit": {"type": "integer", "description": "Maximum number of results (default: 5)", "default": 5},
                            "score_threshold": {"type": "number", "description": "Minimum similarity score 0-1 (default: 0.6)", "default": 0.6}
                        },
                        required=["query"]
                    ),
                    required=False,
                )
                memory_tools.append(short_search_tool)
        
        if self.long_term_memory:
            if self.memory_access.get("long_term_save", False):
                long_save_tool = AgentTool(
                    id="long_term_save",
                    name="long_term_save",
                    description="Save task and output to long-term memory. Format: task|output",
                    args=ToolArgs(type="input", description="Data or insight to save forever.", data="task|output"),
                    required=False,
                )
                memory_tools.append(long_save_tool)
            
            if self.memory_access.get("long_term_search", False):
                long_search_tool = AgentTool(
                    id="long_term_search",
                    name="long_term_search",
                    description="Search long-term memory for relevant past tasks and outputs. Returns matching entries sorted by relevance.",
                    args=ToolArgs(
                        type="object",
                        description="Search parameters",
                        properties={
                            "query": {"type": "string", "description": "Search query to find relevant memory entries"},
                            "limit": {"type": "integer", "description": "Maximum number of results (default: 5)", "default": 5},
                            "score_threshold": {"type": "number", "description": "Minimum similarity score 0-1 (default: 0.6)", "default": 0.6}
                        },
                        required=["query"]
                    ),
                    required=False,
                )
                memory_tools.append(long_search_tool)
        
        agent_tools = self._convert_tools_to_agent_tools(self.tools) + self._convert_subagents_to_tools() + memory_tools + [agent_end_tool]
        system_prompt = self.system_prompt or self.description or self.prompt
        barebone_model = self.get_barebone(system_prompt, agent_tools)

        messages: List[dict] = [{"role": "user", "content": self.prompt}]
        last_content = ""

        tool_executors = self.build_tool_executors(self.tools, memory_access=self.memory_access, subagents=self.subagents)
        
        for _ in range(self.maxsteps):
            self._enforce_rate_limit()
            step_start = time.time()
            response = chat(
                barebone_model, 
                messages, 
                self.message_history,
                tool_executors=tool_executors,
                logger=self.logger,
                timeout=self.step_timeout
            )
            self.message_history = response.get("message_history", self.message_history)
            last_content = response.get("content", "")
            tool_calls = response.get("tool_calls", []) or []

            _, agent_end_called, agent_end_text = self.parse_control_calls(tool_calls, None)
            if agent_end_called:
                if self.logger:
                    self.logger.log_step(
                        stage_name="simple",
                        step_idx=_,
                        output=last_content,
                        tool_calls=tool_calls,
                        usage=response.get("usage", {}),
                        cost=response.get("cost", {}),
                        elapsed=time.time() - step_start,
                    )
                final = agent_end_text or last_content
                return final, final

            messages = [{"role": "assistant", "content": last_content}]
            if self.logger:
                self.logger.log_step(
                    stage_name="simple",
                    step_idx=_,
                    output=last_content,
                    tool_calls=tool_calls,
                    usage=response.get("usage", {}),
                    cost=response.get("cost", {}),
                    elapsed=time.time() - step_start,
                )
            remaining_after = max(0, self.maxsteps - (_ + 1))
            self._save_agent_checkpoint(remaining_after, last_content)

        return last_content, last_content

    def _build_final_output(self, final_message: str, barebone_model: BareBoneModel) -> Dict[str, str]:
        summary = summarise_message_history(barebone_model, self.message_history) or self.message_history.get("summary", {}).get("message", "")
        if self.logger and summary:
            self.logger.log_summary(summary)
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
            while 0 <= stage_idx < len(self.Stages) and remaining_steps > 0:
                stage_idx, last_content, agent_end_called, end_text, used = self.excute_stage(stage_idx, remaining_steps)
                remaining_steps -= used
                if agent_end_called:
                    agent_end_text = end_text
                    break
            final_message = agent_end_text or last_content
            barebone_model = self.get_barebone(self.message_history["system"]["message"], [])
            return self._build_final_output(final_message, barebone_model)

        if self.next_agent:
            final_message, _ = self.run_simple()
            barebone_model = self.get_barebone(self.system_prompt or self.description or self.prompt, [])
            summary = summarise_message_history(barebone_model, self.message_history) or final_message
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
            barebone_model = self.get_barebone(self.system_prompt or self.description or self.prompt, [])
            return self._build_final_output(final_message, barebone_model)

        original_maxsteps = self.maxsteps
        original_prompt = self.prompt
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            final_message, _ = self.run_simple()
            barebone_model = self.get_barebone(self.system_prompt or self.description or self.prompt, [])
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
                _LOG.warning(f"Final answer validation failed for checks: {failed_check_names}. Retrying (attempt {retry_count}/{max_retries})...")
            else:
                self.maxsteps = original_maxsteps
                self.prompt = original_prompt
                failed_check_names = ", ".join(failed_checks)
                _LOG.warning(f"Final answer validation failed after {max_retries} retries. Returning last output despite failed checks: {failed_check_names}")
                return final_output

        self.maxsteps = original_maxsteps
        self.prompt = original_prompt
        return final_output