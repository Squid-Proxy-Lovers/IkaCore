from typing import Optional, List, Dict, Callable
import json
import sys
from pathlib import Path

from tools import SquidTools
from stages import SquidStage
from squidrag import SquidRAGSource
from memory import SquidMemorySystem
from logging_utils import ParaLogger

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
        RAGSource: Optional[List[type[SquidRAGSource]]] = None,
        memory: bool = False,
        memory_finder: Optional["SquidMemorySystem"] = None,
        final_answer_check: Optional[List] = None,
        logging_level: int = 0,
        logging_file: str = "logs.txt",
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
        self.final_answer_check = final_answer_check
        self.message_history = {
            "system": {"message": self.system_prompt or "", "tokens": 0},
            "first_input": {"message": "", "tokens": 0},
            "summary": {"message": "", "tokens": 0},
            "messages": {},
        }

        self.logging_level = logging_level
        self.logging_file = logging_file
        self.logger = ParaLogger(logging_level, logging_file)

        if self.Stages:
            if self.subagents or self.next_agent or self.feedback_agent:
                raise ValueError("Subagents/next/feedback agents are not allowed when stages are defined.")
        else:
            if self.subagents and self.next_agent:
                raise ValueError("Only one of subagents or next_agent may be set when no stages are provided.")

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

    def build_tool_executors(self, tools: List[SquidTools]) -> Dict[str, Callable]:
        tool_executors = {}
        for tool in tools:
            if hasattr(tool, 'execute_function') and tool.execute_function:
                tool_executors[tool.name] = tool.execute_function
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
        return stage_tools + subagent_tools + [agent_end_tool]

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

        tool_executors = self.build_tool_executors(stage.tools)
        
        for _ in range(step_limit): # run till we reach the step limit or we call agent_end
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
        agent_tools = self._convert_tools_to_agent_tools(self.tools) + self._convert_subagents_to_tools() + [agent_end_tool]
        system_prompt = self.system_prompt or self.description or self.prompt
        barebone_model = self.get_barebone(system_prompt, agent_tools)

        messages: List[dict] = [{"role": "user", "content": self.prompt}]
        last_content = ""

        tool_executors = self.build_tool_executors(self.tools)
        
        for _ in range(self.maxsteps):
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
                final = agent_end_text or last_content
                return final, final

            messages = [{"role": "assistant", "content": last_content}]

        return last_content, last_content

    def _build_final_output(self, final_message: str, barebone_model: BareBoneModel) -> Dict[str, str]:
        summary = summarise_message_history(barebone_model, self.message_history) or self.message_history.get("summary", {}).get("message", "")
        if self.logger and summary:
            self.logger.log_summary(summary)
        return {"final_message": final_message, "summary": summary}

    def execution(self) -> Dict[str, str]:
        last_content = ""

        if self.Stages:
            stage_idx = 0
            agent_end_text = None
            remaining_steps = self.maxsteps
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

        final_message, _ = self.run_simple()
        barebone_model = self.get_barebone(self.system_prompt or self.description or self.prompt, [])
        return self._build_final_output(final_message, barebone_model)