import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from .model import Message, MessageRole, Model, ToolCall
from .prompts import ODIN_GENERAL_PROMPT
from .tools import Tool

_LOG = logging.getLogger(__name__)


class AgentError(Exception):
    pass

@dataclass
class ToolOutput:
    id: str
    output: Any
    observation: str
    tool_call: ToolCall

@dataclass
class Timestamps:
    step_started: float = 0
    model_started: float = 0
    model_completed: float = 0
    tool_calls_started: float = 0
    tool_calls_completed: float = 0
    step_completed: float = 0

@dataclass
class AgentStep:
    step_number: int
    message: Optional[Message] = None
    tool_results: Optional[list[ToolOutput]] = None
    error: Optional[str] = None
    timestamps: Optional[Timestamps] = None

@dataclass
class AgentTrace:
    system_prompt: str
    model_id: str
    tools: list[dict[str, Any]]
    steps: list[AgentStep]

@dataclass
class AgentResult:
    output: Any
    steps: list[AgentStep]
    success: bool
    error: Optional[str] = None

class Agent(ABC):

    def __init__(
        self,
        model: Model,
        tools: Optional[list[Tool]] = None,
        max_steps: int = 10,
        system_prompt: Optional[str] = None,
        on_step_completed_callback: Callable = None,
        verbose: bool = False,
    ):
        self.model = model
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_steps = max_steps
        self.system_prompt = system_prompt or self._default_system_prompt()

        if self.tools:
            tool_guidelines = "\n\n".join(
                f"{tool.tool_guidelines}" for tool in self.tools.values() if tool.tool_guidelines
            )
            if tool_guidelines:
                self.system_prompt += "\n\n# Tool Guidelines\n\n" + tool_guidelines

        self.on_step_completed_callback = on_step_completed_callback
        self.verbose = verbose
        # human-readable agent name (defaults to class name)
        self.name = self.__class__.__name__
        self.steps: list[AgentStep] = []
        self.messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=self.system_prompt)
        ]

        _LOG.info("Initialized %s with %d tools, max_steps=%d", 
                 self.__class__.__name__, len(self.tools), max_steps)

    def get_trace(self) -> AgentTrace:
        return AgentTrace(
            system_prompt=self.system_prompt,
            model_id=self.model.model_id,
            tools=[tool.to_json() for tool in self.tools.values()],
            steps=self.steps
        )

    @abstractmethod
    def _default_system_prompt(self) -> str:
        pass

    @abstractmethod
    def run(self, task: str, **kwargs) -> AgentResult:
        pass


class MultiStepAgent(Agent):
    def _default_system_prompt(self) -> str:
        return ODIN_GENERAL_PROMPT

    def run(self, task: str, **kwargs) -> AgentResult:
        if self.verbose:
            print(f"Task: {task}")
            _LOG.info("Agent %s Task: %s", self.name, task)

        self.steps = []
        self.messages = [Message(role=MessageRole.SYSTEM, content=self.system_prompt)]

        user_message = Message(role=MessageRole.USER, content=task)
        self.messages.append(user_message)
        
        self.steps.append(AgentStep(step_number=0, message=user_message))

        for step_offset in range(1, self.max_steps + 1):
            step_num = step_offset
            _LOG.debug("Executing step %d", step_num)

            step = AgentStep(step_number=step_num, timestamps=Timestamps(step_started=time.time()))

            try:
                if self.verbose:
                    print(f"Step {step_num}: Making model request...")
                    _LOG.info("Agent %s Step %d: Making model request...", self.name, step_num)

                step.timestamps.model_started = time.time()
                response = self.model.generate(
                    messages=self.messages,
                    tools=list(self.tools.values()) if self.tools else None
                )

                step.timestamps.model_completed = time.time()
                step.message = response
                self.messages.append(response)

                if response.tool_calls:
                    tool_results = {}

                    step.timestamps.tool_calls_started = time.time()
                    for tool_call in response.tool_calls:
                        if self.verbose:
                            tool_call_json = json.dumps({
                                "name": tool_call.name,
                                "arguments": tool_call.arguments
                            }, indent=2)
                            print(f"Tool call: {tool_call_json}")
                            _LOG.info("Agent %s Tool call: %s", self.name, tool_call_json)

                        result = self._execute_tool(tool_call)
                        tool_results[tool_call.id] = result

                        response_str = str(result.output) if result.output is not None else str(result.observation) if result.observation else "No output"
                        if self.verbose:
                            print(f"Tool response: {response_str}")
                        _LOG.info("Agent %s Tool response: %s", self.name, response_str)

                        tool_content = str(result.output) if result.output is not None else (str(result.observation) if result.observation else "")
                        self.messages.append(Message(
                            role=MessageRole.TOOL,
                            content=tool_content,
                            tool_call_id=tool_call.id
                        ))

                    step.timestamps.tool_calls_completed = time.time()
                    step.tool_results = tool_results
                    _LOG.debug("Executed %d tool calls in step %d", 
                             len(response.tool_calls), step_num)

                step.timestamps.step_completed = time.time()
                self.steps.append(step)

                if self.on_step_completed_callback:
                    self.on_step_completed_callback(asdict(self.get_trace()))

                if not response.tool_calls and response.content:
                    if self.verbose:
                        print(f"Model response (no tool calls): {response.content[:200]}...")
                    _LOG.info("Task completed in %d steps", step_num)
                    return AgentResult(
                        output=response.content,
                        steps=self.steps,
                        success=True
                    )
                
                if not response.tool_calls and not response.content:
                    _LOG.warning("Step %d: Model returned no tool calls and no content", step_num)
                    if self.verbose:
                        print(f"WARNING: Step {step_num} returned no tool calls and no content")

            except Exception as e:
                error_msg = f"Error in step {step_num}: {str(e)}"
                step.error = error_msg
                self.steps.append(step)
                _LOG.error(error_msg)

                return AgentResult(
                    output=None,
                    steps=self.steps,
                    success=False,
                    error=error_msg
                )

        error_msg = f"Reached maximum steps ({self.max_steps}) without completion"
        _LOG.warning(error_msg)

        return AgentResult(
            output=self.messages[-1].content if self.messages else None,
            steps=self.steps,
            success=False,
            error=error_msg
        )

    def _execute_tool(self, tool_call: ToolCall) -> ToolOutput:
        _LOG.debug("Executing tool call: %s with arguments %s", 
                   tool_call.name, tool_call.arguments)

        tool = self.tools.get(tool_call.name)
        if not tool:
            raise ValueError(f"Tool '{tool_call.name}' not found")

        try:
            result = tool.forward(**tool_call.arguments)
            _LOG.debug("Tool '%s' executed successfully", tool_call.name)
            return ToolOutput(
                id=tool_call.id,
                output=result,
                observation="",
                tool_call=tool_call
            )
        except Exception as e:
            error_msg = f"Error executing tool '{tool_call.name}': {str(e)}"
            _LOG.error(error_msg)
            return ToolOutput(
                id=tool_call.id,
                output=None,
                observation=error_msg,
                tool_call=tool_call
            )