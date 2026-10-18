from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from IkaCore.events import OutputType
from IkaCore.events import current_output as get_cli_output

_LOG = logging.getLogger(__name__)



class AgentEndException(RuntimeError):
    """Raised when agent_end (or equivalent) signals immediate completion."""

    def __init__(self, response: Optional[dict[str, Any]] = None, final_text: Optional[str] = None) -> None:
        super().__init__("agent_end requested termination")
        self.response: dict[str, Any] = response or {}
        response_content = self.response.get("content")
        self.final_text: str | None = final_text or (response_content if isinstance(response_content, str) else None)


class HumanInputRequired(RuntimeError):
    """Raised when HITL execution must pause for human input."""

    def __init__(self, payload: Optional[dict[str, Any]] = None) -> None:
        super().__init__("human input required")
        self.payload: dict[str, Any] = payload or {}


DEFAULT_SYSTEM_PROMPT = """Answer the assigned task clearly and concisely. When finished, call agent_end(input=...) with your complete answer."""

@dataclass(init=False)
class ToolArgs:
    type: str
    description: str
    agent: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[dict[str, Any]] = None
    properties: Optional[dict[str, Any]] = None

    def __init__(
        self, type: str, description: str, agent: Optional[str] = None,
        data: Optional[Any] = None, metadata: Optional[dict[str, Any]] = None,
        properties: Optional[dict[str, Any]] = None, *, required: Optional[list[str]] = None,
    ) -> None:
        self.type, self.description, self.agent = type, description, agent
        self.data, self.metadata = data, metadata
        self.properties = properties if required is None else {**(properties or {}), "__required__": list(required)}

    # we are going to assume that all args are required


@dataclass
class AgentTool:
    def __init__(self, id:str, name: str, description: str, args: ToolArgs, required: bool = False, parallel: bool = False, limit_calls: int = 0) -> None:
        self.validate(name)
        self.id: str = id
        self.name: str = name
        self.description: str = description
        self.args: ToolArgs = args
        self.required: bool = required
        self.parallel: bool = parallel # assume that all tools are not parallel by default
        self.limit_calls: int = limit_calls


    @staticmethod
    def validate(name: str) -> None:
        if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', name):
            # we need for deepseek
            raise ValueError("Name must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.")


@dataclass
class BareBoneModel:
    _on_compaction: Callable[[], None] | None
    _tool_call_counts: dict[str, int]
    context_budget: int | None
    def __init__(
        self,
        model_id: str,
        api_key: str,
        api_url: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        content_prompt: str = "",
        max_tokens: int = 20000,
        temperature: float = 0,
        parallel_tool_calls: bool = False,
        agent_name: Optional[str] = None,
        agent_hierarchy: Optional[List[str]] = None,
        suppress_init_output: bool = True,
        reasoning_effort: Optional[str] = None,
        use_responses_api: bool = True,
        force_control_tool_on_max_tool_calls: bool = True,
        thinking_budget_tokens: Optional[int] = None,
        ) -> None:

        # User MUST provide the following:
        if not api_key:
            raise ValueError("api_key is required for BareBoneModel")
        if not model_id:
            raise ValueError("model_id is required for BareBoneModel")
        if not api_url:
            raise ValueError("api_url is required for BareBoneModel")

        self.model_id: str = model_id
        self.api_key: str = api_key
        self.api_url: str = api_url
        self.system_prompt: str = system_prompt
        self.content_prompt: str = content_prompt
        self.max_tokens: int = max_tokens
        if temperature < 0 or temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        self.temperature: float = temperature
        self.agent_tools: list[AgentTool] = []
        self.deepthinking: bool = False
        self.parallel_tool_calls: bool = parallel_tool_calls
        self.agent_name: Optional[str] = agent_name
        self.agent_hierarchy: Optional[List[str]] = agent_hierarchy or []
        self.reasoning_effort: Optional[str] = reasoning_effort
        self.thinking_budget_tokens: Optional[int] = thinking_budget_tokens
        self.use_responses_api: bool = use_responses_api
        self.force_control_tool_on_max_tool_calls: bool = force_control_tool_on_max_tool_calls
        self.forced_tool_name: Optional[str] = None
        self._current_step: int = 0

        # Display prompts using CLI output (unless suppressed)
        if not suppress_init_output:
            cli = get_cli_output()
            hierarchy = self.agent_hierarchy or [self.agent_name or "Agent"]
            agent_name = hierarchy[0] if hierarchy else "Agent"
            step = cli.get_step(agent_name) if agent_name else 0

            if self.system_prompt:
                cli.emit(
                    OutputType.AGENT_INIT,
                    f"System Prompt:\n{self.system_prompt}",
                    hierarchy,
                    step=step
                )

            if self.content_prompt:
                cli.emit(
                    OutputType.AGENT_INIT,
                    f"Content Prompt:\n{self.content_prompt}",
                    hierarchy,
                    step=step
                )


_SUMMARY_PROMPT_PATH = Path(__file__).parent / "summary_prompt"
try:
    from .resource_loader import read_embedded_resource as _read_ika_resource

    SUMMARY_PROMPT = _read_ika_resource("IkaCore/src/IkaModel/summary_prompt")
except (ImportError, ModuleNotFoundError, OSError):
    try:
        with open(_SUMMARY_PROMPT_PATH, "r", encoding="utf-8") as _prompt_file:
            SUMMARY_PROMPT = _prompt_file.read()
    except FileNotFoundError:
        SUMMARY_PROMPT = "Please summarize the following conversation history concisely, preserving key information and context."
