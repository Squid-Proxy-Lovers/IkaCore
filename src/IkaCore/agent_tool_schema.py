from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Protocol, Tuple

from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools
from IkaModel.base import AgentTool, ToolArgs

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent


class _MemoryToolSchemaState(Protocol):
    short_term_memory: object | None
    long_term_memory: object | None

    def _short_term_save_tool(self, save_key: str) -> AgentTool:
        ...

    def _long_term_save_tool(self, save_key: str) -> AgentTool:
        ...

    def _memory_search_tool(self, search_key: str, memory_type: Literal["short_term", "long_term"]) -> AgentTool:
        ...

    def _build_memory_tools(
        self,
        memory_access: Dict[str, bool],
        memory_type: Literal["short_term", "long_term"],
    ) -> List[AgentTool]:
        ...


class _AgentToolListBuilderState(_MemoryToolSchemaState, Protocol):
    Stages: List[IkaStage]
    tools: List[IkaTools]
    memory_access: Dict[str, bool]

    def _convert_tools_to_agent_tools(self, stage_tools: Sequence[IkaTools | AgentTool]) -> List[AgentTool]:
        ...

    def _convert_subagents_to_tools(self, subagents: Optional[List["IkaBaseAgent"]] = None) -> List[AgentTool]:
        ...

    def _build_stage_memory_tools(self, stage_memory_access: Dict[str, bool]) -> List[AgentTool]:
        ...

    def _stage_agent_end_tool(self) -> AgentTool:
        ...

    def _simple_agent_end_tool(self, submit_tools: List[str]) -> AgentTool:
        ...


class MemoryToolDefinitionMixin:
    @staticmethod
    def _short_term_save_tool(save_key: str) -> AgentTool:
        return AgentTool(
            id=save_key,
            name=save_key,
            description="Saves information to short-term memory for this agent's current execution session. The data persists only during this agent's runtime and is lost when the agent completes. Use this tool when you need to remember temporary context, intermediate results, or insights that may be useful later in the current task. It should be called when you discover information worth retaining for future steps. The tool returns a confirmation message upon successful save.",
            args=ToolArgs(type="input", description="The information, context, or insight to store in short-term memory. Be specific and include relevant details."),
            required=False,
        )

    @staticmethod
    def _long_term_save_tool(save_key: str) -> AgentTool:
        return AgentTool(
            id=save_key,
            name=save_key,
            description="Saves a structured task result to long-term memory for permanent storage across all agent executions. The data persists beyond the current session and can be accessed by any future agent runs with long-term memory access. Use this tool when you complete a task that produces reusable knowledge or results that should be remembered for future executions. Provide a task description, the output, and optional metadata.",
            args=ToolArgs(
                type="object",
                description="Structured long-term memory entry",
                properties={
                    "task": {"type": "string", "description": "Task description"},
                    "output": {"type": "string", "description": "Task output or result"},
                    "metadata": {"type": "object", "description": "Optional metadata for the memory item"},
                    "__required__": ["task", "output"],
                },
            ),
            required=False,
        )

    @staticmethod
    def _memory_search_tool(search_key: str, memory_type: Literal["short_term", "long_term"]) -> AgentTool:
        if memory_type == "short_term":
            desc = "Search parameters for querying short-term memory"
            tool_desc = "Searches the agent's short-term memory for information relevant to a given query using semantic similarity. Only searches data saved during the current agent execution session. Returns a list of matching memory entries sorted by relevance score, with the most relevant entries first. Use this tool when you need to recall information you previously saved to short-term memory. The tool will not search long-term memory or any external sources."
            prop_desc = "The search query string to find relevant memory entries"
        else:
            desc = "Search parameters for querying long-term memory"
            tool_desc = "Searches long-term memory for task-output pairs relevant to a given query using semantic similarity. Searches across all permanently stored data from previous agent executions, not just the current session. Returns a list of matching task-output pairs sorted by relevance score, with the most relevant entries first. Use this tool when you need to recall information or results from past executions that might help with the current task. The tool will not search short-term memory or any external sources."
            prop_desc = "The search query string to find relevant task-output pairs from past executions"

        return AgentTool(
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
                    "__required__": ["query"],
                },
            ),
            required=False,
        )


class MemoryToolSchemaMixin(MemoryToolDefinitionMixin):
    def _build_memory_tools(
        self: _MemoryToolSchemaState,
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
            save_tool = self._short_term_save_tool(save_key) if memory_type == "short_term" else self._long_term_save_tool(save_key)
            tools.append(save_tool)
        if memory_access.get(search_key, False):
            tools.append(self._memory_search_tool(search_key, memory_type))
        return tools

    def _build_stage_memory_tools(self: _MemoryToolSchemaState, stage_memory_access: Dict[str, bool]) -> List[AgentTool]:
        tools: List[AgentTool] = []
        tools.extend(self._build_memory_tools(stage_memory_access, "short_term"))
        tools.extend(self._build_memory_tools(stage_memory_access, "long_term"))
        return tools


class AgentToolListBuilderMixin(MemoryToolSchemaMixin):
    @staticmethod
    def _stage_agent_end_tool() -> AgentTool:
        return AgentTool(
            id="agent_end",
            name="agent_end",
            description="Terminates the agent execution and returns the final answer to the user or calling system. This tool must be called when you have completed the task specified in your initial prompt. The final answer should be comprehensive, addressing all requirements from the original task. It should be based on your initial prompt and any context you have gathered throughout execution. This tool will immediately end the agent loop, so ensure your answer is complete before calling it. The tool can only be called once per execution.",
            args=ToolArgs(type="input", description="Your complete final answer addressing the original task. This parameter is required and cannot be empty."),
            required=True,
            limit_calls=1,
        )

    @staticmethod
    def _simple_agent_end_config(submit_tools: List[str]) -> Tuple[str, bool]:
        if not submit_tools:
            return (
                "Terminates the agent execution and returns the final answer to the user or calling system. This tool must be called when you have completed the task specified in your initial prompt. The final answer should be comprehensive, addressing all requirements from the original task. It should be based on your initial prompt and any context you have gathered throughout execution. This tool will immediately end the agent loop, so ensure your answer is complete before calling it. The tool can only be called once per execution.",
                True,
            )
        return (
            "Terminates the agent execution and returns the final answer to the "
            "calling system. This tool is only valid after the task-specific "
            f"structured submission tool has succeeded. Available structured "
            f"submission tools in this task: {', '.join(submit_tools)}. If you "
            "have not called the correct submit_* tool yet, do that before "
            "agent_end. The final answer should be a short summary of the "
            "structured submission.",
            False,
        )

    @classmethod
    def _simple_agent_end_tool(cls, submit_tools: List[str]) -> AgentTool:
        agent_end_description, agent_end_required = cls._simple_agent_end_config(submit_tools)
        return AgentTool(
            id="agent_end",
            name="agent_end",
            description=agent_end_description,
            args=ToolArgs(type="input", description="Your complete final answer addressing the original task. This parameter is required and cannot be empty."),
            required=agent_end_required,
            limit_calls=1,
        )

    def build_stage(self: _AgentToolListBuilderState, stage: IkaStage) -> List[AgentTool]:
        stage_tools = self._convert_tools_to_agent_tools(stage.tools)
        subagent_tools = self._convert_subagents_to_tools(getattr(stage, "subagents", None))

        if stage == self.Stages[-1]:
            stage_tools.append(self._stage_agent_end_tool())
            stage_end_index = next((i for i, t in enumerate(stage_tools) if t.name == "stage_end"), None)
            if stage_end_index is not None:
                stage_tools.pop(stage_end_index)

        stage_memory_access = getattr(stage, "memory_access", None) or self.memory_access
        memory_tools = self._build_stage_memory_tools(stage_memory_access)
        return stage_tools + subagent_tools + memory_tools

    def build_simple_tools(self: _AgentToolListBuilderState) -> List[AgentTool]:
        submit_tools = [
            getattr(t, "name", "")
            for t in (self.tools or [])
            if getattr(t, "name", "").startswith("submit_")
        ]
        memory_tools = self._build_stage_memory_tools(self.memory_access)
        return (
            self._convert_tools_to_agent_tools(self.tools)
            + self._convert_subagents_to_tools()
            + memory_tools
            + [self._simple_agent_end_tool(submit_tools)]
        )
