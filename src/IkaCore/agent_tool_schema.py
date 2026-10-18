from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple

from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools
from IkaModel.base import AgentTool, ToolArgs

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent


class _AgentToolListBuilderState(Protocol):
    Stages: List[IkaStage]
    tools: List[IkaTools]

    def _convert_tools_to_agent_tools(self, stage_tools: Sequence[IkaTools | AgentTool]) -> List[AgentTool]:
        ...

    def _convert_subagents_to_tools(self, subagents: Optional[List["IkaBaseAgent"]] = None) -> List[AgentTool]:
        ...

    def _stage_agent_end_tool(self) -> AgentTool:
        ...

    def _simple_agent_end_tool(self, submit_tools: List[str]) -> AgentTool:
        ...


class AgentToolListBuilderMixin:
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

        return stage_tools + subagent_tools

    def build_simple_tools(self: _AgentToolListBuilderState) -> List[AgentTool]:
        submit_tools = [
            getattr(t, "name", "")
            for t in (self.tools or [])
            if getattr(t, "name", "").startswith("submit_")
        ]
        return (
            self._convert_tools_to_agent_tools(self.tools)
            + self._convert_subagents_to_tools()
            + [self._simple_agent_end_tool(submit_tools)]
        )
