# pyright: strict

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

from IkaCore.tools import IkaTools
from IkaModel.base import AgentTool, ToolArgs

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent

JsonDict = dict[str, Any]


class _SubagentToolConversionState(Protocol):
    subagents: list["IkaBaseAgent"]


class AgentToolSchemaConversionMixin:
    @staticmethod
    def _is_agent_tool(tool: object) -> bool:
        if isinstance(tool, AgentTool):
            return True
        return (
            tool.__class__.__name__ == "AgentTool"
            and hasattr(tool, "id")
            and hasattr(tool, "name")
            and hasattr(tool, "description")
            and hasattr(tool, "args")
        )

    @staticmethod
    def _property_from_parameter(param_value: object) -> JsonDict:
        if isinstance(param_value, Mapping):
            mapped = cast(Mapping[str, Any], param_value)
            return {key: value for key, value in mapped.items() if key != "required"}
        if isinstance(param_value, str):
            return {"type": "string", "description": param_value}
        return {"type": "string", "description": str(param_value)}

    @classmethod
    def _tool_schema_from_parameters(cls, params: object) -> tuple[JsonDict, list[str]]:
        if not params:
            return {}, []
        if not isinstance(params, Mapping):
            return {}, []
        mapped = cast(Mapping[str, Any], params)
        if "properties" in mapped:
            raw_properties = mapped.get("properties", {})
            properties = dict(cast(Mapping[str, Any], raw_properties)) if isinstance(raw_properties, Mapping) else {}
            raw_required = mapped.get("required", [])
            required = [item for item in cast(list[object], raw_required) if isinstance(item, str)] if isinstance(raw_required, list) else []
            return properties, required

        properties: JsonDict = {}
        required_params: list[str] = []
        for param_name, param_value in mapped.items():
            properties[param_name] = cls._property_from_parameter(param_value)
            if isinstance(param_value, Mapping) and cast(Mapping[str, object], param_value).get("required", False):
                required_params.append(param_name)
        return properties, required_params


class AgentSubagentToolConversionMixin(AgentToolSchemaConversionMixin):
    @staticmethod
    def _subagent_tool(subagent: "IkaBaseAgent") -> AgentTool:
        subagent_name = getattr(subagent, "name", "subagent")
        subagent_desc = getattr(subagent, "description", "Subagent")
        tool_args = ToolArgs(
            type="input",
            description=f"The task or request to delegate to the {subagent_name} subagent. Provide a clear, detailed description of what you need the subagent to accomplish.",
        )
        return AgentTool(
            id=subagent_name,
            name=subagent_name,
            description=subagent_desc,
            args=tool_args,
            required=False,
        )

    def _convert_subagents_to_tools(
        self: _SubagentToolConversionState,
        subagents: list["IkaBaseAgent"] | None = None,
    ) -> list[AgentTool]:
        source_subagents = subagents if subagents is not None else self.subagents
        return [AgentSubagentToolConversionMixin._subagent_tool(subagent) for subagent in source_subagents]


class AgentIkaToolConversionMixin(AgentSubagentToolConversionMixin):
    @staticmethod
    def _agent_tool_from_ika_tool(tool: IkaTools, properties: JsonDict) -> AgentTool:
        tool_args = ToolArgs(
            type="object" if properties else "input",
            description=tool.description or "Tool input",
            properties=properties,
        )
        return AgentTool(
            id=tool.id,
            name=tool.name,
            description=tool.description,
            args=tool_args,
            required=tool.required,
            parallel=getattr(tool, "parallel", True),
            limit_calls=getattr(tool, "limit_calls", 0),
        )

    def _convert_tools_to_agent_tools(self, stage_tools: Sequence[IkaTools | AgentTool]) -> list[AgentTool]:
        """
        Convert high level IkaTools to AgentTools, stripping away unnecessary details.
        """
        converted: list[AgentTool] = []
        for tool in stage_tools:
            if self._is_agent_tool(tool):
                converted.append(cast(AgentTool, tool))
                continue

            ika_tool = cast(IkaTools, tool)
            properties, required_params = self._tool_schema_from_parameters(getattr(ika_tool, "parameters", None))
            if required_params:
                properties["__required__"] = required_params
            converted.append(self._agent_tool_from_ika_tool(ika_tool, properties))
        return converted


class AgentToolConversionMixin(AgentIkaToolConversionMixin):
    pass
