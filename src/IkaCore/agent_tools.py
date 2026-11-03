# pyright: strict

from __future__ import annotations

from .agent_parse import AgentParseMixin
from .agent_tool_conversion import AgentToolConversionMixin
from .agent_tool_execution import AgentToolExecutorMixin
from .agent_tool_schema import AgentToolListBuilderMixin


class AgentToolsMixin(AgentToolConversionMixin, AgentToolExecutorMixin, AgentToolListBuilderMixin, AgentParseMixin):
    pass
