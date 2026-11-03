# pyright: strict
"""Optional memory tools. Core agents need no memory backend or memory-specific branch."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Dict, List, Literal, Optional

from IkaCore.tools import IkaTools, ToolExecutor, ToolParameters
from IkaMem import LTMemory, STMemory
from IkaModel.base import AgentTool, ToolArgs

from .agent_memory import MemoryOperations

JsonDict = dict[str, Any]
ToolExecutorMap = dict[str, ToolExecutor]
LongTermFilter = Callable[..., Any]




def _string_arg(args: ToolParameters, key: str, default: str='') -> str:
    value = args.get(key, default)
    return value if isinstance(value, str) else str(value)

def _first_string_arg(args: ToolParameters, *keys: str) -> str:
    for key in keys:
        value = args.get(key)
        if value:
            return value if isinstance(value, str) else str(value)
    return ''

def _int_arg(args: ToolParameters, key: str, default: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _float_arg(args: ToolParameters, key: str, default: float) -> float:
    value = args.get(key, default)
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class MemoryTools:
    """Build ordinary IkaTools from explicitly supplied backends and access permissions."""
    def __init__(self, name: str, *, short_term: STMemory | None = None,
                 long_term: LTMemory | None = None,
                 before_call: Callable[[str], None] | None = None):
        self.operations: MemoryOperations = MemoryOperations(name, short_term, long_term)
        self.before_call: Callable[[str], None] = before_call or (lambda name: None)

    def tools(self, *, access: Mapping[str, bool] | None = None,
              long_term_filter: LongTermFilter | None = None) -> list[IkaTools]:
        permissions: dict[str, bool] = dict.fromkeys(('short_term_save', 'short_term_search',
                                    'long_term_save', 'long_term_search'), True)
        if access is not None:
            permissions.update(access)
        definitions = self._build_stage_memory_tools(permissions)
        executors = self._short_term_memory_executors(permissions)
        executors.update(self._long_term_memory_executors(permissions, long_term_filter))
        return [IkaTools(tool.name, tool.description,
                         {"properties": tool.args.properties or {"input": {"type": "string"}},
                          "required": (tool.args.properties or {}).get("__required__", ["input"])},
                         execute_function=executors[tool.name], parallel=False,
                         replay_policy="allow" if tool.name.endswith('_search') else "deny",
                         side_effect_type="read" if tool.name.endswith('_search') else "write")
                for tool in definitions]

    @staticmethod
    def _short_term_save_tool(save_key: str) -> AgentTool:
        return AgentTool(id=save_key, name=save_key, description="Saves information to short-term memory for this agent's current execution session. The data persists only during this agent's runtime and is lost when the agent completes. Use this tool when you need to remember temporary context, intermediate results, or insights that may be useful later in the current task. It should be called when you discover information worth retaining for future steps. The tool returns a confirmation message upon successful save.", args=ToolArgs(type='input', description='The information, context, or insight to store in short-term memory. Be specific and include relevant details.'), required=False)

    @staticmethod
    def _long_term_save_tool(save_key: str) -> AgentTool:
        return AgentTool(id=save_key, name=save_key, description='Saves a structured task result to long-term memory for permanent storage across all agent executions. The data persists beyond the current session and can be accessed by any future agent runs with long-term memory access. Use this tool when you complete a task that produces reusable knowledge or results that should be remembered for future executions. Provide a task description, the output, and optional metadata.', args=ToolArgs(type='object', description='Structured long-term memory entry', properties={'task': {'type': 'string', 'description': 'Task description'}, 'output': {'type': 'string', 'description': 'Task output or result'}, 'metadata': {'type': 'object', 'description': 'Optional metadata for the memory item'}, '__required__': ['task', 'output']}), required=False)

    @staticmethod
    def _memory_search_tool(search_key: str, memory_type: Literal['short_term', 'long_term']) -> AgentTool:
        if memory_type == 'short_term':
            desc = 'Search parameters for querying short-term memory'
            tool_desc = "Searches the agent's short-term memory for information relevant to a given query using semantic similarity. Only searches data saved during the current agent execution session. Returns a list of matching memory entries sorted by relevance score, with the most relevant entries first. Use this tool when you need to recall information you previously saved to short-term memory. The tool will not search long-term memory or any external sources."
            prop_desc = 'The search query string to find relevant memory entries'
        else:
            desc = 'Search parameters for querying long-term memory'
            tool_desc = 'Searches long-term memory for task-output pairs relevant to a given query using semantic similarity. Searches across all permanently stored data from previous agent executions, not just the current session. Returns a list of matching task-output pairs sorted by relevance score, with the most relevant entries first. Use this tool when you need to recall information or results from past executions that might help with the current task. The tool will not search short-term memory or any external sources.'
            prop_desc = 'The search query string to find relevant task-output pairs from past executions'
        return AgentTool(id=search_key, name=search_key, description=tool_desc, args=ToolArgs(type='object', description=desc, properties={'query': {'type': 'string', 'description': prop_desc}, 'limit': {'type': 'integer', 'description': 'Maximum number of results to return (default: 5)', 'default': 5}, 'score_threshold': {'type': 'number', 'description': 'Minimum similarity score from 0.0 to 1.0, where 1.0 is perfect match (default: 0.6)', 'default': 0.6}, '__required__': ['query']}), required=False)

    def _build_memory_tools(self, memory_access: Dict[str, bool], memory_type: Literal['short_term', 'long_term']) -> List[AgentTool]:
        tools: List[AgentTool] = []
        prefix = memory_type
        if memory_type == 'short_term' and (not self.operations.short_term_memory):
            return tools
        if memory_type == 'long_term' and (not self.operations.long_term_memory):
            return tools
        save_key = f'{prefix}_save'
        search_key = f'{prefix}_search'
        if memory_access.get(save_key, False):
            save_tool = self._short_term_save_tool(save_key) if memory_type == 'short_term' else self._long_term_save_tool(save_key)
            tools.append(save_tool)
        if memory_access.get(search_key, False):
            tools.append(self._memory_search_tool(search_key, memory_type))
        return tools

    def _build_stage_memory_tools(self, stage_memory_access: Dict[str, bool]) -> List[AgentTool]:
        tools: List[AgentTool] = []
        tools.extend(self._build_memory_tools(stage_memory_access, 'short_term'))
        tools.extend(self._build_memory_tools(stage_memory_access, 'long_term'))
        return tools

    def _short_term_memory_executors(self, effective_access: dict[str, bool]) -> ToolExecutorMap:
        if not self.operations.short_term_memory:
            return {}
        executors: ToolExecutorMap = {}
        if effective_access.get('short_term_save', False):
    
            def short_save_executor(args: ToolParameters) -> str:
                self.before_call('short_term_save')
                data = _first_string_arg(args, 'input', 'data')
                return self.operations.save_short_term(data)
            executors['short_term_save'] = short_save_executor
        if effective_access.get('short_term_search', False):
    
            def short_search_executor(args: ToolParameters) -> str:
                self.before_call('short_term_search')
                query = _string_arg(args, 'query')
                limit = _int_arg(args, 'limit', 5)
                score_threshold = _float_arg(args, 'score_threshold', 0.6)
                result = self.operations.search_short_term(query, limit=limit, score_threshold=score_threshold)
                return json.dumps(result)
            executors['short_term_search'] = short_search_executor
        return executors

    def _long_term_memory_executors(self, effective_access: dict[str, bool], long_term_filter: Optional[LongTermFilter]) -> ToolExecutorMap:
        if not self.operations.long_term_memory:
            return {}
        executors: ToolExecutorMap = {}
        if effective_access.get('long_term_save', False):
    
            def long_save_executor(args: ToolParameters) -> str:
                self.before_call('long_term_save')
                return self.operations.save_long_term(args)
            executors['long_term_save'] = long_save_executor
        if effective_access.get('long_term_search', False):
    
            def long_search_executor(args: ToolParameters) -> str:
                self.before_call('long_term_search')
                query = _string_arg(args, 'query')
                limit = _int_arg(args, 'limit', 5)
                score_threshold = _float_arg(args, 'score_threshold', 0.6)
                result = self.operations.search_long_term(query, limit=limit, score_threshold=score_threshold, filter_func=long_term_filter)
                return json.dumps(result)
            executors['long_term_search'] = long_search_executor
        return executors


def memory_access_defaults(enabled: bool, overrides: dict[str, bool] | None = None) -> dict[str, bool]:
    access: dict[str, bool] = dict.fromkeys(('short_term_save', 'short_term_search', 'long_term_save', 'long_term_search'), enabled)
    access.update(overrides or {})
    return access
