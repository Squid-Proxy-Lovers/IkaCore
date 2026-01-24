from __future__ import annotations

import json
from typing import Optional, List, Dict, Callable

from IkaCore.tools import IkaTools
from IkaCore.stages import IkaStage
from IkaModel.base import AgentTool, ToolArgs

from .agent_parse import AgentParseMixin


class AgentToolsMixin(AgentParseMixin):

    def _convert_subagents_to_tools(self, subagents: Optional[List["IkaBaseAgent"]] = None) -> List[AgentTool]:
        """
        Convert subagents to AgentTools, we do this by creating a new AgentTool for each subagent and adding it to the list of agent tools.

        Args:
            subagents: Optional list of subagents to convert to AgentTools

        Returns:
            List of AgentTools
        """

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

    def _convert_tools_to_agent_tools(self, stage_tools: List[IkaTools]) -> List[AgentTool]:
        """
        Convert high level IkaTools to AgentTools, stripping away unnecessary details.
        """
        converted: List[AgentTool] = []
        for tool in stage_tools:
            if isinstance(tool, AgentTool):
                converted.append(tool)
                continue
            
            properties = None
            required_params = []
            if tool.parameters:
                properties = {}
                for param_name, param_value in tool.parameters.items():
                    if isinstance(param_value, dict):
                        param_dict = {k: v for k, v in param_value.items() if k != "required"}
                        properties[param_name] = param_dict
                        if param_value.get("required", False):
                            required_params.append(param_name)
                    elif isinstance(param_value, str):
                        properties[param_name] = {
                            "type": "string",
                            "description": param_value
                        }
                    else:
                        properties[param_name] = {
                            "type": "string",
                            "description": str(param_value)
                        }
            
            if properties and required_params:
                properties["__required__"] = required_params
            
            tool_args = ToolArgs(
                type="object" if properties else "input",
                description=tool.description or "Tool input",
                properties=properties,
            )
            converted.append(
                AgentTool(
                    id=tool.id,
                    name=tool.name,
                    description=tool.description,
                    args=tool_args,
                    required=tool.required,
                    parallel=getattr(tool, 'parallel', True),
                    limit_calls=getattr(tool, 'limit_calls', 0),
                )
            )
        return converted

    def _build_subagent_executor(
        self,
        subagent: "IkaBaseAgent",
        parent_hierarchy: Optional[List[str]] = None,
    ) -> Callable:
        """
        Build a callable executor for a subagent, this is used to execute the subagent's
        execution function (since subagents behave like tools).
        """

        def subagent_executor(args: dict) -> str:
            task_input = args.get("input") or args.get("task") or ""
            if not task_input:
                return json.dumps({"error": "No input provided for subagent"})
            
            try:
                if self.logger:
                    self.logger.log_action(f"Calling subagent: {subagent.name}")
                
                # Store parent hierarchy in subagent for use in get_barebone
                subagent._parent_hierarchy = parent_hierarchy or []
                
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

    def build_tool_executors(
        self,
        tools: List[IkaTools],
        memory_access: Optional[Dict[str, bool]] = None,
        long_term_filter: Optional[Callable] = None,
        subagents: Optional[List["IkaBaseAgent"]] = None,
        parent_hierarchy: Optional[List[str]] = None,
    ) -> Dict[str, Callable]:
        tool_executors: Dict[str, Callable] = {}
        
        def stage_end_executor(args: dict) -> str:
            return "Stage end signal received. Moving to next stage."
        
        def change_stage_executor(args: dict) -> str:
            return "Stage change signal received. Processing stage transition."
        
        control_tools = {"stage_end", "change_stage", "agent_end"}
        
        for tool in tools:
            if isinstance(tool, AgentTool):
                tool_name = tool.name
                if tool_name == "stage_end":
                    tool_executors[tool_name] = stage_end_executor
                elif tool_name == "change_stage":
                    tool_executors[tool_name] = change_stage_executor
                elif tool_name not in control_tools:
                    if hasattr(tool, "execute_function") and tool.execute_function:
                        tool_executors[tool_name] = tool.execute_function
            elif hasattr(tool, "execute_function") and tool.execute_function:
                if tool.name not in control_tools:
                    tool_executors[tool.name] = tool.execute_function
        
        # Add subagent executors
        source_subagents = subagents if subagents is not None else self.subagents
        if source_subagents:
            for subagent in source_subagents:
                subagent_name = getattr(subagent, "name", "subagent")
                tool_executors[subagent_name] = self._build_subagent_executor(subagent, parent_hierarchy)
        
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
                    result = self._search_short_term(query, limit=limit, score_threshold=score_threshold)
                    return json.dumps(result)  # Convert dict to JSON string
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
                    result = self._search_long_term(
                        query,
                        limit=limit,
                        score_threshold=score_threshold,
                        filter_func=long_term_filter,
                    )
                    return json.dumps(result)  # Convert dict to JSON string
                tool_executors["long_term_search"] = long_search_executor
        
        def agent_end_executor(args: dict) -> str:
            content = args.get("input", "")
            
            if not content or content.strip() == "":
                raise ValueError(
                    "agent_end was called with empty arguments. You MUST provide your final answer/output "
                    "in the agent_end tool arguments using the 'input' parameter."
                )
            
            stripped_content = content.strip()
            if stripped_content in ["{}", "[]", "null", '""', "''"]:
                raise ValueError(
                    f"agent_end was called with invalid/empty content: '{stripped_content}'. "
                    "You MUST provide a meaningful final answer, not empty JSON objects, arrays, or null values."
                )
            
            try:
                parsed = json.loads(stripped_content)
                if isinstance(parsed, dict):
                    if len(parsed) == 0:
                        raise ValueError(
                            f"agent_end was called with empty JSON object: '{stripped_content}'. "
                            "You MUST provide a meaningful final answer."
                        )
                    if (
                        len(parsed) == 1
                        and "functions" in parsed
                        and isinstance(parsed["functions"], list)
                        and len(parsed["functions"]) == 0
                    ):
                        pass
                elif isinstance(parsed, list) and len(parsed) == 0:
                    raise ValueError(
                        f"agent_end was called with empty JSON array: '{stripped_content}'. "
                        "You MUST provide a meaningful final answer."
                    )
            except json.JSONDecodeError:
                pass
            
            return "Agent execution ended successfully."

        tool_executors["agent_end"] = agent_end_executor
        
        return tool_executors
