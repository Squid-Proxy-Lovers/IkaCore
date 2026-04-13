from __future__ import annotations

import json
import inspect
from typing import Optional, List, Dict, Callable
from types import GeneratorType

from IkaCore.tools import IkaTools
from IkaCore.stages import IkaStage
from IkaModel.base import AgentTool, ToolArgs
from IkaCore.runtime_control import RuntimePauseRequested, set_tool_checkpoint_handler

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
            subagent_name = getattr(subagent, 'name', 'subagent')
            subagent_desc = getattr(subagent, "description", "Subagent")
            tool_args = ToolArgs(
                type="input",
                description=f"The task or request to delegate to the {subagent_name} subagent. Provide a clear, detailed description of what you need the subagent to accomplish.",
            )
            agent_tool = AgentTool(
                id=subagent_name,
                name=subagent_name,
                description=subagent_desc,
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
            if isinstance(tool, AgentTool) or (
                hasattr(tool, "args") and hasattr(tool, "name") and hasattr(tool, "description")
            ):
                converted.append(tool)
                continue

            properties = {}
            required_params = []
            params = getattr(tool, "parameters", None)
            if params:
                if isinstance(params, dict) and "properties" in params:
                    properties = params.get("properties", {})
                    required_params = params.get("required", [])
                else:
                    for param_name, param_value in params.items():
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
            
            if required_params:
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
                    side_effect_type=getattr(tool, "side_effect_type", "pure"),
                    replay_policy=getattr(tool, "replay_policy", None),
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
                subagent.checkpoint_store = self.checkpoint_store
                subagent._runtime_run_id = getattr(self, "_runtime_run_id", None)
                subagent._runtime_parent_frame_id = getattr(self, "_runtime_frame_id", None)
                subagent._runtime_return_to_frame_id = getattr(self, "_runtime_frame_id", None)
                subagent._runtime_invocation_type = "subagent_call"
                
                base = getattr(subagent, "_base_prompt", None) or getattr(subagent, "prompt", "")
                full = (base + "\n\n" + task_input).strip() if base else task_input
                subagent.message_history["first_input"]["message"] = full
                subagent.prompt = full
                
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
        stage: Optional[IkaStage] = None,
    ) -> Dict[str, Callable]:
        tool_executors: Dict[str, Callable] = {}
        runtime_tool_metadata: Dict[str, Dict[str, Any]] = {}
        
        def stage_end_executor(args: dict) -> str:
            return "Stage end signal received. Moving to next stage."
        
        def change_stage_executor(args: dict) -> str:
            reason = args.get("reason", "")
            return f"Stage change requested. Reason: {reason}. Processing stage transition."

        def register_runtime_metadata(
            tool_name: str,
            *,
            side_effect_type: str = "pure",
            replay_policy: Optional[str] = None,
            tool_kind: str = "tool",
        ) -> None:
            runtime_tool_metadata[tool_name] = {
                "side_effect_type": side_effect_type,
                "replay_policy": replay_policy or ("allow" if side_effect_type in {"pure", "idempotent"} else "deny"),
                "tool_kind": tool_kind,
            }

        def wrap_runtime_executor(tool_name: str, executor: Callable) -> Callable:
            metadata = runtime_tool_metadata.get(tool_name, {
                "side_effect_type": "pure",
                "replay_policy": "allow",
                "tool_kind": "tool",
            })

            def wrapped(args: dict):
                active_frame_id = getattr(self, "_runtime_active_stage_frame_id", None) or getattr(self, "_runtime_frame_id", None)
                run_id = getattr(self, "_runtime_run_id", None)
                checkpoint_index = 0

                def tool_checkpoint_handler(*, label: str, payload: dict):
                    nonlocal checkpoint_index
                    checkpoint_index += 1
                    checkpoint_kind = f"tool_progress:{tool_name}:{label}"
                    checkpoint_state = {
                        "scope": "tool",
                        "run_id": run_id,
                        "frame_id": active_frame_id,
                        "tool_name": tool_name,
                        "tool_args": args,
                        "tool_progress_index": checkpoint_index,
                        "tool_progress_payload": payload,
                    }
                    snapshot_id = self._runtime_checkpoint(
                        checkpoint_kind,
                        frame_id=active_frame_id,
                        state=checkpoint_state,
                        resume_strategy="exact",
                    )
                    return {"snapshot_id": snapshot_id, "checkpoint_kind": checkpoint_kind}

                if self.checkpoint_store and run_id and active_frame_id:
                    self.checkpoint_store.record_event(
                        run_id,
                        active_frame_id,
                        "tool_started",
                        {
                            "tool_name": tool_name,
                            "tool_args": args,
                            "tool_kind": metadata["tool_kind"],
                            "side_effect_type": metadata["side_effect_type"],
                            "replay_policy": metadata["replay_policy"],
                        },
                    )
                try:
                    set_tool_checkpoint_handler(tool_checkpoint_handler)
                    result = executor(args)
                    if isinstance(result, GeneratorType) or inspect.isgenerator(result):
                        iterator = result
                        last_progress = None
                        while True:
                            try:
                                progress_payload = next(iterator)
                                last_progress = progress_payload
                                tool_checkpoint_handler(
                                    label=f"yield_{checkpoint_index}",
                                    payload={"yield": progress_payload},
                                )
                            except StopIteration as stop:
                                result = stop.value if stop.value is not None else last_progress
                                break
                except RuntimePauseRequested:
                    raise
                except Exception as exc:
                    if self.checkpoint_store and run_id and active_frame_id:
                        self.checkpoint_store.record_event(
                            run_id,
                            active_frame_id,
                            "tool_failed",
                            {
                                "tool_name": tool_name,
                                "tool_args": args,
                                "tool_kind": metadata["tool_kind"],
                                "side_effect_type": metadata["side_effect_type"],
                                "replay_policy": metadata["replay_policy"],
                                "error": str(exc),
                            },
                        )
                    raise
                finally:
                    set_tool_checkpoint_handler(None)
                if self.checkpoint_store and run_id and active_frame_id:
                    if isinstance(result, str):
                        result_payload = result
                    else:
                        try:
                            result_payload = json.dumps(result)
                        except Exception:
                            result_payload = str(result)
                    self.checkpoint_store.record_event(
                        run_id,
                        active_frame_id,
                        "tool_completed",
                        {
                            "tool_name": tool_name,
                            "tool_args": args,
                            "tool_result": result_payload,
                            "tool_kind": metadata["tool_kind"],
                            "side_effect_type": metadata["side_effect_type"],
                            "replay_policy": metadata["replay_policy"],
                        },
                    )
                return result

            return wrapped
        
        control_tools = {"stage_end", "change_stage", "agent_end"}
        
        for tool in tools:
            if isinstance(tool, AgentTool) or (
                hasattr(tool, "args") and hasattr(tool, "name") and hasattr(tool, "description")
            ):
                tool_name = tool.name
                if tool_name == "stage_end":
                    register_runtime_metadata(tool_name, tool_kind="control")
                    tool_executors[tool_name] = wrap_runtime_executor(tool_name, stage_end_executor)
                elif tool_name == "change_stage":
                    register_runtime_metadata(tool_name, tool_kind="control")
                    tool_executors[tool_name] = wrap_runtime_executor(tool_name, change_stage_executor)
                elif tool_name == "ask_user":
                    if stage and getattr(stage, "hitl", False):
                        sn = stage.name
                        register_runtime_metadata("ask_user", side_effect_type="side_effecting", replay_policy="deny", tool_kind="hitl")
                        tool_executors["ask_user"] = wrap_runtime_executor("ask_user", lambda args, sn=sn: self._prompt_hitl_question(sn, args.get("question") or ""))
                elif tool_name not in control_tools:
                    if hasattr(tool, "execute_function") and tool.execute_function:
                        register_runtime_metadata(
                            tool_name,
                            side_effect_type=getattr(tool, "side_effect_type", "pure"),
                            replay_policy=getattr(tool, "replay_policy", None),
                            tool_kind="tool",
                        )
                        tool_executors[tool_name] = wrap_runtime_executor(tool_name, tool.execute_function)
            elif hasattr(tool, "execute_function") and tool.execute_function:
                if tool.name not in control_tools:
                    register_runtime_metadata(
                        tool.name,
                        side_effect_type=getattr(tool, "side_effect_type", "pure"),
                        replay_policy=getattr(tool, "replay_policy", None),
                        tool_kind="tool",
                    )
                    tool_executors[tool.name] = wrap_runtime_executor(tool.name, tool.execute_function)
        
        # Add subagent executors
        source_subagents = subagents if subagents is not None else self.subagents
        if source_subagents:
            for subagent in source_subagents:
                subagent_name = getattr(subagent, "name", "subagent")
                register_runtime_metadata(subagent_name, tool_kind="subagent")
                tool_executors[subagent_name] = wrap_runtime_executor(subagent_name, self._build_subagent_executor(subagent, parent_hierarchy))
        
        # Add memory tool executors based on access control
        effective_access = memory_access or self.memory_access
        if self.short_term_memory:
            if effective_access.get("short_term_save", False):
                def short_save_executor(args: dict) -> str:
                    self._enforce_rate_limit_tool("short_term_save")
                    data = args.get("input") or args.get("data") or ""
                    return self._save_to_short_term(data)
                register_runtime_metadata("short_term_save", side_effect_type="side_effecting", replay_policy="deny", tool_kind="memory")
                tool_executors["short_term_save"] = wrap_runtime_executor("short_term_save", short_save_executor)
            
            if effective_access.get("short_term_search", False):
                def short_search_executor(args: dict) -> str:
                    self._enforce_rate_limit_tool("short_term_search")
                    query = args.get("query", "")
                    limit = args.get("limit", 5)
                    score_threshold = args.get("score_threshold", 0.6)
                    result = self._search_short_term(query, limit=limit, score_threshold=score_threshold)
                    return json.dumps(result)  # Convert dict to JSON string
                register_runtime_metadata("short_term_search", tool_kind="memory")
                tool_executors["short_term_search"] = wrap_runtime_executor("short_term_search", short_search_executor)
        
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
                register_runtime_metadata("long_term_save", side_effect_type="side_effecting", replay_policy="deny", tool_kind="memory")
                tool_executors["long_term_save"] = wrap_runtime_executor("long_term_save", long_save_executor)
            
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
                register_runtime_metadata("long_term_search", tool_kind="memory")
                tool_executors["long_term_search"] = wrap_runtime_executor("long_term_search", long_search_executor)
        
        def agent_end_executor(args: dict) -> str:
            content = args.get("input", "")

            # Normalize boolean inputs that may have been coerced by validate_tool_args
            if isinstance(content, bool):
                content = "true" if content else "false"

            if not content or (isinstance(content, str) and content.strip() == ""):
                raise ValueError(
                    "agent_end was called with empty arguments. You MUST provide your final answer/output "
                    "in the agent_end tool arguments using the 'input' parameter."
                )

            stripped_content = str(content).strip()
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

        register_runtime_metadata("agent_end", tool_kind="control")
        tool_executors["agent_end"] = wrap_runtime_executor("agent_end", agent_end_executor)
        
        return tool_executors
