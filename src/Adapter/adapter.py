from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, Optional, Iterable, List

from IkaCore.tools import IkaTools


def tool(func: Callable) -> Callable:
    setattr(func, "_ika_tool", True)
    return func


def _infer_parameters_from_signature(func: Callable) -> Dict[str, Dict[str, Any]]:
    unwrapped_func = func
    while hasattr(unwrapped_func, '__wrapped__'):
        unwrapped_func = unwrapped_func.__wrapped__
    
    signature = inspect.signature(unwrapped_func)
    parameters: Dict[str, Dict[str, Any]] = {}
    
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        
        param_spec: Dict[str, Any] = {
            "type": "string",
            "description": f"Parameter {name}",
        }
        
        if param.annotation != inspect.Parameter.empty:
            ann_str = str(param.annotation)
            if "int" in ann_str or "float" in ann_str:
                param_spec["type"] = "number"
            elif "bool" in ann_str:
                param_spec["type"] = "boolean"
            elif "list" in ann_str.lower() or "List" in ann_str:
                param_spec["type"] = "array"
                param_spec["items"] = {"type": "string"}
            elif "dict" in ann_str.lower() or "Dict" in ann_str:
                param_spec["type"] = "object"
        
        if param.default != inspect.Parameter.empty:
            param_spec["default"] = param.default
        
        if param_spec.get("default") is None:
            param_spec["required"] = True
        else:
            param_spec["required"] = False
        
        parameters[name] = param_spec
    
    return parameters if parameters else {"input": {"type": "string", "description": "Input parameter", "required": True}}


def function_to_ika_tool(
    func: Callable,
    *,
    tool_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parallel: bool = True,
    required: bool = False
) -> IkaTools:
    """Create an IkaTools instance from a plain Python function.
    
    Args:
        func: The function to convert to a tool
        tool_id: Optional tool ID (defaults to function name)
        name: Optional tool name (defaults to function name)
        description: Optional tool description (defaults to function docstring)
        parallel: Whether the tool can run in parallel (default: True)
        required: Whether the tool is required (default: False)
    
    Returns:
        IkaTools instance ready for use with IkaBaseAgent
    """
    if not callable(func):
        raise TypeError("func must be callable")
    
    unwrapped_func = func
    while hasattr(unwrapped_func, '__wrapped__'):
        unwrapped_func = unwrapped_func.__wrapped__
    
    func_name = name or getattr(unwrapped_func, "__name__", "tool") or getattr(func, "__name__", "tool")
    tool_id = tool_id or func_name
    tool_description = description or (inspect.getdoc(unwrapped_func) or inspect.getdoc(func) or f"Tool: {func_name}")
    
    parameters = _infer_parameters_from_signature(unwrapped_func)
    
    def execute_wrapper(params: dict) -> str:
        try:
            sig = inspect.signature(unwrapped_func)
            param_names = list(sig.parameters.keys())
            
            if not param_names or param_names == ["self"]:
                result = unwrapped_func()
            else:
                kwargs = {}
                for param_name in param_names:
                    if param_name == "self":
                        continue
                    if param_name in params:
                        kwargs[param_name] = params[param_name]
                    elif param_name in parameters and not parameters[param_name].get("required", False):
                        continue
                    else:
                        kwargs[param_name] = params.get(param_name)
                
                result = unwrapped_func(**kwargs)
            
            if isinstance(result, str):
                return result
            elif isinstance(result, (dict, list)):
                return json.dumps(result)
            else:
                return str(result)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return IkaTools(
        id=tool_id,
        name=func_name,
        description=tool_description.strip(),
        parameters=parameters,
        execute_function=execute_wrapper,
        parallel=parallel,
        required=required
    )


def collect_tools_from_module(module) -> List[IkaTools]:
    """Collect all functions marked with @tool in a module and convert to IkaTools.
    
    Args:
        module: Python module to scan for @tool decorated functions
    
    Returns:
        List of IkaTools instances
    """
    tools: List[IkaTools] = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and getattr(obj, "_ika_tool", False):
            if inspect.ismethod(obj):
                func = obj.__func__
            else:
                func = obj
            tools.append(function_to_ika_tool(func))
    return tools


def make_delegate_tool(
    tool_name: str,
    agent: Any,
    description: str = "",
    parallel: bool = False,
    required: bool = False
) -> IkaTools:
    """Create an IkaTools instance that delegates to a sub-agent.
    
    Args:
        tool_name: Name of the tool exposed to the model
        agent: IkaBaseAgent instance (must have .execution() method)
        description: Optional description for the tool
        parallel: Whether the tool can run in parallel (default: False for subagents)
        required: Whether the tool is required (default: False)
    
    Returns:
        IkaTools instance that forwards calls to the sub-agent
    """
    desc = description or f"Delegate to sub-agent '{tool_name}' with a task input."
    
    def delegate_executor(params: dict) -> str:
        task_input = params.get("input") or params.get("task") or params.get("message") or ""
        
        if not task_input:
            return json.dumps({"error": "No input/task provided for sub-agent"})
        
        try:
            if hasattr(agent, "execution"):
                agent.message_history["first_input"]["message"] = task_input
                agent.prompt = task_input
                result = agent.execution()
                final_output = result.get("final_message", result.get("summary", ""))
                return final_output or str(result)
            elif hasattr(agent, "run"):
                result = agent.run(task_input)
                return str(getattr(result, "output", result))
            else:
                return json.dumps({"error": f"Agent {tool_name} does not have execution() or run() method"})
        except Exception as e:
            return json.dumps({"error": f"Error executing sub-agent '{tool_name}': {str(e)}"})
    
    return IkaTools(
        id=tool_name,
        name=tool_name,
        description=desc,
        parameters={
            "input": {
                "type": "string",
                "description": "Task or input to pass to the sub-agent",
                "required": True
            }
        },
        execute_function=delegate_executor,
        parallel=parallel,
        required=required
    )


def make_delegate_tools_from_agents(
    agents: Dict[str, Any],
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    description_template: str = "Delegate to sub-agent '{name}'",
    parallel: bool = False
) -> List[IkaTools]:
    """Generate delegate tools for a mapping of name->sub-agent.
    
    Args:
        agents: Dictionary mapping agent names to IkaBaseAgent instances
        include: Optional list of agent names to include (default: all except 'manager_agent')
        exclude: Optional list of agent names to exclude
        description_template: Template for tool descriptions (uses {name} placeholder)
        parallel: Whether delegate tools can run in parallel (default: False)
    
    Returns:
        List of IkaTools instances for each sub-agent
    """
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else set()
    
    tools: List[IkaTools] = []
    for name, subagent in agents.items():
        if name == "manager_agent":
            continue
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        
        desc = description_template.format(name=name)
        tools.append(make_delegate_tool(
            tool_name=name,
            agent=subagent,
            description=desc,
            parallel=parallel
        ))
    
    return tools
