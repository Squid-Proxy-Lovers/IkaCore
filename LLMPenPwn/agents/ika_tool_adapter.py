#!/usr/bin/env python3
"""
Adapter to convert odin @tool decorated functions to IkaTools.
"""

import sys
import inspect
from pathlib import Path
from typing import Callable, Dict, Any, Optional

# Add Para-Core src to path for IkaCore
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from IkaCore.tools import IkaTools


def _infer_parameters_from_signature(func: Callable) -> Dict[str, Dict[str, Any]]:
    """Infer IkaTools parameters schema from function signature."""
    unwrapped_func = func
    while hasattr(unwrapped_func, '__wrapped__'):
        unwrapped_func = unwrapped_func.__wrapped__
    
    signature = inspect.signature(unwrapped_func)
    parameters: Dict[str, Dict[str, Any]] = {}
    required: list[str] = []
    
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        
        param_type = "string"
        if param.annotation != inspect.Parameter.empty:
            ann_str = str(param.annotation)
            if "int" in ann_str or "float" in ann_str:
                param_type = "number"
            elif "bool" in ann_str:
                param_type = "boolean"
            elif "list" in ann_str.lower() or "List" in ann_str:
                param_type = "array"
        
        param_spec: Dict[str, Any] = {
            "type": param_type,
            "description": f"Argument {name}",
        }
        
        if param.default is not inspect._empty:
            param_spec["default"] = param.default
        else:
            required.append(name)
        
        parameters[name] = param_spec
    
    return parameters, required


def function_to_ika_tool(func: Callable, *, name: Optional[str] = None, description: Optional[str] = None) -> IkaTools:
    """Convert a @tool decorated function to IkaTools."""
    if not callable(func):
        raise TypeError("func must be callable")
    
    unwrapped_func = func
    while hasattr(unwrapped_func, '__wrapped__'):
        unwrapped_func = unwrapped_func.__wrapped__
    
    tool_name = name or unwrapped_func.__name__
    tool_description = description or inspect.getdoc(unwrapped_func) or f"Tool: {tool_name}"
    
    parameters, required_list = _infer_parameters_from_signature(unwrapped_func)
    
    def execute_wrapper(params: dict) -> str:
        try:
            result = unwrapped_func(**params)
            if result is None:
                return ""
            return str(result)
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    return IkaTools(
        id=tool_name,
        name=tool_name,
        description=tool_description,
        parameters=parameters,
        required=True,
        execute_function=execute_wrapper
    )


def collect_tools_from_module(module) -> list[IkaTools]:
    """Collect all functions marked with @tool in a module and convert to IkaTools."""
    tools: list[IkaTools] = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and getattr(obj, "_squid_tool", False):
            tools.append(function_to_ika_tool(obj))
    return tools
