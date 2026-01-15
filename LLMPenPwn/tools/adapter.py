from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Iterable

try:
    # Prefer odin if available
    from odin.tools.base import Tool as OdinTool
except Exception:  # pragma: no cover
    OdinTool = object  # type: ignore


def _ensure_agent_cost_callback(agent: Any) -> None:
    """Ensure the given agent has a step-completed callback that at least records/prints costs.

    This is a best-effort fallback when a higher-level system (e.g., BaseCTFSystem)
    hasn't attached its own callback yet. We attach a local callback that prints
    a short cost summary so delegate agents still surface spend information.
    """
    try:
        existing = getattr(agent, "on_step_completed_callback", None)
        if callable(existing):
            return

        def _local_cb(trace: dict):
            try:
                steps = trace.get("steps", [])
                if not steps:
                    return
                last = steps[-1]
                msg = last.get("message") or {}
                tu = msg.get("token_usage") or {}
                input_cost = float(tu.get("input_cost", 0.0) or 0.0)
                output_cost = float(tu.get("output_cost", 0.0) or 0.0)
                step_cost = input_cost + output_cost
                key = getattr(agent, "name", getattr(agent, "__class__", type(agent)).__name__)
                total = getattr(agent, "_local_cost", 0.0) + step_cost
                setattr(agent, "_local_cost", total)
                # Print a concise line so users see spend immediately when using delegate tools
                print(f"[COST][{key}] spent ${total:.6f} (+${step_cost:.6f})")
            except Exception:
                # Best-effort only; don't crash caller
                return

        setattr(agent, "on_step_completed_callback", _local_cb)
    except Exception:
        return


def tool(func: Callable) -> Callable:
    """Lightweight no-op decorator compatible with previous smolagents usage.

    Marks the function as a tool for dynamic wrapping into an Odin Tool.
    """
    setattr(func, "_squid_tool", True)
    return func


def _infer_inputs_from_signature(func: Callable) -> Dict[str, Dict[str, Any]]:
    # Unwrap function if it's been decorated/wrapped
    unwrapped_func = func
    while hasattr(unwrapped_func, '__wrapped__'):
        unwrapped_func = unwrapped_func.__wrapped__
    
    signature = inspect.signature(unwrapped_func)
    inputs: Dict[str, Dict[str, Any]] = {}
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        spec: Dict[str, Any] = {
            "type": "string",
            "description": f"Argument {name}",
        }
        if param.default is not inspect._empty:
            spec["nullable"] = True
        inputs[name] = spec
    # Odin requires non-empty inputs mapping
    return inputs or {"_": {"type": "string", "description": "unused"}}


def function_to_odin_tool(func: Callable, *, name: Optional[str] = None, description: Optional[str] = None) -> OdinTool:
    """Create an Odin Tool instance from a plain function.

    The wrapper maps function parameters to Odin inputs schema and forwards calls.
    """
    if not hasattr(func, "__call__"):
        raise TypeError("func must be callable")

    # Check if this is already a smolagents Tool instance
    try:
        from smolagents.tools import Tool as SmolagentsTool
        is_smolagents_tool = isinstance(func, SmolagentsTool)
    except (ImportError, AttributeError):
        is_smolagents_tool = False
    
    if is_smolagents_tool:
        # Extract name and description from the Tool instance
        tool_name = (name or getattr(func, 'name', None) or "wrapped_tool").strip()
        tool_description = (description or getattr(func, 'description', None) or "").strip()
        # Get the forward method to extract signature
        unwrapped_func = getattr(func, 'forward', func)
        # Try to get original function if forward is wrapped
        while hasattr(unwrapped_func, '__wrapped__'):
            unwrapped_func = unwrapped_func.__wrapped__
        # If forward is a staticmethod, get the underlying function
        if hasattr(unwrapped_func, '__func__'):
            unwrapped_func = unwrapped_func.__func__
        # Use the Tool's inputs if available, otherwise infer from signature
        if hasattr(func, 'inputs') and func.inputs:
            inputs_schema = func.inputs.copy()
        else:
            inputs_schema = _infer_inputs_from_signature(unwrapped_func)
    else:
        # Unwrap function if it's been decorated/wrapped to get the original function
        unwrapped_func = func
        while hasattr(unwrapped_func, '__wrapped__'):
            unwrapped_func = unwrapped_func.__wrapped__
        
        # Get name and description from unwrapped function
        # Try multiple ways to get the function name
        func_name = None
        if name:
            func_name = name
        elif hasattr(unwrapped_func, "__name__"):
            func_name = unwrapped_func.__name__
        elif hasattr(func, "__name__"):
            func_name = func.__name__
        elif hasattr(unwrapped_func, "__qualname__"):
            func_name = unwrapped_func.__qualname__.split('.')[-1]
        elif hasattr(func, "__qualname__"):
            func_name = func.__qualname__.split('.')[-1]
        
        tool_name = (func_name or "wrapped_tool").strip()
        tool_description = (description or (inspect.getdoc(unwrapped_func) or inspect.getdoc(func) or f"Wrapped tool for {tool_name}")).strip()
        inputs_schema = _infer_inputs_from_signature(unwrapped_func)

    # Build a forward(self, <params...>) with a signature matching the wrapped function
    
    sig = inspect.signature(unwrapped_func)
    param_items = [(n, p) for n, p in sig.parameters.items() if n != "self"]

    # If the wrapped function has no params, create a synthetic optional param
    # that will be ignored, and ensure inputs contains the same key
    if not param_items:
        if "_" not in inputs_schema:
            inputs_schema["_"] = {"type": "string", "description": "unused", "nullable": True}
        dummy_param = inspect.Parameter("_", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
        param_items = [("_", dummy_param)]

    # Sort parameters: required (no default) first, then optional (with default)
    def _param_sort_key(item):
        name, param = item
        # Parameters with defaults come after those without
        return (param.default is not inspect._empty, name)
    
    param_items = sorted(param_items, key=_param_sort_key)

    def _format_default(p: inspect.Parameter) -> str:
        if p.default is inspect._empty:
            return ""
        try:
            return f"={repr(p.default)}"
        except Exception:
            return ""

    params_sig = ", ".join([f"{name}{_format_default(p)}" for name, p in param_items])

    ns: Dict[str, Any] = {"_wrapped_func": unwrapped_func}
    if len(sig.parameters) == 0:
        # Original function takes no args; ignore synthetic param when calling
        src = (
            "def forward(self, "
            + params_sig
            + "):\n"
            + "    return _wrapped_func()\n"
        )
    else:
        params_call = ", ".join([name for name, _ in param_items])
        src = (
            "def forward(self, "
            + params_sig
            + "):\n"
            + f"    return _wrapped_func({params_call})\n"
        )
    exec_locals: Dict[str, Any] = {}
    exec(src, ns, exec_locals)
    forward_fn = exec_locals["forward"]

    # Precompute a robust schema from the inferred inputs (avoid relying on base to_schema)
    properties: Dict[str, Any] = {}
    required: list[str] = []
    for in_name, spec in inputs_schema.items():
        t = spec.get("type", "string")
        if isinstance(t, list):
            t = t[0] if t else "string"
        if t == "any":
            t = "string"
        prop: Dict[str, Any] = {"type": t}
        if t == "array":
            prop["items"] = spec.get("items", {"type": "string"})
        if "description" in spec:
            prop["description"] = spec["description"]
        if spec.get("nullable", False):
            pass
        else:
            required.append(in_name)
        if "enum" in spec:
            prop["enum"] = spec["enum"]
        properties[in_name] = prop

    tool_schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
    
    # Validate that tool_name is not the fallback
    if tool_name == "wrapped_tool":
        import warnings
        warnings.warn(f"Tool name extraction failed for function {func}. Using fallback name 'wrapped_tool'. This may cause issues.")

    # Build a concrete to_schema(self) to avoid any descriptor edge cases
    # Store schema in closure to ensure it's accessible
    schema_dict = tool_schema.copy()
    def to_schema_fn(self):
        return schema_dict

    # Dynamically construct a subclass of Odin Tool
    attrs: Dict[str, Any] = {
        "name": tool_name,
        "description": tool_description,
        "inputs": inputs_schema,
        "output_type": "string",
        "forward": forward_fn,
        "to_schema": to_schema_fn,
        # Defensive dict-like methods in case callers mistakenly treat tools as dicts
        "get": lambda self, k, d=None: to_schema_fn(self).get(k, d),
        "keys": lambda self: to_schema_fn(self)["function"].keys() if isinstance(to_schema_fn(self), dict) and "function" in to_schema_fn(self) else [],
        "__getitem__": lambda self, k: to_schema_fn(self)[k],
    }

    ToolCls = type(f"Wrapped_{tool_name}", (OdinTool,), attrs)
    return ToolCls()


def collect_tools_from_module(module) -> list[OdinTool]:
    """Collect all functions marked with @tool in a module and wrap into Odin Tools."""
    tools: list[OdinTool] = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and getattr(obj, "_squid_tool", False):
            tools.append(function_to_odin_tool(obj))
    return tools


# -----------------------------
# Sub-agent delegate adapters
# -----------------------------

def make_delegate_tool(tool_name: str, agent: Any, description: str = "") -> OdinTool:
    """Create a Tool that forwards a JSON string payload to a sub-agent's run().

    - tool_name: tool name exposed to the model (e.g., 'crypto_vulnerability_agent')
    - agent: an Odin MultiStepAgent (duck-typed: must have .run(task) -> result)
    - description: optional description shown to the model
    """
    desc = description or f"Delegate to sub-agent '{tool_name}' with a JSON payload."

    def __init__(self, _agent: Any):
        OdinTool.__init__(self)  # explicit base init
        self._agent = _agent

    def forward(self, payload: str) -> str:  # type: ignore
        result = self._agent.run(payload)
        try:
            return str(getattr(result, "output", result))
        except Exception:
            return str(result)

    attrs: Dict[str, Any] = {
        "name": tool_name,
        "description": desc,
        "inputs": {
            "payload": {
                "type": "string",
                "description": "JSON payload to pass to the sub-agent",
            }
        },
        "output_type": "string",
        "__init__": __init__,
        "forward": forward,
    }

    DelegateTool = type(f"DelegateTool_{tool_name}", (OdinTool,), attrs)
    inst = DelegateTool(agent)
    # Ensure delegate agent surfaces cost updates even if the system hasn't attached a global callback yet
    _ensure_agent_cost_callback(agent)
    return inst


def make_delegate_tools_from_agents(
    agents: Dict[str, Any],
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    description_template: str = "Delegate to sub-agent '{name}'",
) -> list[OdinTool]:
    """Generate delegate tools for a mapping of name->sub-agent.

    - include: names to include (default: all except 'manager_agent')
    - exclude: names to exclude
    """
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else set()

    tools: list[OdinTool] = []
    for name, subagent in agents.items():
        if name == "manager_agent":
            continue
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        desc = description_template.format(name=name)
        tools.append(make_delegate_tool(name=name, agent=subagent, description=desc))
    return tools
