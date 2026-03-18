"""Mixin providing dynamic-tool meta-tool executors."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, TYPE_CHECKING

from .models import DynamicToolDefinition

if TYPE_CHECKING:
    from .execution_env import IkaExecutionEnvironment


class DynamicToolMixin:
    """Executor methods for dynamic-tool meta-tools.

    Mixed into :class:`IkaExecutionEnvironment`.
    """

    # ── make ───────────────────────────────────────────────────────────────

    def _exec_make_tool_call(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        name = args.get("name", "")
        if name in self._dynamic_tools:
            return json.dumps({
                "error": f"Dynamic tool '{name}' already exists. "
                "Use edit_tool_call to modify."
            })

        params_str = args.get("parameters", "{}")
        try:
            params = (
                json.loads(params_str) if isinstance(params_str, str) else params_str
            )
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in parameters"})

        code = args.get("code", "")

        # Validate the code defines execute()
        try:
            compile_dynamic_tool(code)
        except Exception as e:
            return json.dumps({"error": f"Code validation failed: {e}"})

        defn = DynamicToolDefinition(
            name=name,
            description=args.get("description", ""),
            parameters=params,
            code=code,
            created_at=datetime.now(),
        )
        self._dynamic_tools[name] = defn
        return json.dumps({"status": "created", "name": name})

    # ── run ────────────────────────────────────────────────────────────────

    def _exec_run_tool_call(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        name = args.get("name", "")
        if name not in self._dynamic_tools:
            return json.dumps({"error": f"Dynamic tool '{name}' not found"})

        tool_args_str = args.get("args", "{}")
        try:
            tool_args = (
                json.loads(tool_args_str)
                if isinstance(tool_args_str, str)
                else tool_args_str
            )
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in args"})

        defn = self._dynamic_tools[name]
        try:
            executor = compile_dynamic_tool(defn.code)
            result = executor(tool_args)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            return json.dumps({"error": f"Dynamic tool '{name}' failed: {e}"})

    # ── edit ───────────────────────────────────────────────────────────────

    def _exec_edit_tool_call(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        name = args.get("name", "")
        if name not in self._dynamic_tools:
            return json.dumps({"error": f"Dynamic tool '{name}' not found"})

        defn = self._dynamic_tools[name]
        if args.get("description"):
            defn.description = args["description"]
        if args.get("parameters"):
            try:
                defn.parameters = json.loads(args["parameters"])
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON in parameters"})
        if args.get("code"):
            try:
                compile_dynamic_tool(args["code"])
            except Exception as e:
                return json.dumps({"error": f"Code validation failed: {e}"})
            defn.code = args["code"]

        return json.dumps({"status": "updated", "name": name})

    # ── list ───────────────────────────────────────────────────────────────

    def _exec_list_tools(
        self: "IkaExecutionEnvironment", _args: dict
    ) -> str:
        result: List[dict] = []
        for t in self._user_tools:
            result.append({
                "name": t.name,
                "description": t.description,
                "type": "user",
            })
        for name, defn in self._dynamic_tools.items():
            result.append({
                "name": name,
                "description": defn.description,
                "type": "dynamic",
            })
        for n in META_TOOL_NAMES:
            result.append({"name": n, "type": "meta"})
        return json.dumps({"tools": result, "total": len(result)})


# ── Helpers ────────────────────────────────────────────────────────────────

META_TOOL_NAMES = [
    "create_context_pack", "load_context_pack", "unload_context_pack", "delete_context_pack",
    "update_context_pack", "list_context_packs", "search_context_packs",
    "make_subagent", "run_subagent", "edit_subagent", "list_subagents",
    "make_tool_call", "run_tool_call", "edit_tool_call", "list_tools",
    "branch_execution_path", "reset_to_checkpoint", "create_checkpoint",
    "view_checkpoints", "view_execution_history", "check_budget", "end_execution",
]


def compile_dynamic_tool(code: str) -> Callable:
    """Compile dynamic tool code and return the ``execute`` function.

    The code must define ``execute(args: dict) -> str``.
    Runs in the main process — the agent is trusted.
    """
    namespace: Dict[str, Any] = {"json": json, "datetime": datetime}
    exec(code, namespace)  # noqa: S102 — trusted agent code
    fn = namespace.get("execute")
    if fn is None or not callable(fn):
        raise ValueError("Code must define a callable 'execute(args: dict) -> str'")
    return fn
