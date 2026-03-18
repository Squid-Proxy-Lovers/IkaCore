"""Mixin that builds the full set of meta-tools and their executors."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, TYPE_CHECKING

from IkaCore.tools import IkaTools

if TYPE_CHECKING:
    from .execution_env import IkaExecutionEnvironment


class MetaToolRegistryMixin:
    """Builds all meta-tool :class:`IkaTools` and their executor map.

    Mixed into :class:`IkaExecutionEnvironment`.
    """

    def _build_meta_tools(
        self: "IkaExecutionEnvironment",
    ) -> Tuple[List[IkaTools], Dict[str, Callable]]:
        """Return ``(tools, executors)`` for every meta-tool."""
        tools: List[IkaTools] = []
        executors: Dict[str, Callable] = {}

        def _reg(
            name: str,
            description: str,
            parameters: dict,
            executor: Callable,
            *,
            parallel: bool = True,
        ) -> None:
            tools.append(IkaTools(
                name=name,
                description=description,
                parameters=parameters,
                execute_function=executor,
                parallel=parallel,
            ))
            executors[name] = executor

        # ── Context Pack Tools ─────────────────────────────────────────────

        _reg(
            "create_context_pack",
            "Create a named context pack in the fixed CCP shelf, optionally choosing the book. If the pack already exists, this tool replaces it by deleting the old entry and creating a new one. Root plan packs in the `plans` book must not be rewritten with materially identical content during the same execution.",
            _params(
                {"name": "Unique name for the pack", "book": "Book to store the pack in (optional)",
                 "content": "Content to store",
                 "tags": "Comma-separated tags (optional)"},
                required=["name", "content"],
            ),
            self._exec_create_context_pack,
        )
        _reg(
            "load_context_pack",
            "Load a context pack from the fixed shelf into the active context window.",
            _params(
                {"name": "Name of the context pack to load",
                 "book": "Book containing the pack (optional)",
                 "chunk": ("integer", "Chunk index for large packs (0-indexed, optional)")},
                required=["name"],
            ),
            self._exec_load_context_pack,
        )
        _reg(
            "unload_context_pack",
            "Mark a context pack as unloaded.",
            _params({"name": "Name of the pack to unload", "book": "Book containing the pack (optional)"}, required=["name"]),
            self._exec_unload_context_pack,
        )
        _reg(
            "delete_context_pack",
            "Delete a context pack from the fixed shelf. Prefer deleting and then re-creating a pack when you need to rewrite it substantially.",
            _params({"name": "Name of the pack to delete", "book": "Book containing the pack (optional)"}, required=["name"]),
            self._exec_delete_context_pack,
        )
        _reg(
            "update_context_pack",
            "Legacy edit helper for an existing context pack. This is implemented by deleting the old entry and creating a replacement entry. Prefer delete_context_pack plus create_context_pack for explicit rewrites.",
            _params(
                {"name": "Name of the pack", "book": "Book containing the pack (optional)",
                 "content": "New content (optional)",
                 "tags": "New comma-separated tags (optional)"},
                required=["name"],
            ),
            self._exec_update_context_pack,
        )
        _reg(
            "list_context_packs",
            "List context packs in the fixed shelf, optionally filtering by book.",
            _params({"book": "Only list packs from this book (optional)"}),
            self._exec_list_context_packs,
        )
        _reg(
            "search_context_packs",
            "Search context packs in the fixed shelf and return matching pack names and metadata.",
            _params({"query": "Search query for context pack names, tags, or content", "book": "Restrict search to this book (optional)"}, required=["query"]),
            self._exec_search_context_packs,
        )

        # ── Sub-Agent Tools ────────────────────────────────────────────────

        _reg(
            "make_subagent",
            "Create a sub-agent definition with its own execution environment.",
            _params(
                {"name": "Unique sub-agent name",
                 "description": "System prompt / role description",
                 "prompt": "Initial task prompt",
                 "tools": "Comma-separated user tool names (optional)",
                 "model": "Model ID — inherits parent if omitted (optional)",
                 "model_params": 'JSON model params e.g. {"temperature": 0.5} (optional)',
                 "context_packs": "Comma-separated pack names to pre-load (optional)"},
                required=["name", "description", "prompt"],
            ),
            self._exec_make_subagent,
        )
        _reg(
            "run_subagent",
            "Execute a named sub-agent and wait for its final result. Maximum chain depth is 3: root -> level1 -> level2.",
            _params({"name": "Name of the sub-agent to run"}, required=["name"]),
            self._exec_run_subagent,
            parallel=False,
        )
        _reg(
            "edit_subagent",
            "Modify a sub-agent definition for future invocations.",
            _params(
                {"name": "Sub-agent to edit",
                 "description": "New system prompt (optional)",
                 "prompt": "New task prompt (optional)",
                 "tools": "New tool names (optional)",
                 "model": "New model ID (optional)",
                 "model_params": "New JSON model params (optional)",
                 "context_packs": "New pack names (optional)"},
                required=["name"],
            ),
            self._exec_edit_subagent,
        )
        _reg(
            "list_subagents",
            "List all sub-agent definitions and their status.",
            _no_params(),
            self._exec_list_subagents,
        )

        # ── Dynamic Tool Tools ─────────────────────────────────────────────

        _reg(
            "make_tool_call",
            "Create a dynamic tool. Code must define: execute(args: dict) -> str",
            _params(
                {"name": "Unique tool name", "description": "What the tool does",
                 "parameters": "JSON schema for tool parameters",
                 "code": "Python code defining execute(args: dict) -> str"},
                required=["name", "description", "parameters", "code"],
            ),
            self._exec_make_tool_call,
        )
        _reg(
            "run_tool_call",
            "Execute a registered dynamic tool with given arguments.",
            _params(
                {"name": "Name of the dynamic tool",
                 "args": "JSON string of arguments (optional)"},
                required=["name"],
            ),
            self._exec_run_tool_call,
        )
        _reg(
            "edit_tool_call",
            "Modify a dynamic tool definition for future calls.",
            _params(
                {"name": "Tool to edit", "description": "New description (optional)",
                 "parameters": "New JSON schema (optional)",
                 "code": "New Python code (optional)"},
                required=["name"],
            ),
            self._exec_edit_tool_call,
        )
        _reg(
            "list_tools",
            "List all available tools: user-provided, dynamic, and meta.",
            _no_params(),
            self._exec_list_tools,
        )

        # ── Execution Control Tools ────────────────────────────────────────

        _reg(
            "branch_execution_path",
            "Fork into N parallel execution paths with distinct prompts. Returns all results. Maximum chain depth is 3: root -> level1 -> level2.",
            _params(
                {"prompts": "JSON array of prompt strings, one per branch (min 2)"},
                required=["prompts"],
            ),
            self._exec_branch_execution_path,
            parallel=False,
        )
        _reg(
            "reset_to_checkpoint",
            "Roll back execution memory to a checkpoint (by name or step number).",
            _params({"target": "Checkpoint name or step number"}, required=["target"]),
            self._exec_reset_to_checkpoint,
        )
        _reg(
            "create_checkpoint",
            "Create a named checkpoint at the current execution step.",
            _params({"name": "Name for the checkpoint"}, required=["name"]),
            self._exec_create_checkpoint,
        )
        _reg(
            "view_checkpoints",
            "List all available checkpoints (auto and named).",
            _no_params(),
            self._exec_view_checkpoints,
        )
        _reg(
            "view_execution_history",
            "Review recent execution steps, messages, and tool results.",
            _params({"limit": ("integer", "Max entries to show (default 10)")}),
            self._exec_view_execution_history,
        )
        _reg(
            "check_budget",
            "View remaining cost budget and current usage statistics.",
            _no_params(),
            self._exec_check_budget,
        )
        _reg(
            "end_execution",
            "Terminate execution and return the final result.",
            _params({"result": "The final result to return"}, required=["result"]),
            self._exec_end_execution,
        )

        return tools, executors


# ── Parameter schema helpers ───────────────────────────────────────────────


def _no_params() -> dict:
    """Schema for a tool with no parameters."""
    return {"type": "object", "properties": {}}


def _params(
    fields: Dict[str, str | tuple],
    required: List[str] | None = None,
) -> dict:
    """Build a parameter schema from a compact description.

    *fields* maps param names to either a description string (type defaults
    to ``"string"``) or a ``(type, description)`` tuple.
    """
    properties: dict = {}
    for name, spec in fields.items():
        if isinstance(spec, tuple):
            ptype, desc = spec
        else:
            ptype, desc = "string", spec
        properties[name] = {"type": ptype, "description": desc}
    schema: dict = {"properties": properties}
    if required:
        schema["required"] = required
    return schema
