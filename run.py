#!/usr/bin/env python3
"""Simple runner for IkaGeneral with deepseek-chat."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from IkaGeneral import IkaExecutionEnvironment
from IkaCore.tools import IkaTools

API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-c5f5a35f7d1d4a2e9eba6ba5d5ad8523")
MODEL_ID = "deepseek-chat"
ROOT_SYSTEM_PROMPT = """
You are the root orchestrator for a complex self-executing, self-improving agent system.

Operate as a planner-manager first, not as a direct worker.

Core behavior:
- Before doing any substantive work, create a strong explicit plan for the task.
- Break the task into clear phases, dependencies, and deliverables.
- Prefer creating specialized subagents for nearly all substantive execution.
- The root agent should mainly do: planning, delegation, supervision, synthesis, checkpointing, and final integration.
- Minimize direct raw tool usage by the root agent. Do not repeatedly call low-level tools yourself when a subagent can do that work.
- If the task is non-trivial, create subagents early and assign them focused responsibilities.
- Use context packs to persist plans, intermediate findings, and reusable knowledge.
- When useful, search context packs first before recreating work.
- Re-plan when new information changes the task or invalidates prior assumptions.

Context-pack organization policy:
- The CCP shelf is system-controlled. You do not control the shelf.
- You may choose the context-pack book, and you should do so intentionally.
- Separate context by book instead of mixing everything together.
- Use `plans` for execution plans, task decomposition, milestones, and revisions.
- Use `findings` for evidence, discoveries, intermediate conclusions, and research notes.
- Use `artifacts` for patches, code snippets, outputs, structured deliverables, and reusable assets.
- Use `handoffs` for subagent summaries, integration notes, conflict resolution notes, and final synthesis inputs.
- When loading or updating a context pack that is not in the default book, explicitly specify the `book`.
- Prefer listing/searching context packs within the relevant book before creating a new pack.
- Edits to context packs should be done by deleting the old pack and creating a fresh replacement.
- Prefer `delete_context_pack` followed by `create_context_pack` when you need to rewrite or refresh a pack.
- `create_context_pack` may replace an existing pack by deleting the old entry and creating a new one.
- If you need to inspect an existing pack before replacing it, use `load_context_pack`, `list_context_packs`, or `search_context_packs`.
- Do not keep rewriting the same plan pack. Only rewrite a `plans` pack if the plan has materially changed.
- If the plan has not materially changed, continue execution instead of re-saving the same plan.

Execution policy:
- Start by producing an execution plan internally, then create the right subagents and context structure.
- Use direct root-level tool calls only when they are necessary for orchestration or when delegation would be wasteful.
- Avoid redundant tool calls, redundant subagent runs, and duplicated context.
- Keep the root context clean and high-signal.
- Periodically checkpoint meaningful milestones.
- When the task is complete, integrate the subagent outputs into one coherent final result and call end_execution.

Delegation policy:
- Prefer multiple narrow subagents over one unfocused subagent.
- Give each subagent a crisp scope, success condition, and relevant context packs.
- Use the root agent to compare subagent outputs, resolve conflicts, and decide next actions.
- Maximum execution chain depth is 3: `root -> level1 -> level2`. Do not create deeper subagent or branch chains.

Completion policy:
- Do not end early.
- Do not call end_execution until the plan has been executed or intentionally revised and completed.
- The final result must reflect integrated work, not just a raw partial trace.
""".strip()


def _resolve_within_root(root: Path, raw_path: str | None, *, allow_missing: bool = False) -> Path:
    candidate = Path(raw_path or ".")
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path '{candidate}' is outside allowed root '{root}'")
    if not allow_missing and not candidate.exists():
        raise ValueError(f"path does not exist: {candidate}")
    return candidate


def _make_repo_tools(allowed_root: Path) -> list[IkaTools]:
    def get_repo_root_execute(_args: dict) -> str:
        return str(allowed_root)

    def list_directory_execute(args: dict) -> str:
        target = _resolve_within_root(allowed_root, args.get("directory_path"))
        if not target.is_dir():
            return json.dumps({"error": f"not a directory: {target}"})
        entries = []
        for entry in sorted(target.iterdir(), key=lambda item: item.name):
            entries.append({
                "name": entry.name,
                "path": str(entry),
                "type": "directory" if entry.is_dir() else "file",
            })
        return json.dumps({"directory": str(target), "entries": entries})

    def read_file_execute(args: dict) -> str:
        target = _resolve_within_root(allowed_root, args.get("file_path"))
        if not target.is_file():
            return json.dumps({"error": f"not a file: {target}"})

        start_line = int(args.get("start_line", 1) or 1)
        end_line = args.get("end_line")
        end_line_int = int(end_line) if end_line not in (None, "", 0) else None
        max_chars = int(args.get("max_chars", 20000) or 20000)

        with open(target, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()

        start_idx = max(0, start_line - 1)
        end_idx = len(lines) if end_line_int is None else min(len(lines), end_line_int)
        content = "".join(lines[start_idx:end_idx])
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

        return json.dumps({
            "path": str(target),
            "start_line": start_line,
            "end_line": end_idx,
            "truncated": truncated,
            "content": content,
        })

    def search_text_execute(args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "query is required"})

        target = _resolve_within_root(allowed_root, args.get("directory_path") or ".")
        if not target.is_dir():
            return json.dumps({"error": f"not a directory: {target}"})

        cmd = [
            "rg",
            "-n",
            "--hidden",
            "--glob",
            "!.git",
            "--max-count",
            str(int(args.get("max_count", 200) or 200)),
            query,
            str(target),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return json.dumps({
            "command": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout[:40000],
            "stderr": completed.stderr[:10000],
        })

    def run_command_execute(args: dict) -> str:
        command = str(args.get("command", "")).strip()
        if not command:
            return json.dumps({"error": "command is required"})

        cwd = _resolve_within_root(
            allowed_root,
            args.get("cwd") or ".",
        )
        timeout_seconds = int(args.get("timeout_seconds", 300) or 300)

        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return json.dumps({
            "cwd": str(cwd),
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[:40000],
            "stderr": completed.stderr[:10000],
        })

    return [
        IkaTools(
            name="get_repo_root",
            description="Return the absolute root directory the agent is allowed to operate in.",
            parameters={"None": "No parameters are required"},
            execute_function=get_repo_root_execute,
            limit_calls=3,
        ),
        IkaTools(
            name="list_directory",
            description="List files and directories under the allowed root.",
            parameters={
                "directory_path": {
                    "type": "string",
                    "description": "Directory path relative to the allowed root, or an absolute path inside it.",
                },
            },
            execute_function=list_directory_execute,
        ),
        IkaTools(
            name="read_file",
            description="Read a file inside the allowed root. Prefer bounded line ranges and avoid giant reads.",
            parameters={
                "file_path": {"type": "string", "description": "File path inside the allowed root", "required": True},
                "start_line": {"type": "integer", "description": "1-based start line (optional)"},
                "end_line": {"type": "integer", "description": "1-based inclusive end line (optional)"},
                "max_chars": {"type": "integer", "description": "Maximum number of characters to return (optional)"},
            },
            execute_function=read_file_execute,
        ),
        IkaTools(
            name="search_text",
            description="Search text with ripgrep inside the allowed root.",
            parameters={
                "query": {"type": "string", "description": "Text or regex to search for", "required": True},
                "directory_path": {"type": "string", "description": "Directory to search in (optional)"},
                "max_count": {"type": "integer", "description": "Maximum matches to return (optional)"},
            },
            execute_function=search_text_execute,
        ),
        IkaTools(
            name="run_command",
            description="Run a shell command inside the allowed root. Use this for bounded static analysis commands, builds, grep, git, and targeted reproduction steps.",
            parameters={
                "command": {"type": "string", "description": "Shell command to execute", "required": True},
                "cwd": {"type": "string", "description": "Working directory inside the allowed root (optional)"},
                "timeout_seconds": {"type": "integer", "description": "Command timeout in seconds (optional)"},
            },
            execute_function=run_command_execute,
            parallel=False,
        ),
    ]


def main():
    user_request = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello, what can you do?"
    print(f"Request: {user_request}\n")

    allowed_root_env = os.getenv("IKA_ALLOWED_ROOT")
    allowed_root = Path(allowed_root_env).resolve() if allowed_root_env else None
    root_system_prompt = ROOT_SYSTEM_PROMPT
    user_tools = []
    if allowed_root is not None:
        root_system_prompt += (
            "\n\nRuntime filesystem policy:\n"
            f"- You are operating inside the allowed root: {allowed_root}\n"
            "- Only operate within that root.\n"
            "- Prefer static analysis, targeted commands, and bounded reads.\n"
            "- Do not use fuzzers unless the user explicitly asks for fuzzing.\n"
        )
        user_tools = _make_repo_tools(allowed_root)

    builder = (
        IkaExecutionEnvironment.builder()
        .model(MODEL_ID, api_key=API_KEY)
        .system_prompt(root_system_prompt)
        .initial_prompt(user_request)
        .tools(user_tools)
    )

    ccp_session = os.getenv("IKA_CCP_SESSION")
    if ccp_session:
        builder = builder.ccp_context(
            ccp_session,
            shelf_name=os.getenv("IKA_CCP_SHELF", "ika_general_context"),
            book_name=os.getenv("IKA_CCP_BOOK", "context_packs"),
            client_binary=os.getenv("CCP_CLIENT_BIN"),
            client_home=os.getenv("CCP_CLIENT_HOME"),
        )

    env = builder.build()

    result = asyncio.run(env.run())
    if getattr(env, "_trace_writer", None) is not None:
        env._trace_writer.append_root_text(f"\n--- Result ---\n{result}\n")
    print("\n--- Result ---")
    print(result)


if __name__ == "__main__":
    main()
