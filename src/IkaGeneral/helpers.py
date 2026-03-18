"""Shared utility functions for IkaGeneral."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Ensure src is on path for IkaCore/IkaModel imports
_src_dir = str(Path(__file__).parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from IkaCore.tools import IkaTools  # noqa: E402
from IkaModel.base import AgentTool, ToolArgs  # noqa: E402


def estimate_tokens(text: str) -> int:
    """Quick token estimate: ~4 chars per token."""
    return max(1, len(str(text)) // 4) if text else 0


def resolve_api_url(model_id: str) -> str:
    """Determine API URL from model_id string."""
    mid = model_id.lower()
    if "/" in mid:
        return "https://openrouter.ai/api/v1/chat/completions"
    if "deepseek" in mid:
        return "https://api.deepseek.com/chat/completions"
    if any(k in mid for k in ("gpt", "o1", "o3", "o4")):
        return "https://api.openai.com/v1/chat/completions"
    if "claude" in mid:
        return "https://api.anthropic.com/v1/messages"
    if "gemini" in mid:
        return (
            f"https://generativelanguage.googleapis.com"
            f"/v1beta/models/{model_id}:generateContent"
        )
    return "https://api.openai.com/v1/chat/completions"


ENV_PREAMBLE = (
    "You are an autonomous agent in a continuous execution environment (IkaGeneral). "
    "You have access to meta-tools for:\n"
    "- Context packs: Store, load, and manage information across your execution\n"
    "- Sub-agents: Create and run specialized agents for subtasks\n"
    "- Dynamic tools: Create custom tools at runtime with Python code\n"
    "- Branching: Fork into parallel execution paths for exploration\n"
    "- Checkpoints: Save and restore execution state\n"
    "- Introspection: View execution history, budget, and available tools\n\n"
    "Work autonomously towards your goal. Call end_execution with your final result "
    "when complete.\n\n"
)


def convert_tools(tools: List[IkaTools]) -> List[AgentTool]:
    """Convert a list of IkaTools to AgentTool instances for the LLM."""
    converted: List[AgentTool] = []
    for tool in tools:
        if isinstance(tool, AgentTool):
            converted.append(tool)
            continue

        properties: dict = {}
        required_params: list = []
        params = getattr(tool, "parameters", None)

        if params and isinstance(params, dict):
            if "properties" in params:
                properties = params.get("properties", {})
                required_params = params.get("required", [])
            else:
                for pname, pval in params.items():
                    if isinstance(pval, dict):
                        p = {k: v for k, v in pval.items() if k != "required"}
                        properties[pname] = p
                        if pval.get("required", False):
                            required_params.append(pname)
                    elif isinstance(pval, str):
                        properties[pname] = {"type": "string", "description": pval}

        if required_params:
            properties["__required__"] = required_params

        tool_args = ToolArgs(
            type="object" if properties else "input",
            description=tool.description or "Tool input",
            properties=properties if properties else None,
        )
        converted.append(
            AgentTool(
                id=tool.id,
                name=tool.name,
                description=tool.description,
                args=tool_args,
                required=getattr(tool, "required", True),
                parallel=getattr(tool, "parallel", True),
                limit_calls=getattr(tool, "limit_calls", 0),
            )
        )
    return converted
