from typing import Optional, List
import sys
from pathlib import Path

from tools import SquidTools
from squidrag import SquidRAGSource

import importlib.util

base_model_path = Path(__file__).parent.parent / "ika-model" / "base.py"
base_spec = importlib.util.spec_from_file_location("base", base_model_path)
base = importlib.util.module_from_spec(base_spec)
sys.modules["base"] = base
base_spec.loader.exec_module(base)

AgentTool = base.AgentTool
ToolArgs = base.ToolArgs


class SquidStage:
    def __init__(
        self,
        name: str,
        prompt: str,
        tools: List[SquidTools],
        RAGSource: List[type[SquidRAGSource]] = [],
        stage_max_step: int = 1,
        subagents: Optional[list] = None,
        allowed_back_to: Optional[List[int]] = None,
    ):
        self.name = name
        self.prompt = prompt
        self.tools = tools
        self.RAGSource = RAGSource
        self.stage_max_step = stage_max_step
        self.subagents = subagents or []
        self.allowed_back_to = allowed_back_to or []

        self.tools.append(
            AgentTool(
                id="stage_end",
                name="stage_end",
                description="Signal that the current stage is complete.",
                args=ToolArgs(type="stage_end", description="Optional reason for ending the stage."),
                required=False,
            )
        )

        if self.allowed_back_to:
            self.tools.append(
                AgentTool(
                    id="change_stage",
                    name="change_stage",
                    description=f"Request to move back to a previous stage. Allowed targets: {allowed_back_to}" if allowed_back_to else "Request to move back to a previous stage.",
                    args=ToolArgs(type="stage_index", description="Stage index to move back to (must be in allowed_back_to)."),
                    required=True,
                )
            )