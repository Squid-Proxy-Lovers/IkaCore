from typing import Optional, List, Dict, Callable, Any

from IkaCore.tools import IkaTools
from IkaModel.base import AgentTool, ToolArgs


class IkaStage:
    def __init__(
        self,
        name: str,
        prompt: str,
        tools: List[IkaTools],
        stage_max_step: int = 1,
        subagents: Optional[list] = None,
        allowed_back_to: Optional[List[int]] = None,
        hitl: bool = False,
        memory_access: Optional[Dict[str, bool]] = None,
        long_term_filter: Optional[Callable[..., Any]] = None,
        checkpoint: bool = False,
    ):
        self.name = name
        self.prompt = prompt
        self.tools = list(tools)
        self.stage_max_step = stage_max_step
        self.subagents = subagents or []
        self.allowed_back_to = allowed_back_to or []
        self.hitl = hitl
        self.memory_access = memory_access  # If None, inherits from agent
        self.long_term_filter = long_term_filter  # Optional stage-specific filter applied after search results
        self.checkpoint = checkpoint

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