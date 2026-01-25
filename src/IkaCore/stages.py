from typing import Optional, List, Dict, Callable, Any

from IkaCore.tools import IkaTools
from IkaModel.base import AgentTool, ToolArgs


class IkaStage:
    def __init__(
        self,
        name: str,
        prompt: str,
        tools: List[IkaTools],
        stage_max_step: int = 100,
        subagents: Optional[list] = None,
        allowed_back_to: Optional[List[int]] = None,
        hitl: bool = False,
        memory_access: Optional[Dict[str, bool]] = None,
        long_term_filter: Optional[Callable[..., Any]] = None,
        checkpoint: bool = False,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        self.name = name
        self.prompt = prompt
        self.tools = list(tools)
        self.stage_max_step = stage_max_step
        self.subagents = subagents or []
        self.allowed_back_to = allowed_back_to or []
        self.hitl = hitl
        self.memory_access = memory_access
        self.long_term_filter = long_term_filter
        self.checkpoint = checkpoint
        self.model_id = model_id
        self.api_key = api_key
        self.api_url = api_url
        self.max_tokens = max_tokens
        self.temperature = temperature

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
                    description=f"Request to move to a previous stage. Allowed targets: {allowed_back_to}. Provide the target stage index and the reason for the change." if allowed_back_to else "Request to move to a previous stage. Provide the target stage index and the reason for the change.",
                    args=ToolArgs(
                        type="object",
                        description="Target stage to change to and reason for the change.",
                        properties={
                            "stage_index": {"type": "integer", "description": "Stage index to move to (must be in allowed_back_to)."},
                            "reason": {"type": "string", "description": "Reason for requesting the stage change."},
                            "__required__": ["stage_index", "reason"],
                        },
                    ),
                    required=True,
                )
            )

        if self.hitl:
            self.tools.append(
                AgentTool(
                    id="ask_user",
                    name="ask_user",
                    description="Ask the user a question when you need their input. Use only when you need the user to answer something. The question must be clear and direct. You can call this as often as needed.",
                    args=ToolArgs(
                        type="object",
                        description="The question to ask the user.",
                        properties={
                            "question": {"type": "string", "description": "Clear and direct question for the user."},
                            "__required__": ["question"],
                        },
                    ),
                    required=False,
                )
            )