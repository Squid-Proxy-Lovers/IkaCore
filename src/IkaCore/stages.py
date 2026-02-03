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
                description="Signals that the current stage has been completed successfully and the agent should proceed to the next stage in the workflow. Use this tool when you have accomplished all objectives for the current stage as defined in the stage prompt. Upon calling this tool, the agent will transition to the next stage with a fresh context. You can optionally provide a reason for ending the stage to help with tracking and debugging. This tool does not terminate the entire agent execution, only the current stage.",
                args=ToolArgs(type="stage_end", description="Optional explanation of why the stage is complete and what was accomplished."),
                required=False,
            )
        )

        if self.allowed_back_to:
            allowed_stages_str = ", ".join(str(idx) for idx in allowed_back_to)
            self.tools.append(
                AgentTool(
                    id="change_stage",
                    name="change_stage",
                    description=f"Requests to move execution back to a previous stage in the workflow. This tool can only move backwards, not forwards. The target stage index must be one of the allowed values: [{allowed_stages_str}]. Use this tool when you need to revisit a previous stage because new information has emerged, an error was discovered, or additional work is needed in that stage. You must provide a clear reason for the stage change. Upon calling this tool, the agent will immediately transition to the specified stage. This tool does not work for moving to the next stage; use stage_end for forward progression.",
                    args=ToolArgs(
                        type="object",
                        description="Parameters specifying which stage to move to and why",
                        properties={
                            "stage_index": {"type": "integer", "description": f"The index of the stage to move back to. Must be one of: {allowed_stages_str}"},
                            "reason": {"type": "string", "description": "Detailed explanation of why you need to return to this previous stage"},
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
                    description="Prompts the user with a question and waits for their response. Use this tool when you need human input, clarification, or decision-making that cannot be determined automatically. The question should be clear, direct, and specific about what information you need. The user's response will be returned as a string. This tool pauses agent execution until the user provides an answer. You can call this tool multiple times throughout execution if needed. It should be used for critical decisions, ambiguous situations, or when explicit user preferences are required.",
                    args=ToolArgs(
                        type="object",
                        description="The question to present to the user",
                        properties={
                            "question": {"type": "string", "description": "A clear, direct question asking the user for specific information or a decision"},
                            "__required__": ["question"],
                        },
                    ),
                    required=False,
                )
            )