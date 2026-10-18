from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from IkaCore.agent_helper_support import (
    FinalAnswerCheck,
    MessageHistory,
    initial_message_history,
    validate_required_fields,
)
from IkaCore.agent_initialization import initialize_agent
from IkaCore.agent_snapshot_execution import execute_with_snapshots
from IkaCore.agent_state import AgentRunState
from IkaCore.agent_tools import ToolExecutorMap
from IkaCore.config import (
    AgentConfig,
    AgentTopology,
    ExecutionLimits,
    ExecutionPolicy,
    OutputConfig,
    PromptConfig,
    ProviderConfig,
)
from IkaCore.execution_types import StageExecutionResult, StageTarget
from IkaCore.runtime_control import RuntimeControl
from IkaCore.snapshot_session import SnapshotSession
from IkaCore.snapshot_store import SnapshotStore
from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools
from IkaModel.base import AgentTool, BareBoneModel
from IkaModel.model_metadata import get_api_url_for_model

from .agent_chat_support import AgentChat
from .agent_execution_loop import AgentTurnLoop
from .agent_lifecycle import AgentLifecycle
from .agent_parse import AgentControl
from .agent_policy import AgentPolicy
from .agent_runtime_dispatch import AgentExecution
from .agent_runtime_foundation import AgentPreparation
from .agent_snapshots import AgentSnapshots
from .agent_tools import AgentTools

"Public agent facade composed from explicit runtime components."


class IkaBaseAgent:
    _snapshot_resume_messages: list[dict[str, Any]] | None
    _snapshot_runtime: SnapshotSession | None
    context_budget: int | None
    reasoning_effort: str | None
    extend_stage_steps_by: int
    max_stage_extensions: int
    extend_steps_by: int
    max_step_extensions: int
    max_tool_calls: int
    client: Optional[Any]
    final_answer_checks: list[FinalAnswerCheck]
    max_tool_rounds: int
    use_async: bool
    summarize_final: bool
    per_tool_rate_limit: dict[str, float]
    rate_limit_per_min: Optional[float]
    logger: Any
    logging_file: str
    logging_level: int
    step_timeout: int
    next_agent: Optional["IkaBaseAgent"]
    subagents: list["IkaBaseAgent"]
    Stages: list[IkaStage]
    temperature: float
    max_tokens: int
    api_url: Optional[str]
    api_key: str
    model_id: str
    tools: list[IkaTools]
    end_prompt: Optional[str]
    start_prompt: Optional[str]
    system_prompt: Optional[str]
    description: str
    name: str
    state: AgentRunState

    def __init__(
        self,
        name: str,
        description: str,
        prompt: str,
        *,
        model_id: str = "",
        api_key: str = "",
        tools: list[IkaTools] | None = None,
        Stages: list[IkaStage] | None = None,
        provider: ProviderConfig | None = None,
        prompts: PromptConfig | None = None,
        limits: ExecutionLimits | None = None,
        policy: ExecutionPolicy | None = None,
        topology: AgentTopology | None = None,
        output: OutputConfig | None = None,
    ):
        self.config = AgentConfig(
            name,
            description,
            prompt,
            model_id,
            api_key,
            tuple(tools or ()),
            tuple(Stages or ()),
            provider or ProviderConfig(),
            prompts or PromptConfig(),
            limits or ExecutionLimits(),
            policy or ExecutionPolicy(),
            topology or AgentTopology(),
            output or OutputConfig(),
        )
        initialize_agent(self)
        self.preparation = AgentPreparation(self)
        self.turns = AgentTurnLoop(self)
        self.execution_runtime = AgentExecution(self)
        self.policy_runtime = AgentPolicy(self)
        self.tool_runtime = AgentTools(self)
        self.controls = AgentControl(self)
        self.chat = AgentChat(self)
        self.lifecycle = AgentLifecycle(self)
        self.snapshots = AgentSnapshots(self)
        self._snapshot_runtime = None

    def shutdown(self) -> None:
        from IkaCore.agent_initialization import shutdown_agent

        shutdown_agent(self)

    @property
    def message_history(self) -> dict[str, Any]:
        return self.state.message_history

    @message_history.setter
    def message_history(self, value: dict[str, Any]) -> None:
        self.state.message_history = value

    @property
    def maxsteps(self) -> int:
        return self.state.maxsteps

    @maxsteps.setter
    def maxsteps(self, value: int) -> None:
        self.state.maxsteps = value

    @property
    def prompt(self) -> str:
        return self.state.prompt

    @prompt.setter
    def prompt(self, value: str) -> None:
        self.state.prompt = value

    @property
    def _resume_state(self) -> dict[str, Any] | None:
        return self.state._resume_state

    @_resume_state.setter
    def _resume_state(self, value: dict[str, Any] | None) -> None:
        self.state._resume_state = value

    @property
    def _last_interrupt(self) -> dict[str, Any] | None:
        return self.state._last_interrupt

    @_last_interrupt.setter
    def _last_interrupt(self, value: dict[str, Any] | None) -> None:
        self.state._last_interrupt = value

    @property
    def _tool_call_counts(self) -> dict[str, int]:
        return self.state._tool_call_counts

    @_tool_call_counts.setter
    def _tool_call_counts(self, value: dict[str, int]) -> None:
        self.state._tool_call_counts = value

    @property
    def _total_usage(self) -> dict[str, int]:
        return self.state._total_usage

    @_total_usage.setter
    def _total_usage(self, value: dict[str, int]) -> None:
        self.state._total_usage = value

    @property
    def _total_cost(self) -> dict[str, float]:
        return self.state._total_cost

    @_total_cost.setter
    def _total_cost(self, value: dict[str, float]) -> None:
        self.state._total_cost = value

    @property
    def _last_api_call_ts(self) -> float:
        return self.state._last_api_call_ts

    @_last_api_call_ts.setter
    def _last_api_call_ts(self, value: float) -> None:
        self.state._last_api_call_ts = value

    def apply_workflow_stage_wiring(self, stage_wiring: Dict[int, Dict[str, List["IkaBaseAgent"]]]) -> None:
        return self.lifecycle.apply_workflow_stage_wiring(stage_wiring)

    def build_simple_tools(self) -> List[AgentTool]:
        return self.tool_runtime.build_simple_tools()

    def build_stage(self, stage: IkaStage) -> List[AgentTool]:
        return self.tool_runtime.build_stage(stage)

    def build_tool_executors(
        self,
        tools: list[IkaTools],
        subagents: Optional[list["IkaBaseAgent"]] = None,
        parent_hierarchy: Optional[list[str]] = None,
        stage: Optional[IkaStage] = None,
        stage_index: Optional[int] = None,
        remaining_steps: Optional[int] = None,
    ) -> ToolExecutorMap:
        return self.tool_runtime.build_tool_executors(
            tools, subagents, parent_hierarchy, stage, stage_index, remaining_steps
        )

    def chat_wrapper(
        self,
        barebone_model: BareBoneModel,
        messages: List[dict[str, Any]],
        tool_executors: Optional[Dict[str, Callable[..., Any]]] = None,
        logger: Optional[Any] = None,
        timeout: float = 900.0,
        max_tool_rounds: int = 5,
        max_tool_calls: Optional[int] = None,
        current_stage_index: Optional[int] = None,
        total_stages: int = 0,
        client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return self.chat.chat_wrapper(
            barebone_model,
            messages,
            tool_executors,
            logger,
            timeout,
            max_tool_rounds,
            max_tool_calls,
            current_stage_index,
            total_stages,
            client,
        )

    def clear_pause_requests(self, *checkpoint_kinds: str) -> None:
        return self.snapshots.clear_pause_requests(*checkpoint_kinds)

    def clone_for_run(
        self, *, name: Optional[str] = None, prompt: Optional[str] = None, include_history: bool = False
    ) -> "IkaBaseAgent":
        return self.lifecycle.clone_for_run(name=name, prompt=prompt, include_history=include_history)

    def configure_runtime_control(
        self,
        *,
        pause_points: Optional[set[str]] = None,
        parallel_replay: Optional[dict[str, list[int]]] = None,
        reuse_historical_parallel_results: bool = True,
    ) -> RuntimeControl:
        return self.snapshots.configure_runtime_control(
            pause_points=pause_points,
            parallel_replay=parallel_replay,
            reuse_historical_parallel_results=reuse_historical_parallel_results,
        )

    def enable_snapshots(
        self, db_path: str = "runtime-data/snapshots.db", control: Optional[RuntimeControl] = None
    ) -> SnapshotStore:
        return self.snapshots.enable_snapshots(db_path, control)

    def execute_stage(
        self, stage_index: int, remaining_steps: int, resume_input: Optional[str] = None
    ) -> StageExecutionResult:
        return self.turns.execute_stage(stage_index, remaining_steps, resume_input)

    def final_prompt(self, stage: IkaStage) -> str:
        return self.lifecycle.final_prompt(stage)

    def fork_from_snapshot(self, snapshot_id: str, *, allow_unsafe_replay: bool = False) -> dict[str, Any]:
        return self.snapshots.fork_from_snapshot(snapshot_id, allow_unsafe_replay=allow_unsafe_replay)

    def get_barebone(
        self,
        system_prompt: str,
        agent_tools: List[AgentTool],
        parent_hierarchy: Optional[List[str]] = None,
        suppress_init_output: bool = False,
        model_overrides: Optional[Dict[str, Any]] = None,
        content_prompt_override: Optional[str] = None,
    ) -> BareBoneModel:
        return self.chat.get_barebone(
            system_prompt, agent_tools, parent_hierarchy, suppress_init_output, model_overrides, content_prompt_override
        )

    def inject_workflow_context(self, context: str, label: str = "Context from upstream workflow steps:") -> None:
        return self.lifecycle.inject_workflow_context(context, label)

    def parse_control_calls(
        self,
        tool_calls: List[dict[str, Any]],
        stage: Optional[IkaStage],
        current_stage_idx: int = 0,
        response_content: Optional[str] = None,
    ) -> Tuple[Optional[StageTarget], bool, Optional[str]]:
        return self.controls.parse_control_calls(tool_calls, stage, current_stage_idx, response_content)

    def request_pause_at(self, *checkpoint_kinds: str) -> None:
        return self.snapshots.request_pause_at(*checkpoint_kinds)

    def restart_from_frame(self, frame_id: str, *, allow_unsafe_replay: bool = False) -> dict[str, Any]:
        return self.snapshots.restart_from_frame(frame_id, allow_unsafe_replay=allow_unsafe_replay)

    def resume_from_snapshot(
        self, snapshot_id: str, *, allow_unsafe_replay: bool = False, resume_input: Optional[str] = None
    ) -> dict[str, Any]:
        return self.snapshots.resume_from_snapshot(
            snapshot_id, allow_unsafe_replay=allow_unsafe_replay, resume_input=resume_input
        )

    def run_simple(self) -> tuple[str, str]:
        return self.turns.run_simple()

    def execution(self, resume_input: str | None = None) -> dict[str, Any]:
        return execute_with_snapshots(self, self.execution_runtime.execution, resume_input)

    @staticmethod
    def _validate_required_fields(fields: dict[str, object]) -> None:
        validate_required_fields(fields)

    @staticmethod
    def _initial_message_history(system_prompt: Optional[str]) -> MessageHistory:
        return initial_message_history(system_prompt)

    @staticmethod
    def geturl(model_id: str, use_responses_api: bool = True) -> str:
        """
        Get the API URL for a model.

        IMPORTANT: Check for OpenRouter format (/) BEFORE checking provider names
        to handle cases like "google/gemini-3-flash-preview" on OpenRouter.
        """
        return get_api_url_for_model(model_id, use_responses_api)
