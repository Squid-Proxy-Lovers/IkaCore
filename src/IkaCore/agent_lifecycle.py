from __future__ import annotations

import time
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, cast

from IkaCore.cli_output import OutputType, get_cli_output
from IkaCore.stages import IkaStage

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent


class _WorkflowContextState(Protocol):
    message_history: Dict[str, Dict[str, object]]
    Stages: List[IkaStage]
    start_prompt: Optional[str]
    end_prompt: Optional[str]
    description: str


class _CloneConfigState(_WorkflowContextState, Protocol):
    checkpoint_store: Any
    name: str
    prompt: str
    system_prompt: Optional[str]
    tools: List[Any]
    model_id: str
    api_key: str
    api_url: Optional[str]
    max_tokens: int
    temperature: float
    checkpoint: bool
    subagents: List["IkaBaseAgent"]
    next_agent: Optional["IkaBaseAgent"]
    maxsteps: int
    step_timeout: int
    rate_limit_per_min: Optional[float]
    per_tool_rate_limit: Dict[str, float]
    memory: bool
    memory_access: Dict[str, bool]
    logging_file: str
    summarize_final: bool
    use_async: bool
    max_tool_rounds: int

    def _clone_stage_for_run(self, stage: IkaStage) -> IkaStage:
        ...

    def _clone_checkpoint_db_path(self) -> str:
        ...


class _CloneState(_CloneConfigState, Protocol):
    logging_level: int
    logger: Any
    final_answer_checks: List[Callable]
    short_term_memory: Any
    long_term_memory: Any

    def _initial_message_history(self, system_prompt: Optional[str]) -> Dict[str, Dict[str, object]]:
        ...

    def _clone_constructor_kwargs(self, name: Optional[str], prompt: Optional[str]) -> dict:
        ...

    def _sync_clone_runtime_state(self, clone: "IkaBaseAgent", include_history: bool) -> None:
        ...

    def _copy_clone_overrides(self, clone: "IkaBaseAgent") -> None:
        ...


class _CheckpointState(Protocol):
    Stages: List[IkaStage]
    name: str
    message_history: Dict[str, Dict[str, object]]
    maxsteps: int
    memory_access: Dict[str, bool]
    logger: Any
    checkpoint_store: Any
    _resume_checkpoint: Optional[dict]

    def _stage_checkpoint_enabled(self, stage_index: int, scope: str) -> bool:
        ...

    def _stage_checkpoint_payload(
        self,
        stage_index: int,
        remaining_steps: int,
        last_content: str,
        scope: str,
        extra_payload: Optional[dict],
    ) -> dict:
        ...

    def _log_stage_checkpoint(self, scope: str, stage_index: int, stage_name: str, checkpoint_uid: str, remaining_steps: int) -> None:
        ...

    def _emit_stage_checkpoint(self, scope: str, stage_index: int, stage_name: str, checkpoint_uid: str, remaining_steps: int) -> None:
        ...

    def _agent_checkpoint_payload(self, remaining_steps: int, last_content: str) -> dict:
        ...


class AgentWorkflowContextMixin:
    def inject_workflow_context(self: _WorkflowContextState, context: str) -> None:
        if not context:
            return
        current_first_input = self.message_history.get("first_input", {}).get("message", "")
        combined = f"{context}\n\n{current_first_input}" if current_first_input else context
        self.message_history["first_input"]["message"] = combined

    def apply_workflow_stage_wiring(self: _WorkflowContextState, stage_wiring: Dict[int, Dict[str, List["IkaBaseAgent"]]]) -> None:
        if not stage_wiring or not self.Stages:
            return
        for stage_idx, wiring in stage_wiring.items():
            if 0 <= stage_idx < len(self.Stages) and "subagents" in wiring:
                self.Stages[stage_idx].subagents = wiring["subagents"]

    @staticmethod
    def _clone_stage_for_run(stage: IkaStage) -> IkaStage:
        cloned = copy(stage)
        cloned.tools = list(getattr(stage, "tools", []))
        cloned.subagents = list(getattr(stage, "subagents", []))
        cloned.allowed_back_to = list(getattr(stage, "allowed_back_to", []))
        cloned.memory_access = deepcopy(getattr(stage, "memory_access", None))
        return cloned

    def final_prompt(self: _WorkflowContextState, stage: IkaStage) -> str:
        parts = [self.start_prompt, stage.prompt, self.end_prompt]
        if not any(parts):
            parts = [self.description, stage.prompt]
        return "\n\n".join([part for part in parts if part])


class AgentCloneConfigMixin:
    def _clone_checkpoint_db_path(self: _CloneConfigState) -> str:
        if self.checkpoint_store is not None:
            return getattr(self.checkpoint_store, "db_path", "checkpoints.db")
        return "checkpoints.db"

    def _clone_constructor_kwargs(self: _CloneConfigState, name: Optional[str], prompt: Optional[str]) -> dict:
        return {
            "name": name or self.name,
            "description": self.description,
            "prompt": self.prompt if prompt is None else prompt,
            "system_prompt": self.system_prompt,
            "start_prompt": self.start_prompt,
            "end_prompt": self.end_prompt,
            "tools": list(self.tools),
            "model_id": self.model_id,
            "api_key": self.api_key,
            "api_url": self.api_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "use_responses_api": getattr(self, "use_responses_api", True),
            "checkpoint": self.checkpoint,
            "Stages": [self._clone_stage_for_run(stage) for stage in self.Stages],
            "subagents": list(self.subagents),
            "next_agent": self.next_agent,
            "maxsteps": self.maxsteps,
            "step_timeout": self.step_timeout,
            "rate_limit_per_min": self.rate_limit_per_min,
            "per_tool_rate_limit": dict(self.per_tool_rate_limit),
            "memory": self.memory,
            "memory_access": deepcopy(self.memory_access),
            "final_answer_check": None,
            "logging_level": 0,
            "logging_file": self.logging_file,
            "checkpoint_db_path": self._clone_checkpoint_db_path(),
            "summarize_final": self.summarize_final,
            "use_async": self.use_async,
            "max_tool_rounds": self.max_tool_rounds,
            "max_tool_calls": getattr(self, "max_tool_calls", None),
            "max_step_extensions": getattr(self, "max_step_extensions", 2),
            "extend_steps_by": getattr(self, "extend_steps_by", 3),
            "max_stage_extensions": getattr(self, "max_stage_extensions", 2),
            "extend_stage_steps_by": getattr(self, "extend_stage_steps_by", 3),
            "reasoning_effort": getattr(self, "reasoning_effort", None),
        }


class AgentCloneStateMixin(AgentCloneConfigMixin):
    def _sync_clone_runtime_state(self: _CloneState, clone: "IkaBaseAgent", include_history: bool) -> None:
        clone.logging_level = self.logging_level
        clone.logger = self.logger
        clone.final_answer_checks = list(self.final_answer_checks)
        clone.message_history = (
            deepcopy(self.message_history)
            if include_history
            else self._initial_message_history(clone.system_prompt)
        )
        clone.short_term_memory = self.short_term_memory
        clone.long_term_memory = self.long_term_memory
        clone.client = None

    def _copy_clone_overrides(self: _CloneState, clone: "IkaBaseAgent") -> None:
        for attribute_name in ("context_budget", "_base_prompt"):
            if hasattr(self, attribute_name):
                setattr(clone, attribute_name, getattr(self, attribute_name))
        if hasattr(self, "_parent_hierarchy"):
            setattr(clone, "_parent_hierarchy", list(getattr(self, "_parent_hierarchy", [])))
        for override_name in ("execution", "async_execution"):
            if override_name in getattr(self, "__dict__", {}):
                setattr(clone, override_name, getattr(self, override_name))


class AgentCloneMixin(AgentCloneStateMixin):
    def clone_for_run(
        self: _CloneState,
        *,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
        include_history: bool = False,
    ) -> "IkaBaseAgent":
        clone = cast("IkaBaseAgent", self.__class__(**self._clone_constructor_kwargs(name, prompt)))
        self._sync_clone_runtime_state(clone, include_history)
        self._copy_clone_overrides(clone)
        return clone


class StageCheckpointPayloadMixin:
    def _stage_checkpoint_enabled(self: _CheckpointState, stage_index: int, scope: str) -> bool:
        if stage_index < len(self.Stages):
            stage = self.Stages[stage_index]
            return scope != "stage" or getattr(stage, "checkpoint", False)
        return True

    def _stage_checkpoint_payload(
        self: _CheckpointState,
        stage_index: int,
        remaining_steps: int,
        last_content: str,
        scope: str,
        extra_payload: Optional[dict],
    ) -> dict:
        payload = {
            "scope": scope,
            "agent_name": self.name,
            "stage_index": stage_index,
            "remaining_steps": remaining_steps,
            "last_content": last_content,
            "message_history": deepcopy(self.message_history),
            "maxsteps": self.maxsteps,
            "memory_access": deepcopy(self.memory_access),
            "timestamp": time.time(),
        }
        if extra_payload:
            payload.update(extra_payload)
        return payload


class StageCheckpointOutputMixin(StageCheckpointPayloadMixin):
    def _log_stage_checkpoint(self: _CheckpointState, scope: str, stage_index: int, stage_name: str, checkpoint_uid: str, remaining_steps: int) -> None:
        logger = self.logger
        if not logger:
            return
        if logger.level == 2:
            logger.log_json(
                {
                    "event": "checkpoint_saved",
                    "scope": scope,
                    "stage_index": stage_index,
                    "stage_name": stage_name,
                    "checkpoint_uid": checkpoint_uid,
                    "remaining_steps": remaining_steps,
                }
            )
            return
        logger.write_line(
            f"[CHECKPOINT] scope={scope} stage_index={stage_index} stage_name={stage_name} "
            f"uid={checkpoint_uid} remaining={remaining_steps}"
        )

    def _emit_stage_checkpoint(self: _CheckpointState, scope: str, stage_index: int, stage_name: str, checkpoint_uid: str, remaining_steps: int) -> None:
        cli = get_cli_output()
        current_hierarchy = getattr(self, "_parent_hierarchy", []) + [self.name, f"Stage {stage_index}: {stage_name}"]
        checkpoint_msg = (
            f"Checkpoint saved at {scope.upper()} Stage {stage_index}: {stage_name}\n"
            f"Checkpoint UID: {checkpoint_uid}\n"
            f"Remaining steps: {remaining_steps}"
        )
        cli.emit(OutputType.AGENT_INIT, checkpoint_msg, current_hierarchy, step=0)


class AgentCheckpointMixin(StageCheckpointOutputMixin):
    def _save_stage_checkpoint(
        self: _CheckpointState,
        stage_index: int,
        remaining_steps: int,
        last_content: str,
        scope: str = "stage",
        extra_payload: Optional[dict] = None,
    ) -> Optional[str]:
        checkpoint_store = self.checkpoint_store
        if not checkpoint_store or not self._stage_checkpoint_enabled(stage_index, scope):
            return None

        payload = self._stage_checkpoint_payload(stage_index, remaining_steps, last_content, scope, extra_payload)
        checkpoint_uid = checkpoint_store.save_checkpoint(scope=scope, payload=payload)
        if checkpoint_uid:
            stage_name = self.Stages[stage_index].name if stage_index < len(self.Stages) else "unknown"
            self._log_stage_checkpoint(scope, stage_index, stage_name, checkpoint_uid, remaining_steps)
            self._emit_stage_checkpoint(scope, stage_index, stage_name, checkpoint_uid, remaining_steps)
        return checkpoint_uid

    def _agent_checkpoint_payload(self: _CheckpointState, remaining_steps: int, last_content: str) -> dict:
        return {
            "scope": "agent",
            "agent_name": self.name,
            "remaining_steps": remaining_steps,
            "last_content": last_content,
            "message_history": deepcopy(self.message_history),
            "maxsteps": self.maxsteps,
            "memory_access": deepcopy(self.memory_access),
            "timestamp": time.time(),
        }

    def _save_agent_checkpoint(self: _CheckpointState, remaining_steps: int, last_content: str) -> Optional[str]:
        checkpoint_store = self.checkpoint_store
        if not checkpoint_store:
            return None
        payload = self._agent_checkpoint_payload(remaining_steps, last_content)
        return checkpoint_store.save_checkpoint(scope="agent", payload=payload)

    def load_checkpoint(self: _CheckpointState, uid: str) -> Optional[dict]:
        checkpoint_store = self.checkpoint_store
        if not checkpoint_store:
            return None
        checkpoint_data = checkpoint_store.load_checkpoint(uid)
        self._resume_checkpoint = checkpoint_data if checkpoint_data else None
        return checkpoint_data


class AgentLifecycleMixin(AgentWorkflowContextMixin, AgentCloneMixin, AgentCheckpointMixin):
    pass
