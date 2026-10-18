from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, cast

from IkaCore.stages import IkaStage

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent


class _WorkflowContextState(Protocol):
    @property
    def message_history(self) -> Dict[str, Dict[str, object]]: ...

    @message_history.setter
    def message_history(self, value: Dict[str, Dict[str, object]]) -> None: ...
    Stages: List[IkaStage]
    start_prompt: Optional[str]
    end_prompt: Optional[str]
    description: str


class _CloneConfigState(_WorkflowContextState, Protocol):
    name: str
    @property
    def prompt(self) -> str: ...

    @prompt.setter
    def prompt(self, value: str) -> None: ...
    system_prompt: Optional[str]
    tools: List[Any]
    model_id: str
    api_key: str
    api_url: Optional[str]
    max_tokens: int
    temperature: float
    subagents: List["IkaBaseAgent"]
    next_agent: Optional["IkaBaseAgent"]
    @property
    def maxsteps(self) -> int: ...

    @maxsteps.setter
    def maxsteps(self, value: int) -> None: ...
    step_timeout: int
    rate_limit_per_min: Optional[float]
    per_tool_rate_limit: Dict[str, float]
    logging_file: str
    summarize_final: bool
    use_async: bool
    max_tool_rounds: int

    def _clone_stage_for_run(self, stage: IkaStage) -> IkaStage:
        ...


class _CloneState(_CloneConfigState, Protocol):
    logging_level: int
    logger: Any
    final_answer_checks: List[Callable]

    def _initial_message_history(self, system_prompt: Optional[str]) -> Dict[str, Dict[str, object]]:
        ...

    def _clone_constructor_kwargs(self, name: Optional[str], prompt: Optional[str]) -> dict:
        ...

    def _sync_clone_runtime_state(self, clone: "IkaBaseAgent", include_history: bool) -> None:
        ...

    def _copy_clone_overrides(self, clone: "IkaBaseAgent") -> None:
        ...


class AgentWorkflowContextMixin:
    def inject_workflow_context(self: _WorkflowContextState, context: str,
                                 label: str = "Context from upstream workflow steps:") -> None:
        if not context:
            return
        from .workflow_prompt_context import inject_context
        inject_context(self, context, label)

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
        return cloned

    def final_prompt(self: _WorkflowContextState, stage: IkaStage) -> str:
        parts = [self.start_prompt, stage.prompt, self.end_prompt]
        if not any(parts):
            parts = [self.description, stage.prompt]
        return "\n\n".join([part for part in parts if part])


class AgentCloneConfigMixin:

    def _clone_constructor_kwargs(self: _CloneConfigState, name: Optional[str], prompt: Optional[str]) -> dict:
        from dataclasses import replace
        config = getattr(self, "config")
        return {
            "name": name or self.name, "description": self.description,
            "prompt": self.prompt if prompt is None else prompt,
            "model_id": self.model_id, "api_key": self.api_key, "tools": list(self.tools),
            "Stages": [self._clone_stage_for_run(stage) for stage in self.Stages],
            "provider": config.provider, "prompts": config.prompts,
            "limits": replace(config.limits, maxsteps=self.maxsteps),
            "policy": replace(config.policy, final_answer_checks=()),
            "topology": config.topology, "output": replace(config.output, logging_level=0),
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
        session = getattr(self, "_snapshot_runtime", None)
        if session is not None:
            clone.enable_snapshots(session.store.db_path, deepcopy(session.control))


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


class AgentLifecycleMixin(AgentWorkflowContextMixin, AgentCloneMixin):
    pass
