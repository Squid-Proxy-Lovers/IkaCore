# pyright: strict

from __future__ import annotations

import time
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast, get_type_hints

from IkaCore.checkpoint import CheckpointStore
from IkaCore.logging_utils import IkaLogger
from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools
from IkaModel.base import HumanInputRequired
from IkaModel.model_metadata import get_api_url_for_model

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent

JsonDict = dict[str, Any]
MessageHistory = dict[str, Any]
FinalAnswerCheck = Callable[[JsonDict], bool]
RawFinalAnswerCheck = Callable[[JsonDict], object]


class _FinalAnswerValidationState(Protocol):
    message_history: MessageHistory


class _RuntimeUtilityState(Protocol):
    name: str
    use_async: bool
    step_timeout: int
    rate_limit_per_min: Optional[float]
    per_tool_rate_limit: dict[str, float]
    logger: Optional[IkaLogger]
    client: Optional[Any]

    def _enforce_rate_limit(self, per_minute: Optional[float], last_ts_attr: str) -> None:
        ...


def _return_type_allows_bool(return_type: object) -> bool:
    get_origin = getattr(typing, "get_origin", None)
    get_args = getattr(typing, "get_args", None)
    origin = get_origin(return_type) if get_origin else None
    args = get_args(return_type) if (origin and get_args) else ()
    return return_type is bool or (origin is not None and bool in args)


def validate_final_answer_checks_for_history(
    message_history: MessageHistory,
    final_answer_check: Optional[list[RawFinalAnswerCheck]],
) -> list[FinalAnswerCheck]:
    if final_answer_check is None:
        return []

    for check in final_answer_check:
        if not callable(check):
            raise ValueError("final_answer_check must be a list of callable functions")

        func_name = getattr(check, "__name__", "unknown")
        docstring = (check.__doc__ or "").strip()
        if not docstring:
            raise ValueError(
                f"final_answer_check function '{func_name}' must have a docstring explaining what it does"
            )

        try:
            hints = get_type_hints(check)
            return_type = hints.get("return")
        except (TypeError, AttributeError):
            return_type = None

        if return_type is not None:
            if not _return_type_allows_bool(return_type):
                raise ValueError(
                    f"final_answer_check function '{func_name}' must have return type annotation of bool, got {return_type}"
                )

        result: object = check(message_history)
        if not isinstance(result, bool):
            raise ValueError(
                f"final_answer_check function '{func_name}' must return a boolean, got {type(result).__name__}"
            )

    return cast(list[FinalAnswerCheck], final_answer_check)


_validate_final_answer_checks_for_history = validate_final_answer_checks_for_history


def validate_required_fields(fields: dict[str, object]) -> None:
    for field_name, value in fields.items():
        if not value:
            raise ValueError(f"{field_name} is required for the agent")


def build_memory_access_defaults(
    memory_enabled: bool,
    overrides: Optional[dict[str, bool]] = None,
) -> dict[str, bool]:
    defaults = {
        "short_term_save": True,
        "short_term_search": True,
        "long_term_save": True,
        "long_term_search": True,
    }
    if not memory_enabled:
        defaults = {key: False for key in defaults}
    if overrides:
        defaults.update(overrides)
    return defaults


def initial_message_history(system_prompt: Optional[str]) -> MessageHistory:
    return {
        "system": {"message": system_prompt or "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


class AgentAttributeDeclarationsMixin:
    name: str
    description: str
    prompt: str
    system_prompt: Optional[str]
    start_prompt: Optional[str]
    end_prompt: Optional[str]
    tools: list[IkaTools]
    model_id: str
    api_key: str
    api_url: Optional[str]
    max_tokens: int
    temperature: float
    checkpoint: bool
    Stages: list[IkaStage]
    subagents: list["IkaBaseAgent"]
    next_agent: Optional["IkaBaseAgent"]
    maxsteps: int
    step_timeout: int
    memory: bool
    memory_access: dict[str, bool]
    message_history: MessageHistory
    logging_level: int
    logging_file: str
    logger: Optional[IkaLogger]
    rate_limit_per_min: Optional[float]
    per_tool_rate_limit: dict[str, float]
    _last_api_call_ts: float
    checkpoint_store: Optional[CheckpointStore]
    _resume_checkpoint: Optional[JsonDict]
    short_term_memory: Optional[Any]
    long_term_memory: Optional[Any]
    summarize_final: bool
    use_async: bool
    max_tool_rounds: int
    final_answer_checks: list[FinalAnswerCheck]
    _tool_call_counts: dict[str, int]
    client: Optional[Any]


class AgentConfigDefaultsMixin:
    @staticmethod
    def _validate_required_fields(fields: dict[str, object]) -> None:
        validate_required_fields(fields)

    @staticmethod
    def _build_memory_access_defaults(
        memory_enabled: bool, overrides: Optional[dict[str, bool]] = None
    ) -> dict[str, bool]:
        return build_memory_access_defaults(memory_enabled, overrides)

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


class AgentFinalAnswerValidationMixin(AgentConfigDefaultsMixin):
    @staticmethod
    def _return_type_allows_bool(return_type: object) -> bool:
        return _return_type_allows_bool(return_type)

    def _validate_final_answer_checks(
        self: _FinalAnswerValidationState,
        final_answer_check: Optional[list[RawFinalAnswerCheck]],
    ) -> list[FinalAnswerCheck]:
        return validate_final_answer_checks_for_history(self.message_history, final_answer_check)


class AgentRuntimeUtilityMixin:
    def _reset_tool_call_counts(self: Any):
        self._tool_call_counts = {}

    def _get_chat_client(self: _RuntimeUtilityState) -> Optional[Any]:
        if self.use_async:
            return self.client
        if self.client is None:
            import httpx
            self.client = httpx.Client(timeout=self.step_timeout)
        return self.client

    def _enforce_rate_limit(self, per_minute: Optional[float], last_ts_attr: str) -> None:
        if not per_minute or per_minute <= 0:
            return
        interval = 60.0 / per_minute
        now = time.time()
        last_ts = getattr(self, last_ts_attr, 0.0)
        elapsed = now - last_ts
        if elapsed < interval:
            time.sleep(interval - elapsed)
        setattr(self, last_ts_attr, time.time())

    def _enforce_rate_limit_model(self: _RuntimeUtilityState) -> None:
        cast(AgentRuntimeUtilityMixin, self)._enforce_rate_limit(self.rate_limit_per_min, "_last_api_call_ts")

    def _enforce_rate_limit_tool(self: _RuntimeUtilityState, tool_name: str) -> None:
        per_tool_limit = None
        if hasattr(self, "per_tool_rate_limit"):
            per_tool_limit = self.per_tool_rate_limit.get(tool_name)
        effective = per_tool_limit if per_tool_limit is not None else self.rate_limit_per_min
        last_attr = f"_last_tool_ts_{tool_name}"
        cast(AgentRuntimeUtilityMixin, self)._enforce_rate_limit(effective, last_attr)

    def _prompt_hitl_question(
        self: _RuntimeUtilityState,
        stage_name: str,
        question: str,
        stage_index: Optional[int] = None,
        remaining_steps: Optional[int] = None,
    ) -> str:
        if self.logger:
            self.logger.log_hitl_question(stage_name, question)
        payload = {
            "kind": "hitl",
            "agent_name": self.name,
            "stage_name": stage_name,
            "stage_index": stage_index,
            "remaining_steps": remaining_steps,
            "question": question,
        }
        raise HumanInputRequired(payload)


class AgentHelperSupportMixin(
    AgentAttributeDeclarationsMixin,
    AgentFinalAnswerValidationMixin,
    AgentRuntimeUtilityMixin,
):
    pass
