# pyright: strict

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Coroutine, Optional, cast

from IkaCore.agent_helper_support import (
    JsonDict,
    RawFinalAnswerCheck,
    build_memory_access_defaults,
    initial_message_history,
    validate_final_answer_checks_for_history,
)
from IkaCore.checkpoint import CheckpointStore
from IkaCore.logging_utils import IkaLogger
from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools
from IkaMem import LTMemory, STMemory  # type: ignore
from IkaModel.base import get_global_long_term_memory
from IkaModel.model_metadata import get_api_url_for_model

if TYPE_CHECKING:
    from IkaCore.agents import IkaBaseAgent


class AgentCoreInitializationMixin:
    @staticmethod
    def _normalize_init_collections(
        tools: Optional[list[IkaTools]],
        stages: Optional[list[IkaStage]],
        subagents: Optional[list["IkaBaseAgent"]],
    ) -> tuple[list[IkaTools], list[IkaStage], list["IkaBaseAgent"]]:
        return tools or [], stages or [], subagents or []

    def _assign_identity_and_model_config(
        self,
        name: str,
        description: str,
        prompt: str,
        system_prompt: Optional[str],
        start_prompt: Optional[str],
        end_prompt: Optional[str],
        tools: list[IkaTools],
        model_id: str,
        api_key: str,
        api_url: Optional[str],
        max_tokens: int,
        temperature: float,
        use_responses_api: bool,
        reasoning_effort: Optional[str],
    ) -> None:
        self.name = name
        self.description = description
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.start_prompt = start_prompt
        self.end_prompt = end_prompt
        self.tools = tools
        self.model_id = model_id
        self.api_key = api_key
        self.use_responses_api = use_responses_api
        self.api_url = api_url if api_url else get_api_url_for_model(model_id, use_responses_api)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def _assign_agent_topology(
        self,
        stages: list[IkaStage],
        subagents: list["IkaBaseAgent"],
        next_agent: Optional["IkaBaseAgent"],
    ) -> None:
        self.Stages = stages
        self.subagents = subagents
        self.next_agent = next_agent
        if self.Stages and (self.subagents or self.next_agent):
            raise ValueError("Subagents or next_agent are not allowed when stages are defined.")
        if not self.Stages and self.subagents and self.next_agent:
            raise ValueError("Only one of subagents or next_agent may be set when no stages are provided.")

    def _configure_agent_logger(self, logging_level: int, logging_file: str, show_usage_level0: bool) -> None:
        self.logging_level = logging_level
        self.logging_file = logging_file
        self.logger = IkaLogger(logging_level, logging_file, show_usage_level0=show_usage_level0)
        if logging_level == 3:
            import logging
            logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            logging.getLogger("IkaModel.chat_interface.chat_interface").setLevel(logging.DEBUG)


class AgentExecutionLimitInitializationMixin:
    def _assign_execution_limits(
        self,
        maxsteps: int,
        step_timeout: int,
        max_tool_rounds: Optional[int],
        max_tool_calls: Optional[int],
        max_step_extensions: int,
        extend_steps_by: int,
        max_stage_extensions: int,
        extend_stage_steps_by: int,
    ) -> None:
        self.maxsteps = maxsteps
        self.step_timeout = step_timeout
        self.max_tool_rounds = max_tool_rounds if max_tool_rounds is not None else 5
        self.max_tool_calls = max_tool_calls if max_tool_calls is not None else maxsteps
        self.max_step_extensions = max_step_extensions
        self.extend_steps_by = extend_steps_by
        self.max_stage_extensions = max_stage_extensions
        self.extend_stage_steps_by = extend_stage_steps_by


class AgentRuntimeStateInitializationMixin:
    def _assign_runtime_state(
        self,
        checkpoint: bool,
        checkpoint_db_path: str,
        rate_limit_per_min: Optional[float],
        per_tool_rate_limit: Optional[dict[str, float]],
        memory: bool,
        memory_access: Optional[dict[str, bool]],
        final_answer_check: Optional[list[RawFinalAnswerCheck]],
        summarize_final: bool,
        use_async: bool,
    ) -> None:
        self.checkpoint = checkpoint
        self.memory = memory
        self.memory_access = build_memory_access_defaults(memory, memory_access)
        self.message_history = initial_message_history(getattr(self, "system_prompt", None))
        self.rate_limit_per_min = rate_limit_per_min
        self.per_tool_rate_limit = per_tool_rate_limit or {}
        self._last_api_call_ts: float = 0.0
        self.checkpoint_store = CheckpointStore(checkpoint_db_path) if checkpoint else None
        self._resume_checkpoint: Optional[JsonDict] = None
        self._last_interrupt: Optional[JsonDict] = None
        self.short_term_memory: Optional[STMemory] = None
        self.long_term_memory: Optional[LTMemory] = get_global_long_term_memory()
        self.summarize_final = summarize_final
        self.use_async = use_async
        self.final_answer_checks = validate_final_answer_checks_for_history(self.message_history, final_answer_check)
        self._tool_call_counts: dict[str, int] = {}
        self._total_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cached_tokens": 0,
        }
        self._total_cost: dict[str, float] = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
        self.client = None


class AgentShutdownMixin:
    def shutdown(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            self.client = None
            close = getattr(client, "close", None)
            if callable(close):
                close()
            else:
                aclose = getattr(client, "aclose", None)
                if callable(aclose):
                    import asyncio
                    asyncio.run(cast(Coroutine[Any, Any, Any], aclose()))
        logger = getattr(self, "logger", None)
        if logger:
            logger.shutdown()


class AgentRuntimeInitializationMixin(
    AgentExecutionLimitInitializationMixin,
    AgentRuntimeStateInitializationMixin,
    AgentShutdownMixin,
):
    pass


class AgentInitializationMixin(AgentCoreInitializationMixin, AgentRuntimeInitializationMixin):
    pass
