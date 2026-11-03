from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Protocol

from IkaCore.cli_output import get_cli_output
from IkaModel.base import AgentEndException, AgentTool, BareBoneModel
from IkaModel.chat_interface.chat_interface import async_chat, chat, summarise_message_history


class _ModelFactoryState(Protocol):
    model_id: str
    api_key: str
    api_url: str
    prompt: str
    max_tokens: int
    temperature: float
    name: str


class _ChatRoundState(Protocol):
    use_async: bool
    message_history: Dict[str, Dict[str, object]]
    _total_usage: Dict[str, int]
    _total_cost: Dict[str, float]

    def _run_chat_round(
        self,
        barebone_model: BareBoneModel,
        messages: List[dict],
        message_history: Dict[str, Dict[str, object]],
        tool_executors: Optional[Dict[str, Callable]],
        logger: Optional[Any],
        timeout: float,
        max_tool_rounds: int,
        max_tool_calls: Optional[int],
        current_stage_index: Optional[int],
        total_stages: int,
        client: Optional[Any],
    ) -> Dict[str, Any]:
        ...

    def _accumulate_response_totals(self, response: Dict[str, Any]) -> None:
        ...


class _FinalOutputState(Protocol):
    summarize_final: bool
    message_history: Dict[str, Dict[str, object]]
    logger: Optional[Any]
    name: str
    _total_usage: Dict[str, int]
    _total_cost: Dict[str, float]


class AgentModelFactoryMixin:
    @staticmethod
    def _override_value(overrides: Dict[str, Any], key: str, default: Any) -> Any:
        return overrides[key] if key in overrides and overrides[key] is not None else default

    @staticmethod
    def _parallel_tool_calls(agent_tools: List[AgentTool]) -> bool:
        return all(not hasattr(tool, "parallel") or tool.parallel for tool in agent_tools)

    def get_barebone(
        self: _ModelFactoryState,
        system_prompt: str,
        agent_tools: List[AgentTool],
        parent_hierarchy: Optional[List[str]] = None,
        suppress_init_output: bool = False,
        model_overrides: Optional[Dict[str, Any]] = None,
        content_prompt_override: Optional[str] = None,
    ) -> BareBoneModel:
        """
        conversion to barebone model, this is used to create the model instance for the agent
        barebone model is used for conversion to apis and handling of the model instance.
        """
        overrides = model_overrides or {}
        override = AgentModelFactoryMixin._override_value
        model = BareBoneModel(
            model_id=override(overrides, "model_id", self.model_id),
            api_key=override(overrides, "api_key", self.api_key),
            api_url=override(overrides, "api_url", self.api_url),
            system_prompt=system_prompt,
            content_prompt=content_prompt_override if content_prompt_override is not None else self.prompt,
            max_tokens=override(overrides, "max_tokens", self.max_tokens),
            temperature=override(overrides, "temperature", self.temperature),
            parallel_tool_calls=AgentModelFactoryMixin._parallel_tool_calls(agent_tools),
            agent_name=self.name,
            agent_hierarchy=parent_hierarchy if parent_hierarchy else [self.name],
            suppress_init_output=suppress_init_output,
            reasoning_effort=overrides.get("reasoning_effort") or getattr(self, "reasoning_effort", None),
            use_responses_api=getattr(self, "use_responses_api", True),
        )
        model.agent_tools = agent_tools
        model._current_step = 0
        return model


class AgentChatTotalsMixin:
    def _accumulate_response_totals(self: _ChatRoundState, response: Dict[str, Any]) -> None:
        resp_usage = response.get("usage") or {}
        for key in ("input_tokens", "output_tokens", "total_tokens", "input_cached_tokens"):
            self._total_usage[key] = self._total_usage.get(key, 0) + ((resp_usage.get(key, 0)) or 0)

        resp_cost = response.get("cost") or {}
        for key in ("input_cost", "output_cost", "total_cost"):
            self._total_cost[key] = self._total_cost.get(key, 0.0) + ((resp_cost.get(key, 0.0)) or 0.0)


class AgentChatRoundMixin(AgentChatTotalsMixin):
    def _run_chat_round(
        self: _ChatRoundState,
        barebone_model: BareBoneModel,
        messages: List[dict],
        message_history: Dict[str, Dict[str, object]],
        tool_executors: Optional[Dict[str, Callable]],
        logger: Optional[Any],
        timeout: float,
        max_tool_rounds: int,
        max_tool_calls: Optional[int],
        current_stage_index: Optional[int],
        total_stages: int,
        client: Optional[Any],
    ) -> Dict[str, Any]:
        kwargs = {
            "tool_executors": tool_executors,
            "logger": logger,
            "timeout": timeout,
            "client": client,
            "max_tool_rounds": max_tool_rounds,
            "max_tool_calls": max_tool_calls,
            "current_stage_index": current_stage_index,
            "total_stages": total_stages,
        }
        if self.use_async:
            return asyncio.run(async_chat(barebone_model, messages, message_history, **kwargs))
        return chat(barebone_model, messages, message_history, **kwargs)

    def chat_wrapper(
        self: _ChatRoundState,
        barebone_model: BareBoneModel,
        messages: List[dict],
        tool_executors: Optional[Dict[str, Callable]] = None,
        logger: Optional[Any] = None,
        timeout: float = 900.0,
        max_tool_rounds: int = 5,
        max_tool_calls: Optional[int] = None,
        current_stage_index: Optional[int] = None,
        total_stages: int = 0,
        client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        try:
            result = self._run_chat_round(
                barebone_model,
                messages,
                self.message_history,
                tool_executors,
                logger,
                timeout,
                max_tool_rounds,
                max_tool_calls,
                current_stage_index,
                total_stages,
                client,
            )
        except AgentEndException as exc:
            self._accumulate_response_totals(exc.response or {})
            raise

        self._accumulate_response_totals(result)
        return result


class AgentFinalOutputMixin:
    def _fallback_final_content(
        self,
        agent_end_text: Optional[str],
        content_before_tools: str,
        last_content: str,
    ) -> str:
        """
        Build final content when agent_end was called but tool arguments were empty.

        Fallback order:
        1. agent_end_text from tool arguments
        2. content_before_tools
        3. last_content
        """
        if agent_end_text and agent_end_text.strip() not in [".", ""]:
            return agent_end_text
        if content_before_tools and content_before_tools.strip() not in ["", "{}"]:
            return content_before_tools
        if last_content and last_content.strip() not in ["", "{}"]:
            return last_content
        raise ValueError(
            "agent_end was called but no output was provided. "
            "The agent MUST provide a final answer when calling agent_end."
        )

    def _build_final_output(self: _FinalOutputState, final_message: str, barebone_model: BareBoneModel) -> Dict[str, Any]:
        summary = ""
        if self.summarize_final:
            summary_entry = self.message_history.get("summary", {})
            history_summary = summary_entry.get("message", "") if isinstance(summary_entry, dict) else ""
            if not isinstance(history_summary, str):
                history_summary = str(history_summary or "")
            summary = summarise_message_history(barebone_model, self.message_history) or history_summary
            if summary:
                logger = self.logger
                if logger:
                    logger.log_summary(summary)
                cli = get_cli_output()
                current_hierarchy = getattr(self, "_parent_hierarchy", []) + [self.name]
                step = cli.get_step(self.name) or 0
                cli.summarization(self.name, summary, current_hierarchy, step=step)

        if not summary or summary.strip() == "":
            summary = final_message
        if not final_message or final_message.strip() == "":
            final_message = summary

        return {
            "final_message": final_message,
            "summary": summary,
            "usage": getattr(self, "_total_usage", {}),
            "cost": getattr(self, "_total_cost", {}),
            "model_id": barebone_model.model_id,
        }


class AgentChatSupportMixin(AgentModelFactoryMixin, AgentChatRoundMixin, AgentFinalOutputMixin):
    pass
