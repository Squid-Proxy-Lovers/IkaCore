# pyright: reportPrivateUsage=false
from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from IkaCore.events import current_output as get_cli_output
from IkaModel.async_runner import run_async
from IkaModel.base import AgentEndException, AgentTool, BareBoneModel, HumanInputRequired
from IkaModel.chat_interface.chat_interface import async_chat, chat, summarise_message_history
from IkaModel.execution_hooks import emit_boundary
from IkaModel.request_control import check_request_controls

from .snapshot_agent_state import resume_pending_response

"AgentChat component; owns one execution responsibility."
JsonDict = dict[str, Any]
MessageHistory = dict[str, Any]
FinalAnswerCheck = Callable[[JsonDict], bool]
RawFinalAnswerCheck = Callable[[JsonDict], object]
if TYPE_CHECKING:
    from .agents import IkaBaseAgent


class AgentChat:
    def __init__(self, agent: IkaBaseAgent):
        self.agent = agent

    @staticmethod
    def _override_value(overrides: Dict[str, Any], key: str, default: Any) -> Any:
        return overrides[key] if key in overrides and overrides[key] is not None else default

    @staticmethod
    def _parallel_tool_calls(agent_tools: List[AgentTool]) -> bool:
        return all((not hasattr(tool, "parallel") or tool.parallel for tool in agent_tools))

    def get_barebone(
        self,
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
        agent = self.agent
        overrides = model_overrides or {}
        override = AgentChat._override_value
        model = BareBoneModel(
            model_id=override(overrides, "model_id", agent.model_id),
            api_key=override(overrides, "api_key", agent.api_key),
            api_url=override(overrides, "api_url", agent.api_url),
            system_prompt=system_prompt,
            content_prompt=content_prompt_override if content_prompt_override is not None else agent.prompt,
            max_tokens=override(overrides, "max_tokens", agent.max_tokens),
            temperature=override(overrides, "temperature", agent.temperature),
            parallel_tool_calls=AgentChat._parallel_tool_calls(agent_tools),
            agent_name=agent.name,
            agent_hierarchy=parent_hierarchy if parent_hierarchy else [agent.name],
            suppress_init_output=suppress_init_output,
            reasoning_effort=overrides.get("reasoning_effort") or getattr(agent, "reasoning_effort", None),
            use_responses_api=getattr(agent, "use_responses_api", True),
            thinking_budget_tokens=override(overrides, "thinking_budget_tokens", agent.config.provider.thinking_budget_tokens),
        )
        model.context_budget = agent.context_budget
        def compacted() -> None:
            agent.state.compaction_count += 1
            agent.events.publish("compaction", {"agent_name": agent.name, "count": agent.compaction_count,
                                                 "context_budget": agent.context_budget})
        model._on_compaction = compacted
        model.agent_tools = agent_tools
        model._current_step = 0
        return model

    def _accumulate_response_totals(self, response: Dict[str, Any]) -> None:
        agent = self.agent
        resp_usage = response.get("usage") or {}
        for key in ("input_tokens", "output_tokens", "total_tokens", "input_cached_tokens"):
            agent._total_usage[key] = agent._total_usage.get(key, 0) + (resp_usage.get(key, 0) or 0)
        resp_cost = response.get("cost") or {}
        for key in ("input_cost", "output_cost", "total_cost"):
            agent._total_cost[key] = agent._total_cost.get(key, 0.0) + (resp_cost.get(key, 0.0) or 0.0)

    def _run_chat_round(
        self,
        barebone_model: BareBoneModel,
        messages: List[dict[str, Any]],
        message_history: Dict[str, Dict[str, object]],
        tool_executors: Optional[Dict[str, Callable[..., Any]]],
        logger: Optional[Any],
        timeout: float,
        max_tool_rounds: int,
        max_tool_calls: Optional[int],
        current_stage_index: Optional[int],
        total_stages: int,
        client: Optional[Any],
    ) -> Dict[str, Any]:
        agent = self.agent
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
        if agent.async_requests:
            return run_async(lambda: async_chat(barebone_model, messages, message_history, **kwargs))
        return chat(barebone_model, messages, message_history, **kwargs)

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
        agent = self.agent
        saved = resume_pending_response(agent)
        if saved is not None:
            return saved
        try:
            emit_boundary("pre_model", {"messages": messages, "stage_index": current_stage_index})
            result = agent.chat._run_chat_round(
                barebone_model,
                messages,
                agent.message_history,
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
            agent.chat._accumulate_response_totals(exc.response or {})
            emit_boundary(
                "post_model",
                {
                    "response": exc.response,
                    "agent_end_exception": True,
                    "messages": messages,
                    "stage_index": current_stage_index,
                },
            )
            raise
        agent.chat._accumulate_response_totals(result)
        emit_boundary("post_model", {"response": result, "messages": messages, "stage_index": current_stage_index})
        return result

    def _fallback_final_content(
        self, agent_end_text: Optional[str], content_before_tools: str, last_content: str
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
            "agent_end was called but no output was provided. The agent MUST provide a final answer when calling agent_end."
        )

    def _build_final_output(self, final_message: str, barebone_model: BareBoneModel) -> Dict[str, Any]:
        agent = self.agent
        check_request_controls()
        summary = ""
        if agent.summarize_final:
            summary_entry = agent.message_history.get("summary", {})
            history_summary = summary_entry.get("message", "") if isinstance(summary_entry, dict) else ""
            if not isinstance(history_summary, str):
                history_summary = str(history_summary or "")
            summary = summarise_message_history(barebone_model, agent.message_history) or history_summary
            if summary:
                logger = agent.logger
                if logger:
                    logger.log_summary(summary)
                cli = get_cli_output()
                current_hierarchy = getattr(agent, "_parent_hierarchy", []) + [agent.name]
                step = cli.get_step(agent.name) or 0
                cli.summarization(agent.name, summary, current_hierarchy, step=step)
        if not summary or summary.strip() == "":
            summary = final_message
        if not final_message or final_message.strip() == "":
            final_message = summary
        check_request_controls()
        bucket = agent.message_history.setdefault("messages", {})
        last = next(reversed(bucket.values()), {}) if bucket else {}
        if last.get("message") != final_message or last.get("type", "assistant") != "assistant":
            bucket[str(uuid4())] = {"type": "assistant", "message": final_message,
                                    "tokens": max(1, len(final_message) // 4)}
        return {
            "final_message": final_message,
            "summary": summary,
            "usage": getattr(agent, "_total_usage", {}),
            "cost": getattr(agent, "_total_cost", {}),
            "model_id": barebone_model.model_id,
            "validation": {"status": "not_run", "attempts": 0, "failed_checks": []},
        }

    def _reset_tool_call_counts(self):
        agent = self.agent
        agent._tool_call_counts = {}

    def _get_chat_client(self) -> Optional[Any]:
        agent = self.agent
        if agent.async_requests:
            return agent.client
        if agent.client is None:
            from IkaModel.request_transport import new_sync_client
            from IkaModel.runtime_policy import current_runtime_options

            if current_runtime_options().reuse_connections:
                return None
            agent.client = new_sync_client(agent.step_timeout)
            agent._client_owned = True
        return agent.client

    def _enforce_rate_limit(self, per_minute: Optional[float], last_ts_attr: str) -> None:
        agent = self.agent
        if not per_minute or per_minute <= 0:
            return
        interval = 60.0 / per_minute
        now = time.time()
        last_ts = getattr(agent, last_ts_attr, 0.0)
        elapsed = now - last_ts
        if elapsed < interval:
            from IkaModel.request_lifecycle import sleep_with_controls

            sleep_with_controls(interval - elapsed, time.sleep)
        setattr(agent, last_ts_attr, time.time())

    def _enforce_rate_limit_model(self) -> None:
        agent = self.agent
        agent.chat._enforce_rate_limit(agent.rate_limit_per_min, "_last_api_call_ts")

    def _enforce_rate_limit_tool(self, tool_name: str) -> None:
        agent = self.agent
        per_tool_limit = None
        if hasattr(agent, "per_tool_rate_limit"):
            per_tool_limit = agent.per_tool_rate_limit.get(tool_name)
        effective = per_tool_limit if per_tool_limit is not None else agent.rate_limit_per_min
        last_attr = f"_last_tool_ts_{tool_name}"
        agent.chat._enforce_rate_limit(effective, last_attr)

    def _prompt_hitl_question(
        self, stage_name: str, question: str, stage_index: Optional[int] = None, remaining_steps: Optional[int] = None
    ) -> str:
        agent = self.agent
        if agent.logger:
            agent.logger.log_hitl_question(stage_name, question)
        payload = {
            "kind": "hitl",
            "agent_name": agent.name,
            "stage_name": stage_name,
            "stage_index": stage_index,
            "remaining_steps": remaining_steps,
            "question": question,
        }
        raise HumanInputRequired(payload)
