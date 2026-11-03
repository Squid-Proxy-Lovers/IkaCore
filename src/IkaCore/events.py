"""Structured, scoped runtime events; subscribers own presentation and persistence."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from types import MappingProxyType
from typing import Any, Dict, List, Optional

from IkaModel.model_metadata import resolve_model_cost


class OutputType(Enum):
    AGENT_INIT = "agent_init"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_RESPONSE = "agent_response"
    SUMMARIZATION = "summarization"


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    data: Mapping[str, Any]
    instance_id: int = 0


EventSubscriber = Callable[[RuntimeEvent], None]
_active_output: ContextVar[RuntimeOutput | None] = ContextVar("ikacore_output", default=None)
_output_scoped: ContextVar[bool] = ContextVar("ikacore_output_scoped", default=False)
_instance_id: ContextVar[int] = ContextVar("ikacore_output_instance", default=0)
_shared_cli: ContextVar[Any] = ContextVar("ikacore_workflow_cli", default=None)


def compute_cost(model_id: str, usage: dict[str, Any]) -> dict[str, float]:
    rates = resolve_model_cost(model_id)
    if rates is None:
        return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
    input_rate, cached_rate, output_rate = rates
    cached = usage.get("input_cached_tokens", 0) or 0
    uncached = max((usage.get("input_tokens", 0) or 0) - cached, 0)
    input_cost = (uncached * input_rate + cached * cached_rate) / 1_000_000
    output_cost = (usage.get("output_tokens", 0) or usage.get("total_tokens", 0) or 0) * output_rate / 1_000_000
    return {"input_cost": round(input_cost, 6), "output_cost": round(output_cost, 6),
            "total_cost": round(input_cost + output_cost, 6)}


class RuntimeOutput:
    def __init__(self, subscribers: tuple[EventSubscriber, ...] = (), *, cli_adapter: Any = None) -> None:
        self.subscribers: tuple[EventSubscriber, ...] = subscribers
        self.cli_adapter: Any = cli_adapter
        self._steps: dict[str, int] = {}
        self._lock = Lock()

    def publish(self, kind: str, data: dict[str, Any]) -> None:
        if self.subscribers:
            event = RuntimeEvent(kind, MappingProxyType(data), _instance_id.get())
            for subscriber in self.subscribers:
                subscriber(event)

    def get_step(self, agent_name: str) -> int:
        with self._lock:
            return self._steps.get(agent_name, 0)

    def set_step(self, agent_name: str, step: int) -> None:
        with self._lock:
            self._steps[agent_name] = step
        self.publish("set_step", {"agent_name": agent_name, "step": step})

    def set_instance_id(self, instance_id: int) -> None:
        _instance_id.set(instance_id)

    def start_parallel(self) -> None:
        self.publish("parallel_start", {})

    def end_parallel(self) -> None:
        self.publish("parallel_end", {})

    def is_buffering(self) -> bool:
        return self.cli_adapter is not None and self.cli_adapter.is_buffering()

    @staticmethod
    def compute_cost(model_id: str, usage: dict[str, Any]) -> dict[str, float]:
        return compute_cost(model_id, usage)

    def shutdown(self) -> None:
        for subscriber in self.subscribers:
            close = getattr(subscriber, "shutdown", None)
            if close is not None:
                close()

    def flush(self) -> None:
        for subscriber in self.subscribers:
            flush = getattr(subscriber, "flush", None)
            if flush is not None:
                flush()


    def log_input(self, messages: List[dict[str, Any]]) -> None:
        self.publish('input', {'messages': messages})

    def log_tool_results(self, tool_calls: List[dict[str, Any]], tool_results: List[str]) -> None:
        self.publish('tool_results', {'tool_calls': tool_calls, 'tool_results': tool_results})

    def log_output(self, content: str, usage: Optional[Dict[str, Any]]=None, cost: Optional[Dict[str, float]]=None, message_history: Optional[dict[str, Any]]=None) -> None:
        self.publish('output', {'content': content, 'usage': usage, 'cost': cost, 'message_history': message_history})

    def log_summary(self, summary: str) -> None:
        self.publish('summary', {'summary': summary})

    def log_action(self, action: str) -> None:
        self.publish('action', {'action': action})

    def log_stage_start(self, stage_name: str, hitl: bool, remaining_steps: int, step_limit: int) -> None:
        self.publish('stage_start', {'stage_name': stage_name, 'hitl': hitl, 'remaining_steps': remaining_steps, 'step_limit': step_limit})

    def log_stage_end(self, stage_name: str, used_steps: int) -> None:
        self.publish('stage_end', {'stage_name': stage_name, 'used_steps': used_steps})

    def log_step(self, stage_name: str, step_idx: int, output: str, tool_calls: list[dict[str, Any]], usage: dict[str, Any], cost: dict[str, float], elapsed: float) -> None:
        self.publish('step', {'stage_name': stage_name, 'step_idx': step_idx, 'output': output, 'tool_calls': tool_calls, 'usage': usage, 'cost': cost, 'elapsed': elapsed})

    def log_hitl_prompt(self, stage_name: str) -> None:
        self.publish('hitl_prompt', {'stage_name': stage_name})

    def log_hitl_input(self, stage_name: str, user_text: str) -> None:
        self.publish('hitl_input', {'stage_name': stage_name, 'user_text': user_text})

    def log_hitl_question(self, stage_name: str, question: str) -> None:
        self.publish('hitl_question', {'stage_name': stage_name, 'question': question})

    def log_hitl_answer(self, stage_name: str, user_text: str) -> None:
        self.publish('hitl_answer', {'stage_name': stage_name, 'user_text': user_text})

    def agent_init(self, agent_name: str, hierarchy: List[str], step: int=0, description: str='', **kwargs: Any) -> None:
        self.publish('agent_init', {'agent_name': agent_name, 'hierarchy': hierarchy, 'step': step, 'description': description, **kwargs})

    def agent_response(self, agent_name: str, response: str, hierarchy: List[str], step: int, is_final: bool=False) -> None:
        self.publish('agent_response', {'agent_name': agent_name, 'response': response, 'hierarchy': hierarchy, 'step': step, 'is_final': is_final})

    def summarization(self, agent_name: str, summary: str, hierarchy: Optional[List[str]]=None, step: int=0) -> None:
        self.publish('summarization', {'agent_name': agent_name, 'summary': summary, 'hierarchy': hierarchy, 'step': step})

    def workflow_status(self, workflow_name: str, message: str, step: int=0) -> None:
        self.publish('workflow_status', {'workflow_name': workflow_name, 'message': message, 'step': step})

    def tool_call(self, tool_name: str, args: Dict[str, Any], hierarchy: List[str], step: int) -> None:
        self.publish('tool_call', {'tool_name': tool_name, 'args': args, 'hierarchy': hierarchy, 'step': step})

    def tool_result(self, tool_name: str, result: str, hierarchy: List[str], step: int, is_error: bool=False, is_timeout: bool=False) -> None:
        self.publish('tool_result', {'tool_name': tool_name, 'result': result, 'hierarchy': hierarchy, 'step': step, 'is_error': is_error, 'is_timeout': is_timeout})

    def emit(self, output_type: OutputType, content: str, hierarchy: list[str], step: int | None = None,
             instance_id: int | None = None) -> None:
        self.publish("display", {"output_type": output_type, "content": content,
                                 "hierarchy": hierarchy, "step": step, "instance_id": instance_id})


@contextmanager
def output_scope(output: RuntimeOutput, *, share_cli: bool = False) -> Iterator[RuntimeOutput]:
    scope_token = _output_scoped.set(True)
    token = _active_output.set(output)
    instance_token = _instance_id.set(_instance_id.get())
    cli_token = _shared_cli.set(output.cli_adapter) if share_cli else None
    try:
        yield output
    finally:
        if cli_token is not None:
            _shared_cli.reset(cli_token)
        _instance_id.reset(instance_token)
        _active_output.reset(token)
        _output_scoped.reset(scope_token)


def current_output() -> RuntimeOutput:
    output = _active_output.get()
    if output is None:
        from .config import OutputConfig
        from .output_adapters import make_output
        output = make_output(OutputConfig())
        _active_output.set(output)
    return output


def output_enabled() -> bool:
    return _output_scoped.get()


def can_inherit_silent_output(output: RuntimeOutput) -> bool:
    """A silent workflow may reuse an already silent caller's scope."""
    parent = _active_output.get()
    return not output.subscribers and parent is not None and not parent.subscribers
