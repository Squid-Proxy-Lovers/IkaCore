from __future__ import annotations

import json
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, local
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, cast

_stdout_lock = Lock()
_stdout_enabled = True
_output_sink: Optional[Callable[["OutputContext", str], None]] = None


def set_stdout_enabled(enabled: bool) -> None:
    """Enable or disable all stdout output produced by IkaCore."""
    global _stdout_enabled
    _stdout_enabled = bool(enabled)


def is_stdout_enabled() -> bool:
    """Return current stdout enable state for IkaCore."""
    return _stdout_enabled


def set_output_sink(sink: Optional[Callable[["OutputContext", str], None]]) -> None:
    """Set a callback that receives every rendered output box."""
    global _output_sink
    _output_sink = sink


def _route_to_sink(ctx: "OutputContext", rendered: str) -> None:
    if _output_sink is None:
        return
    try:
        _output_sink(ctx, rendered)
    except Exception:
        pass


def _write_stdout(rendered: str) -> None:
    if not _stdout_enabled:
        return
    with _stdout_lock:
        sys.stdout.write(rendered)
        sys.stdout.flush()


class OutputType(Enum):
    AGENT_INIT = "agent_init"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_RESPONSE = "agent_response"
    SUMMARIZATION = "summarization"


@dataclass
class OutputContext:
    output_type: OutputType
    step_number: int
    hierarchy_chain: List[str]
    thread_id: int
    instance_id: int
    content: str
    timestamp: float = field(default_factory=time.time)
    elapsed_seconds: Optional[float] = None

    @property
    def hierarchy_string(self) -> str:
        return " -> ".join(self.hierarchy_chain) if self.hierarchy_chain else "root"


class _BoxTextState(Protocol):
    use_colors: bool
    COLORS: Dict[OutputType, str]
    RESET: str
    NO_TRUNCATE_TYPES: set[OutputType]
    MAX_CONTENT_LENGTH: int

    def _wrap_paragraph(self, paragraph: str, inner_width: int) -> List[str]:
        ...

    def _add_wrapped_word(self, lines: List[str], current_line: str, word: str, inner_width: int) -> str:
        ...


class _BoxFrameState(Protocol):
    width: int
    MIN_BOX_WIDTH: int
    TYPE_LABELS: Dict[OutputType, str]
    HORIZONTAL: str
    VERTICAL: str
    T_RIGHT: str
    T_LEFT: str

    def _colorize(self, text: str, output_type: OutputType) -> str:
        ...

    def _border(self, left: str, right: str, box_width: int, output_type: OutputType) -> str:
        ...


class _CLIOutputStepState(Protocol):
    _lock: Lock
    _step_counters: Dict[str, int]
    _agent_start_times: Dict[str, float]


class _CLIOutputEmitState(_CLIOutputStepState, Protocol):
    _renderer: BoxRenderer
    _buffer: OutputBuffer
    _thread_local: Any

    def get_step(self, agent_name: str) -> int:
        ...

    def _get_thread_id(self) -> int:
        ...

    def _get_instance_id(self) -> int:
        ...

    def _elapsed_for_agent(self, agent_name: str) -> Optional[float]:
        ...


class _CLIOutputEventsState(Protocol):
    def emit(self, output_type: OutputType, content: str, hierarchy: List[str], step: Optional[int] = None, instance_id: Optional[int] = None) -> None:
        ...

    def _format_config(self, kwargs: Dict[str, Any]) -> str:
        ...

    def _format_tool_args(self, args: Dict[str, Any]) -> str:
        ...

    def _tool_result_prefix(self, is_error: bool, is_timeout: bool) -> str:
        ...


class BoxTextMixin:
    def _colorize(self: _BoxTextState, text: str, output_type: OutputType) -> str:
        if not self.use_colors:
            return text
        return f"{self.COLORS[output_type]}{text}{self.RESET}"

    def _truncate(self: _BoxTextState, content: str, output_type: Optional[OutputType] = None) -> str:
        if output_type in self.NO_TRUNCATE_TYPES:
            return content
        if len(content) <= self.MAX_CONTENT_LENGTH:
            return content
        return content[: self.MAX_CONTENT_LENGTH - 3] + "..."

    def _wrap_line(self: _BoxTextState, text: str, inner_width: int) -> List[str]:
        if not text:
            return [""]

        lines = []
        for paragraph in text.split("\n"):
            lines.extend(self._wrap_paragraph(paragraph, inner_width))
        return lines if lines else [""]

    def _wrap_paragraph(self: _BoxTextState, paragraph: str, inner_width: int) -> List[str]:
        if not paragraph:
            return [""]
        words = paragraph.split()
        if not words:
            return [""]

        lines: List[str] = []
        current_line = ""
        for word in words:
            current_line = self._add_wrapped_word(lines, current_line, word, inner_width)
        if current_line:
            lines.append(current_line)
        return lines

    def _add_wrapped_word(self: _BoxTextState, lines: List[str], current_line: str, word: str, inner_width: int) -> str:
        if len(current_line) + len(word) + 1 <= inner_width:
            return f"{current_line} {word}".strip()
        if current_line:
            lines.append(current_line)
        while len(word) > inner_width:
            lines.append(word[:inner_width])
            word = word[inner_width:]
        return word


class BoxFrameMixin:
    MIN_BOX_WIDTH: int = 5

    def _inner_width(self: _BoxFrameState, box_width: Optional[int] = None) -> int:
        width = self.width if box_width is None else box_width
        return max(width - 4, 1)

    def _header(self: _BoxFrameState, ctx: OutputContext) -> str:
        type_label = self.TYPE_LABELS[ctx.output_type]
        timing = f" | +{ctx.elapsed_seconds:.1f}s" if ctx.elapsed_seconds is not None else ""
        return f"[{type_label}] Step {ctx.step_number}{timing} | Thread:{ctx.thread_id} Instance:{ctx.instance_id}"

    def _box_width(self: _BoxFrameState, text_lines: List[str]) -> int:
        max_line_len = max(len(line) for line in text_lines)
        configured_width = max(self.width, self.MIN_BOX_WIDTH)
        return min(max(max_line_len + 4, 40), configured_width)

    def _border(self: _BoxFrameState, left: str, right: str, box_width: int, output_type: OutputType) -> str:
        return self._colorize(f"{left}{self.HORIZONTAL * (box_width - 2)}{right}", output_type)

    def _text_row(self: _BoxFrameState, text: str, inner_width: int, output_type: OutputType) -> str:
        padded = text.ljust(inner_width)[:inner_width]
        return self._colorize(f"{self.VERTICAL} {padded} {self.VERTICAL}", output_type)

    def _separator(self: _BoxFrameState, box_width: int, output_type: OutputType) -> str:
        return self._border(self.T_RIGHT, self.T_LEFT, box_width, output_type)


class BoxRenderer(BoxFrameMixin, BoxTextMixin):
    TOP_LEFT: str = "\u250c"
    TOP_RIGHT: str = "\u2510"
    BOTTOM_LEFT: str = "\u2514"
    BOTTOM_RIGHT: str = "\u2518"
    HORIZONTAL: str = "\u2500"
    VERTICAL: str = "\u2502"
    T_RIGHT: str = "\u251c"
    T_LEFT: str = "\u2524"

    COLORS: Dict[OutputType, str] = {
        OutputType.AGENT_INIT: "\033[93m",
        OutputType.TOOL_CALL: "\033[92m",
        OutputType.TOOL_RESULT: "\033[91m",
        OutputType.AGENT_RESPONSE: "\033[94m",
        OutputType.SUMMARIZATION: "\033[95m",
    }
    RESET: str = "\033[0m"
    BOLD: str = "\033[1m"

    TYPE_LABELS: Dict[OutputType, str] = {
        OutputType.AGENT_INIT: "AGENT INIT",
        OutputType.TOOL_CALL: "TOOL CALL",
        OutputType.TOOL_RESULT: "TOOL RESULT",
        OutputType.AGENT_RESPONSE: "AGENT RESPONSE",
        OutputType.SUMMARIZATION: "SUMMARIZATION",
    }

    MAX_CONTENT_LENGTH: int = 999999
    DEFAULT_WIDTH: int = 80
    NO_TRUNCATE_TYPES: set[OutputType] = {OutputType.AGENT_INIT, OutputType.SUMMARIZATION}

    def __init__(self, use_colors: bool = True, width: int = 80):
        self.use_colors = use_colors
        self.width = width

    def render(self, ctx: OutputContext) -> str:
        header = self._header(ctx)
        hierarchy_line = f"Chain: {ctx.hierarchy_string}"
        content = self._truncate(ctx.content, ctx.output_type)
        content_lines = self._wrap_line(content, self._inner_width())

        box_width = self._box_width([header, hierarchy_line] + content_lines)
        inner_width = self._inner_width(box_width)
        separator = self._separator(box_width, ctx.output_type)

        lines = [
            self._border(self.TOP_LEFT, self.TOP_RIGHT, box_width, ctx.output_type),
            self._text_row(header, inner_width, ctx.output_type),
            separator,
            self._text_row(hierarchy_line, inner_width, ctx.output_type),
            separator,
        ]
        lines.extend(self._text_row(content_line, inner_width, ctx.output_type) for content_line in content_lines)
        lines.append(self._border(self.BOTTOM_LEFT, self.BOTTOM_RIGHT, box_width, ctx.output_type))
        return "\n".join(lines)


class OutputBuffer:
    def __init__(self, renderer: Optional[BoxRenderer] = None):
        self._lock = Lock()
        self._buffers: Dict[Tuple[str, int], List[OutputContext]] = defaultdict(list)
        self._buffering_enabled = False
        self._renderer = renderer or BoxRenderer()

    def set_renderer(self, renderer: BoxRenderer):
        with self._lock:
            self._renderer = renderer

    def enable_buffering(self):
        with self._lock:
            self._buffering_enabled = True
            self._buffers.clear()

    def disable_buffering(self):
        with self._lock:
            self._buffering_enabled = False

    def is_buffering(self) -> bool:
        with self._lock:
            return self._buffering_enabled

    def add(self, ctx: OutputContext):
        with self._lock:
            if self._buffering_enabled:
                self._buffers[self._buffer_key(ctx)].append(ctx)
                return
        self._print_output(ctx)

    def _buffer_key(self, ctx: OutputContext) -> Tuple[str, int]:
        return (ctx.hierarchy_chain[0] if ctx.hierarchy_chain else "unknown", ctx.instance_id)

    def _print_output(self, ctx: OutputContext):
        rendered = self._renderer.render(ctx) + "\n"
        _route_to_sink(ctx, rendered)
        _write_stdout(rendered)

    def _sorted_buffered_outputs(self) -> List[OutputContext]:
        all_outputs: List[OutputContext] = []
        for outputs in self._buffers.values():
            all_outputs.extend(outputs)
        return sorted(all_outputs, key=lambda x: x.timestamp)

    def _render_buffered_outputs(self, outputs: List[OutputContext]) -> List[str]:
        rendered_outputs: List[str] = []
        for ctx in outputs:
            rendered = self._renderer.render(ctx)
            rendered_outputs.append(rendered)
            _route_to_sink(ctx, rendered + "\n")
        return rendered_outputs

    def flush(self):
        with self._lock:
            if not self._buffers:
                self._buffering_enabled = False
                return
            rendered_outputs = self._render_buffered_outputs(self._sorted_buffered_outputs())
            self._buffers.clear()
            self._buffering_enabled = False

        if rendered_outputs:
            _write_stdout("\n".join(rendered_outputs) + "\n")


class CLIOutputSingletonMixin:
    def __new__(cls):
        cls_any = cast(Any, cls)
        with cls_any._instance_lock:
            if cls_any._instance is None:
                cls_any._instance = super().__new__(cls)
                setattr(cls_any._instance, "_initialized", False)
            return cls_any._instance


class CLIOutputStepMixin:
    def get_step(self: _CLIOutputStepState, agent_name: str) -> int:
        with self._lock:
            return self._step_counters.get(agent_name, 0)

    def set_step(self: _CLIOutputStepState, agent_name: str, step: int):
        with self._lock:
            self._step_counters[agent_name] = step
            if step == 1 and agent_name not in self._agent_start_times:
                self._agent_start_times[agent_name] = time.time()

    def increment_step(self: _CLIOutputStepState, agent_name: str) -> int:
        with self._lock:
            current = self._step_counters.get(agent_name, 0)
            self._step_counters[agent_name] = current + 1
            return current + 1

    def reset_steps(self: _CLIOutputStepState):
        with self._lock:
            self._step_counters.clear()
            self._agent_start_times.clear()


class CLIOutputEmitMixin:
    def configure(self: _CLIOutputEmitState, use_colors: bool = True, width: int = 80):
        self._renderer = BoxRenderer(use_colors=use_colors, width=width)
        self._buffer.set_renderer(self._renderer)

    def start_parallel(self: _CLIOutputEmitState):
        self._buffer.enable_buffering()

    def end_parallel(self: _CLIOutputEmitState):
        self._buffer.flush()

    def is_buffering(self: _CLIOutputEmitState) -> bool:
        return self._buffer.is_buffering()

    def _get_thread_id(self) -> int:
        return threading.current_thread().ident or 0

    def _get_instance_id(self: _CLIOutputEmitState) -> int:
        return getattr(self._thread_local, "instance_id", 0)

    def set_instance_id(self: _CLIOutputEmitState, instance_id: int):
        self._thread_local.instance_id = instance_id

    def emit(self: _CLIOutputEmitState, output_type: OutputType, content: str, hierarchy: List[str], step: Optional[int] = None, instance_id: Optional[int] = None):
        agent_name = hierarchy[0] if hierarchy else "unknown"
        step_number = step if step is not None else self.get_step(agent_name)
        elapsed = self._elapsed_for_agent(agent_name)

        ctx = OutputContext(
            output_type=output_type,
            step_number=step_number,
            hierarchy_chain=hierarchy,
            thread_id=self._get_thread_id(),
            instance_id=instance_id if instance_id is not None else self._get_instance_id(),
            content=content,
            elapsed_seconds=elapsed,
        )
        self._buffer.add(ctx)

    def _elapsed_for_agent(self: _CLIOutputEmitState, agent_name: str) -> Optional[float]:
        with self._lock:
            if agent_name not in self._agent_start_times:
                self._agent_start_times[agent_name] = time.time()
            start = self._agent_start_times.get(agent_name)
        return time.time() - start if start else None


class CLIOutputAgentEventsMixin:
    def agent_init(self: _CLIOutputEventsState, agent_name: str, hierarchy: List[str], step: int = 0, description: str = "", **kwargs):
        content_parts = [f"Initializing agent: {agent_name}"]
        if description:
            content_parts.append(f"Description: {description}")
        if kwargs:
            content_parts.append(f"Config: {self._format_config(kwargs)}")
        self.emit(OutputType.AGENT_INIT, "\n".join(content_parts), hierarchy, step)

    def agent_response(self: _CLIOutputEventsState, agent_name: str, response: str, hierarchy: List[str], step: int, is_final: bool = False):
        status = " (Final)" if is_final else ""
        content = f"Agent: {agent_name}{status}\nResponse:\n{response}"
        self.emit(OutputType.AGENT_RESPONSE, content, hierarchy, step)

    def summarization(self: _CLIOutputEventsState, agent_name: str, summary: str, hierarchy: Optional[List[str]] = None, step: int = 0):
        effective_hierarchy = hierarchy if hierarchy is not None else [agent_name]
        content = f"Summary for {agent_name}:\n{summary}"
        self.emit(OutputType.SUMMARIZATION, content, effective_hierarchy, step)

    def workflow_status(self: _CLIOutputEventsState, workflow_name: str, message: str, step: int = 0):
        content = f"Workflow: {workflow_name}\n{message}"
        self.emit(OutputType.AGENT_INIT, content, [workflow_name], step)

    def _format_config(self, kwargs: Dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in kwargs.items())


class CLIOutputToolEventsMixin:
    def tool_call(self: _CLIOutputEventsState, tool_name: str, args: Dict[str, Any], hierarchy: List[str], step: int):
        content = f"Calling tool: {tool_name}\nArguments:\n{self._format_tool_args(args)}"
        self.emit(OutputType.TOOL_CALL, content, hierarchy, step)

    def tool_result(
        self: _CLIOutputEventsState,
        tool_name: str,
        result: str,
        hierarchy: List[str],
        step: int,
        is_error: bool = False,
        is_timeout: bool = False,
    ):
        content = f"Tool: {tool_name}\n{self._tool_result_prefix(is_error, is_timeout)}:\n{result}"
        self.emit(OutputType.TOOL_RESULT, content, hierarchy, step)

    def _format_tool_args(self, args: Dict[str, Any]) -> str:
        try:
            return json.dumps(args, indent=2, default=str)
        except (TypeError, ValueError):
            return str(args)

    def _tool_result_prefix(self, is_error: bool, is_timeout: bool) -> str:
        if is_timeout:
            return "TIMEOUT"
        if is_error:
            return "ERROR"
        return "Result"


class CLIOutput(
    CLIOutputSingletonMixin,
    CLIOutputToolEventsMixin,
    CLIOutputAgentEventsMixin,
    CLIOutputEmitMixin,
    CLIOutputStepMixin,
):
    _instance = None
    _instance_lock = Lock()

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._renderer = BoxRenderer()
        self._buffer = OutputBuffer(self._renderer)
        self._thread_local = local()
        self._step_counters: Dict[str, int] = {}
        self._agent_start_times: Dict[str, float] = {}
        self._lock = Lock()
        self._initialized = True


_cli_output_instance: Optional[CLIOutput] = None
_cli_output_lock = Lock()


def get_cli_output() -> CLIOutput:
    global _cli_output_instance
    with _cli_output_lock:
        instance = _cli_output_instance
        if instance is None:
            instance = CLIOutput()
            _cli_output_instance = instance
        return instance
