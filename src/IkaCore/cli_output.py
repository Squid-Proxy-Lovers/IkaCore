from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from threading import Lock, local
from collections import defaultdict
import sys
import time
import json
import threading


class OutputType(Enum):
    AGENT_INIT = "agent_init"          # YELLOW
    TOOL_CALL = "tool_call"            # GREEN
    TOOL_RESULT = "tool_result"        # RED
    AGENT_RESPONSE = "agent_response"  # BLUE
    SUMMARIZATION = "summarization"    # MAGENTA


@dataclass
class OutputContext:
    output_type: OutputType
    step_number: int
    hierarchy_chain: List[str]  # e.g., ["recon_agent", "safe_ping_sweep", "nmap_scan"]
    thread_id: int
    instance_id: int
    content: str
    timestamp: float = field(default_factory=time.time)

    @property
    def hierarchy_string(self) -> str:
        return " -> ".join(self.hierarchy_chain) if self.hierarchy_chain else "root"


class BoxRenderer:

    # Unicode box characters
    TOP_LEFT = "\u250c"
    TOP_RIGHT = "\u2510"
    BOTTOM_LEFT = "\u2514"
    BOTTOM_RIGHT = "\u2518"
    HORIZONTAL = "\u2500"
    VERTICAL = "\u2502"
    T_RIGHT = "\u251c"
    T_LEFT = "\u2524"

    # ANSI color codes
    COLORS = {
        OutputType.AGENT_INIT: "\033[93m",      # Yellow
        OutputType.TOOL_CALL: "\033[92m",       # Green
        OutputType.TOOL_RESULT: "\033[91m",     # Red
        OutputType.AGENT_RESPONSE: "\033[94m",  # Blue
        OutputType.SUMMARIZATION: "\033[95m",   # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Type labels for header
    TYPE_LABELS = {
        OutputType.AGENT_INIT: "AGENT INIT",
        OutputType.TOOL_CALL: "TOOL CALL",
        OutputType.TOOL_RESULT: "TOOL RESULT",
        OutputType.AGENT_RESPONSE: "AGENT RESPONSE",
        OutputType.SUMMARIZATION: "SUMMARIZATION",
    }

    MAX_CONTENT_LENGTH = 999999
    DEFAULT_WIDTH = 80

    # Output types that should never be truncated
    NO_TRUNCATE_TYPES = {OutputType.AGENT_INIT, OutputType.SUMMARIZATION}

    def __init__(self, use_colors: bool = True, width: int = 80):
        self.use_colors = use_colors
        self.width = width

    def _colorize(self, text: str, output_type: OutputType) -> str:
        if not self.use_colors:
            return text
        return f"{self.COLORS[output_type]}{text}{self.RESET}"

    def _truncate(self, content: str, output_type: Optional[OutputType] = None) -> str:
        if output_type in self.NO_TRUNCATE_TYPES:
            return content
        if len(content) <= self.MAX_CONTENT_LENGTH:
            return content
        return content[:self.MAX_CONTENT_LENGTH - 3] + "..."

    def _wrap_line(self, text: str, inner_width: int) -> List[str]:
        if not text:
            return [""]

        lines = []
        for paragraph in text.split('\n'):
            if not paragraph:
                lines.append("")
                continue

            words = paragraph.split()
            if not words:
                lines.append("")
                continue

            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= inner_width:
                    current_line = f"{current_line} {word}".strip()
                else:
                    if current_line:
                        lines.append(current_line)
                    # Handle very long words
                    while len(word) > inner_width:
                        lines.append(word[:inner_width])
                        word = word[inner_width:]
                    current_line = word

            if current_line:
                lines.append(current_line)

        return lines if lines else [""]

    def render(self, ctx: OutputContext) -> str:
        inner_width = self.width - 4  # Account for "| " and " |"

        # Build header line
        type_label = self.TYPE_LABELS[ctx.output_type]
        header = f"[{type_label}] Step {ctx.step_number} | Thread:{ctx.thread_id} Instance:{ctx.instance_id}"

        # Build hierarchy line
        hierarchy_line = f"Chain: {ctx.hierarchy_string}"

        # Truncate and wrap content (AGENT_INIT is never truncated)
        content = self._truncate(ctx.content, ctx.output_type)
        content_lines = self._wrap_line(content, inner_width)

        # Calculate box width based on longest line
        all_text_lines = [header, hierarchy_line] + content_lines
        max_line_len = max(len(line) for line in all_text_lines)
        box_width = min(max(max_line_len + 4, 40), self.width)
        inner_width = box_width - 4

        # Build box lines
        lines = []

        # Top border
        top_border = f"{self.TOP_LEFT}{self.HORIZONTAL * (box_width - 2)}{self.TOP_RIGHT}"
        lines.append(self._colorize(top_border, ctx.output_type))

        # Header line
        padded_header = header.ljust(inner_width)[:inner_width]
        lines.append(self._colorize(f"{self.VERTICAL} {padded_header} {self.VERTICAL}", ctx.output_type))

        # Separator after header
        separator = f"{self.T_RIGHT}{self.HORIZONTAL * (box_width - 2)}{self.T_LEFT}"
        lines.append(self._colorize(separator, ctx.output_type))

        # Hierarchy line
        padded_hierarchy = hierarchy_line.ljust(inner_width)[:inner_width]
        lines.append(self._colorize(f"{self.VERTICAL} {padded_hierarchy} {self.VERTICAL}", ctx.output_type))

        # Separator after hierarchy
        lines.append(self._colorize(separator, ctx.output_type))

        # Content lines
        for content_line in content_lines:
            padded_content = content_line.ljust(inner_width)[:inner_width]
            lines.append(self._colorize(f"{self.VERTICAL} {padded_content} {self.VERTICAL}", ctx.output_type))

        # Bottom border
        bottom_border = f"{self.BOTTOM_LEFT}{self.HORIZONTAL * (box_width - 2)}{self.BOTTOM_RIGHT}"
        lines.append(self._colorize(bottom_border, ctx.output_type))

        return "\n".join(lines)


class OutputBuffer:

    def __init__(self, renderer: Optional[BoxRenderer] = None):
        self._lock = Lock()
        self._buffers: Dict[int, List[OutputContext]] = defaultdict(list)
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
                self._buffers[ctx.instance_id].append(ctx)
            else:
                self._print_output(ctx)

    def _print_output(self, ctx: OutputContext):
        print(self._renderer.render(ctx))
        sys.stdout.flush()

    def flush(self):
        with self._lock:
            if not self._buffers:
                self._buffering_enabled = False
                return

            # Sort by instance_id for logical ordering
            sorted_instance_ids = sorted(self._buffers.keys())

            for instance_id in sorted_instance_ids:
                outputs = self._buffers[instance_id]
                # Sort outputs within instance by timestamp
                sorted_outputs = sorted(outputs, key=lambda x: x.timestamp)

                for ctx in sorted_outputs:
                    self._print_output(ctx)

            self._buffers.clear()
            self._buffering_enabled = False


class CLIOutput:

    _instance = None
    _instance_lock = Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        self._renderer = BoxRenderer()
        self._buffer = OutputBuffer(self._renderer)
        self._thread_local = local()
        self._step_counters: Dict[str, int] = {}
        self._lock = Lock()
        self._initialized = True

    def configure(self, use_colors: bool = True, width: int = 80):
        self._renderer = BoxRenderer(use_colors=use_colors, width=width)
        self._buffer.set_renderer(self._renderer)

    def start_parallel(self):
        self._buffer.enable_buffering()

    def end_parallel(self):
        self._buffer.flush()

    def is_buffering(self) -> bool:
        return self._buffer.is_buffering()

    def _get_thread_id(self) -> int:
        return threading.current_thread().ident or 0

    def _get_instance_id(self) -> int:
        return getattr(self._thread_local, 'instance_id', 0)

    def set_instance_id(self, instance_id: int):
        self._thread_local.instance_id = instance_id

    def get_step(self, agent_name: str) -> int:
        with self._lock:
            return self._step_counters.get(agent_name, 0)

    def set_step(self, agent_name: str, step: int):
        with self._lock:
            self._step_counters[agent_name] = step

    def increment_step(self, agent_name: str) -> int:
        with self._lock:
            current = self._step_counters.get(agent_name, 0)
            self._step_counters[agent_name] = current + 1
            return current + 1

    def reset_steps(self):
        with self._lock:
            self._step_counters.clear()

    def emit(
        self,
        output_type: OutputType,
        content: str,
        hierarchy: List[str],
        step: Optional[int] = None,
        instance_id: Optional[int] = None
    ):
        agent_name = hierarchy[0] if hierarchy else "unknown"

        ctx = OutputContext(
            output_type=output_type,
            step_number=step if step is not None else self.get_step(agent_name),
            hierarchy_chain=hierarchy,
            thread_id=self._get_thread_id(),
            instance_id=instance_id if instance_id is not None else self._get_instance_id(),
            content=content
        )

        self._buffer.add(ctx)

    # Convenience methods
    def agent_init(
        self,
        agent_name: str,
        hierarchy: List[str],
        step: int = 0,
        description: str = "",
        **kwargs
    ):
        content_parts = [f"Initializing agent: {agent_name}"]
        if description:
            content_parts.append(f"Description: {description}")
        if kwargs:
            config_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            content_parts.append(f"Config: {config_str}")
        content = "\n".join(content_parts)
        self.emit(OutputType.AGENT_INIT, content, hierarchy, step)

    def tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        hierarchy: List[str],
        step: int
    ):
        try:
            args_str = json.dumps(args, indent=2, default=str)
        except (TypeError, ValueError):
            args_str = str(args)
        content = f"Calling tool: {tool_name}\nArguments:\n{args_str}"
        self.emit(OutputType.TOOL_CALL, content, hierarchy, step)

    def tool_result(
        self,
        tool_name: str,
        result: str,
        hierarchy: List[str],
        step: int,
        is_error: bool = False,
        is_timeout: bool = False
    ):
        if is_timeout:
            prefix = "TIMEOUT"
        elif is_error:
            prefix = "ERROR"
        else:
            prefix = "Result"
        content = f"Tool: {tool_name}\n{prefix}:\n{result}"
        self.emit(OutputType.TOOL_RESULT, content, hierarchy, step)

    def agent_response(
        self,
        agent_name: str,
        response: str,
        hierarchy: List[str],
        step: int,
        is_final: bool = False
    ):
        status = " (Final)" if is_final else ""
        content = f"Agent: {agent_name}{status}\nResponse:\n{response}"
        self.emit(OutputType.AGENT_RESPONSE, content, hierarchy, step)

    def summarization(
        self,
        agent_name: str,
        summary: str,
        hierarchy: Optional[List[str]] = None,
        step: int = 0
    ):
        if hierarchy is None:
            hierarchy = [agent_name]
        content = f"Summary for {agent_name}:\n{summary}"
        self.emit(OutputType.SUMMARIZATION, content, hierarchy, step)

    def workflow_status(
        self,
        workflow_name: str,
        message: str,
        step: int = 0
    ):
        content = f"Workflow: {workflow_name}\n{message}"
        self.emit(OutputType.AGENT_INIT, content, [workflow_name], step)


# Global singleton accessor
_cli_output_instance: Optional[CLIOutput] = None
_cli_output_lock = Lock()


def get_cli_output() -> CLIOutput:
    global _cli_output_instance
    with _cli_output_lock:
        if _cli_output_instance is None:
            _cli_output_instance = CLIOutput()
        return _cli_output_instance
