"""Execution trace writer for root and sub-agent runs."""

from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import IkaCore.cli_output as cli_output_module
from IkaCore.cli_output import OutputContext, OutputType, set_output_sink


class ExecutionTraceWriter:
    """Write rendered CLI output to per-run log files."""

    def __init__(self, run_dir: Path, root_name: str = "root"):
        self.run_dir = run_dir
        self.root_name = root_name
        self.root_log_path = run_dir / "root.log"
        self._lock = threading.Lock()
        self._installed = False
        self._previous_sink: Optional[Callable[[OutputContext, str], None]] = None
        self._subagent_logs: dict[str, Path] = {}

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.root_log_path.touch(exist_ok=True)

    @classmethod
    def create_default(cls, base_dir: Path, root_name: str = "root") -> "ExecutionTraceWriter":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(base_dir / stamp, root_name=root_name)

    def install(self) -> None:
        if self._installed:
            return
        self._previous_sink = getattr(cli_output_module, "_output_sink")
        set_output_sink(self._sink)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        set_output_sink(self._previous_sink)
        self._installed = False

    def _sink(self, ctx: OutputContext, rendered: str) -> None:
        if self._previous_sink is not None:
            self._previous_sink(ctx, rendered)

        target_path = self._get_scoped_log_path(ctx)
        with self._lock:
            self.root_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.root_log_path.open("a", encoding="utf-8") as handle:
                handle.write(rendered)
            if target_path != self.root_log_path:
                with target_path.open("a", encoding="utf-8") as handle:
                    handle.write(rendered)

    def append_root_text(self, text: str) -> None:
        with self._lock:
            with self.root_log_path.open("a", encoding="utf-8") as handle:
                handle.write(text)

    def _get_scoped_log_path(self, ctx: OutputContext) -> Path:
        agent_chain = self._extract_agent_chain(ctx)
        if len(agent_chain) <= 1:
            return self.root_log_path

        subagent_key = "__".join(agent_chain[1:])
        if subagent_key not in self._subagent_logs:
            started_at = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{self._sanitize(subagent_key)}-{started_at}.log"
            self._subagent_logs[subagent_key] = self.run_dir / filename
        return self._subagent_logs[subagent_key]

    def _extract_agent_chain(self, ctx: OutputContext) -> list[str]:
        chain = list(ctx.hierarchy_chain or [])
        if ctx.output_type in {OutputType.TOOL_CALL, OutputType.TOOL_RESULT} and chain:
            return chain[:-1]
        return chain

    @staticmethod
    def _sanitize(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
        return cleaned or "subagent"
