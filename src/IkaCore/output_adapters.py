"""Optional CLI and file subscribers for structured runtime events."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from .events import RuntimeEvent, RuntimeOutput, _shared_cli

if TYPE_CHECKING:
    from .config import OutputConfig


class CLIAdapter:
    def __init__(self) -> None:
        from .cli_output import CLIOutput
        self.cli: CLIOutput = CLIOutput.owned()

    def __call__(self, event: RuntimeEvent) -> None:
        shared = _shared_cli.get()
        if shared is not None and shared is not self:
            shared(event)
            return
        self.cli.set_instance_id(event.instance_id)
        name = {"display": "emit", "parallel_start": "start_parallel", "parallel_end": "end_parallel"}.get(event.kind, event.kind)
        method = getattr(self.cli, name, None)
        if method is not None:
            method(**event.data)

    def is_buffering(self) -> bool:
        return self.cli.is_buffering()

    def flush(self) -> None:
        self.cli.end_parallel()


class FileLogAdapter:
    def __init__(self, config: OutputConfig) -> None:
        from .logging_utils import IkaLogger
        self.logger: IkaLogger = IkaLogger(config.logging_level, config.logging_file,
                                show_usage_level0=config.show_usage_level0)

    def __call__(self, event: RuntimeEvent) -> None:
        method = getattr(self.logger, "log_" + event.kind, None)
        if method is not None:
            method(**event.data)

    def flush(self) -> None:
        self.logger.flush()

    def shutdown(self) -> None:
        self.logger.shutdown()


def write_stdout(rendered: str) -> None:
    sys.stdout.write(rendered)
    sys.stdout.flush()


def make_output(config: OutputConfig) -> RuntimeOutput:
    adapters: list[Any] = list(config.subscribers)
    cli = CLIAdapter() if config.cli else None
    if cli is not None:
        adapters.append(cli)
    if config.logging_level:
        adapters.append(FileLogAdapter(config))
    return RuntimeOutput(tuple(adapters), cli_adapter=cli)


class UsageAdapter:
    """Optional per-step usage display; subscribe explicitly to avoid extra default output."""
    def __init__(self, writer: Any = None) -> None:
        self.writer: Any = writer or write_stdout

    def __call__(self, event: RuntimeEvent) -> None:
        if event.kind == 'step' and event.data.get('usage'):
            self.writer(f"[USAGE] stage={event.data['stage_name']} tokens={event.data['usage']} cost={event.data['cost']}\n")
