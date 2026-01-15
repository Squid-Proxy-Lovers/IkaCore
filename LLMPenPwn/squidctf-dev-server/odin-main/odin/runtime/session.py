import logging
import uuid
from typing import Any

from docker.models.containers import Container

from ..core.container import CompletedExec, ContainerManager


class Session:
    """Lightweight wrapper around a docker container for command execution."""

    def __init__(self, container: Container, *, name: str | None = None) -> None:
        self._container = container
        self.name = name or f"session-{uuid.uuid4().hex[:6]}"
        self.last_result: CompletedExec | None = None
        self.logger = logging.getLogger(f"odin.session.{self.name}")

    def exec(self, cmd: str | list[str], **kwargs: Any) -> CompletedExec:
        result = ContainerManager.exec(self._container, cmd, **kwargs)
        self.last_result = result
        return result

    def run(self):
        self.logger.debug("Base Session.run() does nothing.")

    def close(self):
        self.logger.debug("Session %s closed", self.name)
