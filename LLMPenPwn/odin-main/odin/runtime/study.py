import logging
import signal
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, TypeVar

from docker.models.containers import Container

from ..core.container import ContainerManager
from .session import Session

TSession = TypeVar("TSession", bound=Session)

class Study(ABC):
    """Base class for analyses that manage container-backed sessions."""

    @classmethod
    def add_arguments(cls, parser) -> None:
        pass

    def __init__(self, code_path: Path, *, image: str | None = None, compose_file: Path | None = None):
        self.code_path = code_path
        self.image = image or ContainerManager.DEFAULT_IMAGE
        self._container: Container | None = None
        self._sessions: list[Session] = []
        self.logger = logging.getLogger(self.__class__.__name__)

        self.compose_file = compose_file
        self._orig_sigint_handler = None
        self._sigint_installed = False

        if threading.current_thread() is threading.main_thread():
            self._orig_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._on_sigint)
            self._sigint_installed = True
        else:
            self.logger.debug("Skipping SIGINT handler setup outside main thread")

    def __enter__(self):
        self.logger.info("Starting analysis study in %s", self.code_path)
        self._container = ContainerManager.run(
            self.image,
            self.code_path,
            compose_file=self.compose_file,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.logger.info("Tearing down analysis study...")
        for session in list(self._sessions):
            try:
                session.close()
            except Exception:
                self.logger.exception("Error while closing session %s", session.name)

        if self._container is not None:
            ContainerManager.stop(self._container)
            self._container = None

        if self._sigint_installed:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self._orig_sigint_handler)
            else:
                self.logger.debug("Skipping SIGINT handler restore outside main thread")
            self._sigint_installed = False
        return False

    def new_session(self, session_cls: Type[TSession], *args, **kwargs) -> TSession:
        if self._container is None:
            raise RuntimeError("Study has not been entered with `with` yet.")

        session = session_cls(self._container, *args, **kwargs)
        self._sessions.append(session)
        self.logger.debug("Created session %s", session.name)
        return session

    def _detect_compose_file(self, code_path: Path) -> Path | None:
        compose_names = ["docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"]

        for compose_name in compose_names:
            compose_file = code_path / compose_name
            if compose_file.exists() and compose_file.is_file():
                self.logger.info("Auto-detected compose file: %s", compose_file)
                return compose_file

        self.logger.debug("No compose file found in %s", code_path)
        return None

    def _on_sigint(self, signum, frame):
        self.logger.warning("KeyboardInterrupt - forcing study shutdown...")
        if self._container is not None:
            try:
                ContainerManager.stop(self._container)
            except Exception:
                self.logger.exception("Failed to stop container during SIGINT handling")
            finally:
                self._container = None
        raise KeyboardInterrupt

    @abstractmethod
    def run(self):
        pass
