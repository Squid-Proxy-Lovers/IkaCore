import json
import logging
import signal
import threading
from pathlib import Path
from typing import Any

from docker.models.containers import Container

from ..core.container import CompletedExec, ContainerManager
from ..tools import ContainerBashTool, RemotePythonTool, RemoteSageMathTool


class EnvironmentError(Exception):
    pass


class EnvironmentNotStartedError(EnvironmentError):
    pass


class EnvironmentStoppedError(EnvironmentError):
    pass


class EnvironmentStartupError(EnvironmentError):
    pass


class Environment:
    """
    Manages Odin container lifecycle and provides execution context.

    Can be used as a context manager (automatic cleanup) or manually (explicit start/stop control).
    """

    def __init__(
        self,
        code_path: Path | str,
        *,
        image: str | None = None,
        deploy_service: bool = False,
        compose_file: Path | None = None,
        readonly_src: str = "/opt/resources",
        workdir: str = "/working",
    ) -> None:
        if isinstance(code_path, str):
            code_path = Path(code_path)
        self.code_path: Path = code_path.resolve()
        self.image: str | None = image or ContainerManager.DEFAULT_IMAGE
        self.deploy_service: bool = bool(deploy_service)
        self.compose_file: Path | None = Path(compose_file) if compose_file else None
        self.readonly_src: str = readonly_src
        self.workdir: str = workdir

        self._container: Container | None = None
        self._started: bool = False
        self._stopped: bool = False
        self._orig_sigint_handler = None
        self._sigint_installed = False
        self.logger = logging.getLogger("odin.runtime.environment")

    def _detect_compose_file(self, code_path: Path) -> Path | None:
        compose_names = [
            "docker-compose.yml",
            "docker-compose.yaml",
            "compose.yml",
            "compose.yaml",
        ]
        for name in compose_names:
            candidate = code_path / name
            if candidate.exists() and candidate.is_file():
                self.logger.info("Auto-detected compose file: %s", candidate)
                return candidate
        self.logger.debug("No compose file found in %s", code_path)
        return None

    def start(self) -> "Environment":
        if self._started and not self._stopped:
            self.logger.debug("Environment already started")
            return self

        if self._stopped:
            raise EnvironmentStoppedError(
                "Environment has been stopped and cannot be restarted. Create a new instance."
            )

        compose_to_use: Path | None = None
        if self.deploy_service:
            compose_to_use = self.compose_file or self._detect_compose_file(self.code_path)
            if compose_to_use is None:
                raise EnvironmentStartupError(
                    "deploy_service=True but no compose file found; provide --compose-file or add a compose file to the code path"
                )

        try:
            self._container = ContainerManager.run(
                self.image,
                self.code_path,
                readonly_src=self.readonly_src,
                workdir=self.workdir,
                compose_file=compose_to_use,
            )
        except Exception as e:
            raise EnvironmentStartupError(f"Failed to start environment: {e}") from e

        # Install SIGINT handler (main thread only)
        if threading.current_thread() is threading.main_thread():
            self._orig_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._on_sigint)
            self._sigint_installed = True
        else:
            self.logger.debug("Skipping SIGINT handler (not main thread)")

        self._started = True
        return self

    def stop(self) -> None:
        if self._stopped:
            self.logger.debug("Environment already stopped")
            return
        if self._container is not None:
            try:
                ContainerManager.stop(self._container)
            except Exception as e:
                self.logger.exception("Error stopping container: %s", e)
            finally:
                self._container = None
        # Restore SIGINT
        if self._sigint_installed:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self._orig_sigint_handler)
            self._sigint_installed = False
        self._stopped = True

    def __enter__(self) -> "Environment":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    @property
    def container(self) -> Container:
        if not self._started:
            raise EnvironmentNotStartedError("Call start() or use 'with Environment(...)' first")
        if self._stopped:
            raise EnvironmentStoppedError("Environment has been stopped")
        if self._container is None:
            raise EnvironmentStartupError("Container is not available (failed to start?)")
        return self._container

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped and self._container is not None

    def exec(self, cmd: str | list[str], **kwargs: Any) -> CompletedExec:
        return ContainerManager.exec(self.container, cmd, **kwargs)

    def get_python_port(self) -> int:
        ports = self.container.ports.get("8888/tcp")
        if not ports:
            raise EnvironmentStartupError("No port binding found for 8888/tcp")
        try:
            return int(ports[0]["HostPort"])
        except Exception as e:
            raise EnvironmentStartupError(f"Failed to parse Jupyter port: {e}") from e

    def get_sagemath_port(self) -> int:
        ports = self.container.ports.get("8889/tcp")
        if not ports:
            import time
            time.sleep(100)
            raise EnvironmentStartupError("No port binding found for 8889/tcp")
        try:
            return int(ports[0]["HostPort"])
        except Exception as e:
            raise EnvironmentStartupError(f"Failed to parse Jupyter port: {e}") from e

    def get_service_ports(self) -> dict[str, list[dict[str, Any]]]:
        label_value = getattr(self.container, "labels", None)
        # docker-py exposes labels under .labels
        label_json = {}
        if isinstance(label_value, dict):
            raw = label_value.get("odin.service_ports", "{}")
        else:
            raw = self.container.labels.get("odin.service_ports", "{}")
        try:
            label_json = json.loads(raw)
        except Exception:
            self.logger.error("Failed to parse odin.service_ports label")
            label_json = {}
        return label_json

    def get_bash_tool(self, **overrides) -> ContainerBashTool:
        # ContainerBashTool only expects container in constructor
        return ContainerBashTool(container=self.container)

    def get_python_tool(self, **overrides) -> RemotePythonTool:
        host = overrides.get("host", "127.0.0.1")
        port = overrides.get("port", self.get_python_port())
        exec_timeout_s = overrides.get("exec_timeout_s", 60 * 5)
        return RemotePythonTool(host=host, port=port, exec_timeout_s=exec_timeout_s)

    def get_sagemath_tool(self, **overrides) -> RemoteSageMathTool:
        host = overrides.get("host", "127.0.0.1")
        port = overrides.get("port", self.get_sagemath_port())
        exec_timeout_s = overrides.get("exec_timeout_s", 60 * 5)
        return RemoteSageMathTool(host=host, port=port, exec_timeout_s=exec_timeout_s)

    def get_environment_prompt(self) -> str:
        services = self.get_service_ports()
        svc_lines = []
        if services:
            for svc, ports in services.items():
                for p in ports:
                    lport = p.get("orig_host_port")
                    if lport is not None:
                        svc_lines.append(f"- Service {svc} at localhost:{lport}")
        lines = [
            "<environment>",
            "- Code loaded at `/opt/resources/` (read-only)",
            "- Current working directory is `/working/` (read-write)",
        ]
        lines.extend(svc_lines)
        lines.append("</environment>")
        return "\n".join(lines)

    def _on_sigint(self, signum, frame):
        self.logger.warning("KeyboardInterrupt - stopping environment...")
        try:
            self.stop()
        finally:
            raise KeyboardInterrupt


