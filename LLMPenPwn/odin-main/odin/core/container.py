import json
import logging
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from docker.errors import NotFound
from docker.models.containers import Container

import docker

_LOG = logging.getLogger(__name__)
_CLIENT: docker.DockerClient | None = None
COMPOSE_LABEL = "com.docker.compose.project"

def _get_client() -> docker.DockerClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = docker.from_env()
    return _CLIENT

class ServiceContainerException(Exception):
    pass

def _compose_up(compose_file: Path, project: str, cwd: Path = None) -> None:
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "-p", project, "up", "-d"],
        cwd=cwd or compose_file.parent,
        check=True,
    )

def _compose_down(compose_file: Path, project: str, cwd: Path = None) -> None:
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "-p", project,
             "down", "--remove-orphans", "-v", "--rmi", "all"],
            cwd=cwd or compose_file.parent,
            check=True,
        )
    except subprocess.CalledProcessError:
        _LOG.warning("compose down failed for project %s", project)

def _running_containers_for_project(client: docker.DockerClient, project: str) -> list[Container]:
    return client.containers.list(filters={"label": f"{COMPOSE_LABEL}={project}"})

def _load_compose(compose_file: Path) -> Dict[str, Any]:
    with open(compose_file, "r") as f:
        return yaml.safe_load(f) or {}

def _parse_exposed_container_port(ports_section: str) -> Optional[Tuple[int, Optional[int]]]:
    if not ports_section:
        return None

    def _parse_str(p: str) -> Optional[Tuple[int, Optional[int]]]:
        p = p.strip()
        p = p.split("/", 1)[0]
        parts = p.split(":")
        container_raw = parts[-1]
        host_raw = parts[-2] if len(parts) >= 2 else None

        def _to_int(token: Optional[str]) -> Optional[int]:
            if token is None:
                return None
            token = token.strip()
            if "-" in token:
                token = token.split("-", 1)[0]
            try:
                return int(token)
            except (TypeError, ValueError):
                return None

        container = _to_int(container_raw)
        if container is None:
            return None
        host = _to_int(host_raw) if host_raw is not None else None
        return (container, host)

    return _parse_str(ports_section)

def _write_ephemeral_compose(base_compose: Path, *, cport_mapping: List[Tuple[str, Any]]) -> Path:
    data = _load_compose(base_compose)

    for svc_name, ports in cport_mapping:
        data['services'][svc_name]["ports"] = [{
            "target": cport,
            "published": 0,
            "protocol": "tcp",
        } for cport, _ in ports if cport is not None]

    for svc_name in data.get('services', {}):
        if 'container_name' in data['services'][svc_name]:
            data['services'][svc_name].pop('container_name', None)

    tmp = Path(tempfile.mkstemp(prefix=f".odin-service-", suffix=".compose.yml", dir=base_compose.parent)[1])
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    _LOG.info("Generated ephemeral compose: %s", tmp)
    return tmp

def _discover_host_ports_for_project(compose_project: str) -> Dict[str, List[Dict[str, Any]]]:
    client = _get_client()
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for c in _running_containers_for_project(client, compose_project):
        svc = c.labels.get("com.docker.compose.service", c.name)
        ports = (c.attrs.get("NetworkSettings", {}).get("Ports")) or {}
        for container_port, bindings in ports.items():
            if not bindings:
                continue
            for b in bindings:
                try:
                    host_port = int(b.get("HostPort", 0))
                except (TypeError, ValueError):
                    continue
                mapping.setdefault(svc, []).append({
                    "container_port": container_port,
                    "host_ip": b.get("HostIp") or "",
                    "host_port": host_port,
                })
    return mapping

@dataclass(slots=True)
class CompletedExec:
    stdout: str
    stderr: str
    exit_code: int

    def __bool__(self) -> bool:
        return self.exit_code == 0

    @property
    def returncode(self) -> int:
        return self.exit_code

class ContainerManager:
    DEFAULT_IMAGE = "odin-base:latest"
    DEFAULT_SAGE_IMAGE = "odin-base:sage"

    @classmethod
    def run(cls,
        image: Optional[str],
        code_path: Path,
        *,
        readonly_src: str = "/opt/resources",
        workdir: str = "/working",
        compose_file: Optional[Path] = None,
    ) -> Container:
        # TODO find containers that may need cleanup? or rely on python to always close

        image = image or cls.DEFAULT_IMAGE
        if not code_path.exists():
            raise FileNotFoundError(code_path)
        code_path = code_path.resolve()

        client = _get_client()

        compose_project: Optional[str] = None
        compose_override: Optional[Path] = None
        discovered_ports: Dict[str, List[Dict[str, Any]]] = {}

        cport_mapping: List[Tuple[str, List[Tuple[int, Optional[int]]]]] = []
        if compose_file and compose_file.exists():
            compose_file = compose_file.resolve()
            base = _load_compose(compose_file)

            cport_mapping = [
                (
                    svc, [_parse_exposed_container_port(port) for port in base.get("services", {}).get(svc, {}).get("ports", [])]
                )
                for svc in (base.get("services") or {}).keys()
                if base.get("services", {}).get(svc, {}).get("ports")
            ]
            if not cport_mapping:
                raise RuntimeError(f"Could not detect container port for service '{compose_file}'")

            compose_project = f"{compose_file.parent.name}_{uuid.uuid4().hex[:16]}".lower()
            compose_override = _write_ephemeral_compose(compose_file, cport_mapping=cport_mapping)

            _compose_up(compose_override, compose_project, cwd=compose_file.parent)
            discovered_ports = _discover_host_ports_for_project(compose_project)
            # only keep keys container_port & host_port then deduplicate host ports
            discovered_ports = {
                svc: list(set([
                    (p["container_port"], p["host_port"])
                    for p in plist
                ]))
                for svc, plist in discovered_ports.items()
            }
            discovered_ports = {
                svc: [{"container_port": cp, "host_port": hp} for cp, hp in plist]
                for svc, plist in discovered_ports.items()
            }

            _LOG.info("Discovered host ports for project %s: %s", compose_project, discovered_ports)
            host_ports = {
                svc: list(set([p["host_port"] for p in plist]))
                for svc, plist in discovered_ports.items()
            }

            if len(host_ports) == 0:
                raise ServiceContainerException("No services with exposed ports found in override compose file.")

            if len(host_ports) != len(cport_mapping):
                raise ServiceContainerException("Mismatch between discovered host ports and expected container ports.")

            if not all(k in host_ports for k, _ in cport_mapping):
                raise ServiceContainerException("Not all expected services have discovered host ports.")

        work_volume_name = f"odin_work_{uuid.uuid4().hex[:16]}"
        _LOG.debug("Creating anonymous work volume %s", work_volume_name)
        client.volumes.create(name=work_volume_name)

        volumes: Dict[str, Dict[str, Any]] = {
            str(code_path): {"bind": readonly_src, "mode": "ro"},
            work_volume_name: {"bind": workdir, "mode": "rw"},
        }

        extra_hosts = {
            "host.docker.internal": "host-gateway",
        }

        _LOG.info("Launching Odin container '%s' (cport_mapping=%s)...",
                  image, json.dumps(cport_mapping))

        service_ports = {}
        for svc_name, ports in cport_mapping:
            d_ports = discovered_ports.get(svc_name, [])
            if len(d_ports) != len(ports):
                raise ServiceContainerException(
                    f"Mismatch in port count for service '{svc_name}': expected {len(ports)}, discovered {len(d_ports)}"
                )

            service_ports[svc_name] = []
            for (cport, hport), real_hport in zip(ports, d_ports):
                if hport is None:
                    continue

                service_ports[svc_name].append({
                    "container_port": cport,
                    "orig_host_port": hport,
                    "real_host_port": real_hport["host_port"],
                })

        container = client.containers.run(
            image,
            command=None,
            detach=True,
            tty=False,
            stdin_open=False,
            read_only=False,
            volumes=volumes,
            auto_remove=False,
            ports={'8888/tcp': None} if image == cls.DEFAULT_IMAGE else {'8888/tcp': None, '8889/tcp': None},
            extra_hosts=extra_hosts,
            labels={
                "odin.compose_project": compose_project or "",
                "odin.compose_file": str(compose_file or ""),
                "odin.compose_ephemeral": str(compose_override or ""),
                "odin.service_ports": json.dumps(service_ports),
                "odin.work_volume": work_volume_name,
            },
        )

        retries = 0
        while container.status != "running" and retries < 60:
            _LOG.info("Waiting for Odin container %s to start, status: %s", container.short_id, container.status)
            time.sleep(1)
            container.reload()
            retries += 1

        if service_ports:
            for svc_name, ports in service_ports.items():
                for port_info in ports:
                    _LOG.info(
                        "Forwarding local port %d to service '%s' container port %d (host port %d)",
                        port_info["orig_host_port"], svc_name,
                        port_info["container_port"], port_info["real_host_port"]
                    )
                    _start_local_forwarder(
                        container,
                        local_port=port_info["orig_host_port"],
                        remote_host="host.docker.internal",
                        remote_port=port_info["real_host_port"],
                    )

        return container

    @classmethod
    def exec(
        cls,
        container: Container,
        cmd: str | list[str],
        *,
        workdir: str = "/working",
        timeout: Optional[float] = None,
        demux: bool = False,
    ) -> CompletedExec:
        # TODO: fix timeout
        cmd_display = " ".join(shlex.quote(c) for c in cmd) if isinstance(cmd, list) else cmd
        _LOG.debug("[container %s] $ %s", container.short_id, cmd_display)

        result = container.exec_run(
            cmd,
            workdir=workdir,
            stdout=True,
            stderr=True,
            demux=demux,
            tty=False,
            privileged=False,
            user="root",
            socket=False,
            stream=False,
            environment={"TERM": "dumb"},
        )

        exit_code: int = result.exit_code
        raw_output = result.output

        stdout = ""
        stderr = ""
        if demux and isinstance(raw_output, tuple):
            stdout_bytes, stderr_bytes = raw_output
            stdout = (stdout_bytes or b"").decode()
            stderr = (stderr_bytes or b"").decode()
        elif isinstance(raw_output, bytes):
            stdout = raw_output.decode()

        _LOG.debug(
            "[container %s] exit %s | stdout=%dB stderr=%dB",
            container.short_id,
            exit_code,
            len(stdout.encode()),
            len(stderr.encode()),
        )
        return CompletedExec(stdout=stdout, stderr=stderr, exit_code=exit_code)

    @classmethod
    def stop(cls, container: Container) -> None:
        if container is None:
            return

        client = _get_client()

        compose_project = container.labels.get("odin.compose_project") or ""
        compose_file_str = container.labels.get("odin.compose_file") or ""
        compose_ephemeral = container.labels.get("odin.compose_ephemeral") or ""
        work_volume = container.labels.get("odin.work_volume") or ""

        try:
            _LOG.info("Stopping Odin container %s…", container.short_id)
            container.kill()
        except NotFound:
            _LOG.debug("Odin container already removed")
        finally:
            try:
                container.remove(force=True)
            except NotFound:
                pass

        if work_volume:
            _LOG.debug("Removing work volume %s", work_volume)
            try:
                client.volumes.get(work_volume).remove(force=True)
            except NotFound:
                _LOG.debug("Volume already removed")

        if compose_project and compose_file_str:
            compose_dir = Path(compose_file_str).resolve().parent
            if compose_ephemeral and Path(compose_ephemeral).exists():
                compose_file = Path(compose_ephemeral)
            _compose_down(compose_file, compose_project, cwd=compose_dir)

        if compose_ephemeral:
            try:
                Path(compose_ephemeral).unlink(missing_ok=True)
                _LOG.debug("Removed ephemeral compose file %s", compose_ephemeral)
            except Exception as e:
                _LOG.warning("Failed to remove ephemeral compose file %s: %s", compose_ephemeral, e)

def _start_local_forwarder(container: Container, local_port: int, remote_host: str, remote_port: int) -> None:
    cmd = [
        "sh", "-lc",
        (
            f"nohup socat TCP-LISTEN:{local_port},fork,bind=127.0.0.1,reuseaddr TCP:{remote_host}:{remote_port} "
            "> /dev/null 2>&1 &"
        )
    ]
    _ = ContainerManager.exec(container, cmd)