import io
import logging
import math
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docker.models.containers import Container

from ..core.container import ContainerManager
from .base import Tool

_LOG = logging.getLogger(__name__)

class ThinkTool(Tool):
    name = "think"
    description = textwrap.dedent("""
    Use the tool to think about something that will be user visible. It will not obtain new information or change the database, but just append the thought to the log.
    Use it when complex reasoning or some cache memory is needed.

    IMPORTANT: Remember that this is user visible, so do not use it for internal reasoning that should not be shown to the user.

    Args:
        thought (str): The thought to append to the log.

    Returns:
        None
    """)
    inputs = {
        "thought": {
            "type": "string",
            "description": "The thought to append to the log."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, thought: str) -> None:
        _LOG.info("Thinking: %s", thought)
        return

class BashTool(Tool):
    name = "shell"
    description = "Runs a shell command and returns its output."
    inputs = {
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The command to execute",
            "required": True
        },
        "workdir": {
            "type": "string",
            "description": "The working directory to execute the command in",
            "required": False,
        },
        "timeout_ms": {
            "type": "integer",
            "description": "The timeout for the command in milliseconds",
            "required": False,
        }
    }
    output_type = "string"
    tool_guidelines = textwrap.dedent("""\
    ## Shell commands

    When using the shell, you must adhere to the following guidelines:

    - When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)
    - Read files in chunks with a max chunk size of 250 lines. Do not use python scripts to attempt to output larger chunks of a file. Command line output will be truncated after 10 kilobytes or 256 lines of output, regardless of the command used.
    - Do not call `python` or `python3` directly from the shell. Instead, use the python tool for any python code execution.
    """).strip()

    def __init__(self):
        super().__init__()

    def _exec(self, command: List[str], workdir: str, timeout_ms: int) -> Tuple[int, bytes, bytes]:
        try:
            result = subprocess.run(
                command, capture_output=True, text=True,
                timeout=timeout_ms / 1000, cwd=workdir,
                env={**os.environ, "TERM": "dumb"}
            )
            return result.returncode, result.stdout.encode(), result.stderr.encode()
        except subprocess.TimeoutExpired:
            _LOG.error(f"Command timed out after {timeout_ms / 1000} seconds")
            return -1, b"", b""

    def forward(self, command: List[str], workdir: str = "/working", timeout_ms: int = 30_000) -> str:
        rc, stdout, stderr = self._exec(command, workdir, timeout_ms)

        if rc == -1:
            return "[Command timed out]"
        if rc == 127:
            return (stderr.decode().strip()
                    or f"[Command not found: {' '.join(command)}]")

        output = ""
        if stdout:
            output += stdout.decode(errors="replace").strip()
        if stderr:
            if output:
                output += "\n"
            output += "STDERR: " + stderr.decode(errors="replace").strip()

        # Trim output (10KB or 256 lines), preserving summary behavior
        over_256_lines = len(output.splitlines()) > 256
        over_10kb = len(output.encode()) > 10 * 1024
        if over_10kb:
            output = output.encode()[:10 * 1024].decode(errors="replace") + "\n...[output truncated, 10KB max]..."
        elif over_256_lines:
            output = "\n".join(output.splitlines()[:256]) + "\n...[output truncated, 256 lines max]..."

        if rc != 0:
            output += f"\n[Command exited with code {rc}]"

        return output or "[Command completed with no output]"

class ContainerBashTool(BashTool):
    def __init__(self, container: Container):
        super().__init__()
        self.container = container

    def _exec(self, command: List[str], workdir: str, timeout_ms: int) -> Tuple[int, bytes, bytes]:
        try:
            secs = max(1, math.ceil(timeout_ms / 1000))
            cmd_str = " ".join(shlex.quote(arg) for arg in command)
            wrapped = [
                "sh", "-lc",
                f"exec timeout --signal=TERM --kill-after=2s {secs}s {cmd_str}"
            ]

            result = self.container.exec_run(
                wrapped,
                workdir=workdir,
                demux=True,
                environment={"TERM": "dumb"}
            )
            rc = result.exit_code
            stdout, stderr = (result.output or (b"", b""))

            if rc == 124:
                note = f"\n[timeout after {secs}s]".encode()
                return -1, stdout or b"", (stderr or b"") + note

            return rc, stdout or b"", stderr or b""

        except Exception as e:
            _LOG.error(f"Error executing in container: {e}")
            return -1, b"", f"[exec error: {e}]".encode()