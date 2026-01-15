import io
import json
import logging
import os
import re
import shlex
import subprocess
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from docker.models.containers import Container
from websocket import (WebSocketConnectionClosedException,
                       WebSocketTimeoutException, create_connection)

from ..core.container import ContainerManager
from .base import Tool

_LOG = logging.getLogger(__name__)

class RemoteDecompileBinaryTool(Tool):
    name = "decompile_binary"
    description = textwrap.dedent("""
    Decompile a binary file using Ghidra and outputs the decompiled code to a working directory.
    """).strip()
    inputs = {
        "binary_path": {
            "type": "string",
            "description": "Path to the binary file to decompile (absolute path inside the container)"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory to output the decompiled code (absolute path inside the container, must be writable)"
        }
    }
    output_type = "string"
    tool_guidelines = textwrap.dedent("""\
    ## Decompile Binary Tool

    For any targets that have ELF or PE binaries, you can use this tool to decompile them using headless Ghidra into c code.

    When using the decompile binary tool, keep in mind the following:
    - Trace the output of the code execution from main/start/entrypoint functions. Usually the actual decompilation code will be at the end large code file created by Ghidra.
    - Trace code flow step-by-step to understand how the binary works.
    """).strip()

    def __init__(self, container: Container):
        super().__init__()
        self._container = container

    def _get_python_code(self, binary_path: str, output_dir: str) -> str:
        code = textwrap.dedent(f"""
        import subprocess
        import pathlib
        import os

        binary_path = r\"\"\"{binary_path}\"\"\"
        output_dir = r\"\"\"{output_dir}\"\"\"

        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Binary file not found: {{binary_path}}")

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        project_name = "temp_project"
        project_dir = "/tmp/" + project_name

        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)

        decompile_command = [
            "/opt/ghidra/support/analyzeHeadless",
            project_dir,
            project_name,
            "-import",
            binary_path,
            "-scriptPath",
            "/opt/ghidra/Decompile",
            "-postScript",
            "/opt/ghidra/Decompile/DecompilerExplorer.java",
            output_dir + "/out"
        ]

        env = os.environ.copy()

        decomp = subprocess.run(decompile_command, capture_output=True, env=env)
        if decomp.returncode != 0:
            raise RuntimeError(f"Decompilation failed: {{decomp.stdout.decode()}}\\n{{decomp.stderr.decode()}}")

        print("Decompilation completed successfully.")
        """)
        return code

    def forward(self, binary_path: str, output_dir: str) -> str:
        python_code = self._get_python_code(binary_path, output_dir)
        res = self._container.exec_run(
            cmd=["python3", "-c", python_code],
            workdir="/working",
            demux=True,
        )
        stdout, stderr = res.output
        if res.exit_code != 0:
            return f"Error during decompilation (exit code {res.exit_code}): {stderr.decode() if stderr else 'Unknown error'}"

        return stdout.decode() if stdout else "Decompilation completed, but no output."
