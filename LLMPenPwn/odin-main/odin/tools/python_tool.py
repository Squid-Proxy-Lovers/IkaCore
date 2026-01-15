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

ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07\x1B]*(?:\x07|\x1B\\))')

class RemotePythonTool(Tool):
    name = "python"
    description = textwrap.dedent("""
    Executes python code in a stateful Jupyter notebook environment.
    IMPORTANT: Ensure code is ONLY python code.

    Args:
        code (str): Python code to execute

    Returns:
        Execution result or error message
    """)
    inputs = {
        "code": {
            "type": "string",
            "description": "Python code to execute"
        }
    }
    output_type = "string"
    tool_guidelines = textwrap.dedent("""\
    ## Python tool

    This is the most powerful tool have access to, please use it often in the following senarios:

    - Developing a proof-of-concept exploit. Ensure to test exploits against a network targets if provided. Favor using python requests package for HTTP/HTTPS targets and pwntools for other socket connections
    - Testing and validating attacks. You have access already to standard libraries such as `numpy`, `pycryptodome`, and `scikit-learn` to help you iterate faster.

    IMPORTANT: Remember that the python tool is executed in a jupyter notebook environment and it will keep state between calls. You can define functions, import libraries, and store variables to use in future calls.
    """).strip()

    def __init__(self, host: str = "127.0.0.1", port: int = 8888, exec_timeout_s: int = 60 * 5):
        super().__init__()
        self.host = host
        self.port = port
        self.exec_timeout_s = exec_timeout_s

        retries = 30
        while retries > 0:
            try:
                self.base_url = f"http://{host}:{port}"
                r = requests.post(f"{self.base_url}/api/kernels")
                if r.status_code != 201:
                    error_details = {
                        "status_code": r.status_code,
                        "headers": dict(r.headers),
                        "url": r.url,
                        "body": r.text,
                        "request_method": r.request.method,
                        "request_headers": dict(r.request.headers),
                        "request_body": r.request.body,
                    }
                    _LOG.error(f"Failed to create kernel. Details: {json.dumps(error_details, indent=2)}")
                    raise RuntimeError(f"Failed to create kernel: Status {r.status_code}\nResponse: {r.text}") from None

                self.kernel_id = r.json()["id"]
                ws_url = f"ws://{host}:{port}/api/kernels/{self.kernel_id}/channels"
                self.ws = create_connection(ws_url)
                self.ws.settimeout(1.0)
                _LOG.info("Kernel %s is running", self.kernel_id)
                self._disable_colors_in_kernel()
            except Exception as e:
                _LOG.error(f"Error initializing Jupyter kernel: {e}. Retries left: {retries-1}. Sleeping 10 seconds before retrying...")
                retries -= 1
                time.sleep(10)
                if retries == 0:
                    raise RuntimeError("Failed to initialize Jupyter kernel after multiple retries.") from e
            else:
                break

    def _clean_text(self, s: str) -> str:
        if not s:
            return s
        s = ANSI_RE.sub('', s)
        s = ''.join(part.split('\r')[-1] for part in s.splitlines(True))
        s = re.sub(r'.\x08', '', s)
        return s

    def _reconnect_websocket(self):
        _LOG.info("Attempting websocket reconnection...")
        if hasattr(self, 'ws') and self.ws:
            try:
                self.ws.close()
            except Exception as close_err:
                _LOG.warning(f"Error closing old websocket: {close_err}")

        ws_url = f"ws://{self.host}:{self.port}/api/kernels/{self.kernel_id}/channels"
        try:
            self.ws = create_connection(ws_url, timeout=30)
            self.ws.settimeout(1.0)
            _LOG.info("Successfully reconnected to kernel websocket.")
        except Exception as reconn_err:
            _LOG.error(f"Failed to reconnect websocket: {reconn_err}")
            raise RuntimeError(f"Failed to reconnect websocket: {reconn_err}") from reconn_err

    def _ensure_ws_connection(self):
        if not hasattr(self, 'ws') or not self.ws or not self.ws.connected:
            _LOG.info("Websocket connection found closed or uninitialized. Reconnecting...")
            self._reconnect_websocket()
            return

        try:
            self.ws.ping()
        except (WebSocketConnectionClosedException, ConnectionResetError, BrokenPipeError, AttributeError) as e:
            _LOG.info(f"Websocket ping failed ({type(e).__name__}), connection assumed lost. Reconnecting...")
            self._reconnect_websocket()
        except Exception as e:
            _LOG.warning("Attempting reconnection after unexpected ping error...")
            try:
                self._reconnect_websocket()
            except Exception as reconn_e:
                _LOG.error(f"Reconnection failed after unexpected ping error: {reconn_e}")
                raise

    def _interrupt_kernel(self):
        try:
            requests.post(f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt", timeout=5)
            _LOG.warning("Kernel %s interrupted due to timeout.", self.kernel_id)
        except Exception as e:
            _LOG.error(f"Interrupt request failed: {e}")

    def _disable_colors_in_kernel(self):
        setup = """
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip:
        ip.colors = "NoColor"
except Exception:
    pass
"""
        try:
            self.run_code_raise_errors(setup)
        except Exception:
            pass

    def run_code_raise_errors(self, code_action: str, return_final_answer: bool = False) -> str:
        max_retries = 3
        attempt = 0

        while attempt <= max_retries:
            try:
                self._ensure_ws_connection()

                if attempt != 0:
                    _LOG.warning(f"Retrying code execution (attempt {attempt+1}/{max_retries+1})...")

                msg_id = self._send_execute_request(code_action)
                outputs = []
                deadline = time.monotonic() + self.exec_timeout_s
                interrupted = False
                grace_deadline = None

                while True:
                    now = time.monotonic()
                    if not interrupted and now >= deadline:
                        self._interrupt_kernel()
                        interrupted = True
                        grace_deadline = now + 5

                    try:
                        msg_str = self.ws.recv()
                    except WebSocketTimeoutException:
                        if interrupted and grace_deadline and time.monotonic() >= grace_deadline:
                            self._restart_kernel()
                            return f"Execution timed out after {self.exec_timeout_s}s and was interrupted."
                        continue

                    msg = json.loads(msg_str)

                    if msg.get("parent_header", {}).get("msg_id") != msg_id:
                        continue

                    t = msg.get("msg_type", "")
                    if t == "stream":
                        outputs.append(msg["content"].get("text", ""))
                    elif t in ("execute_result", "display_data"):
                        pass
                    elif t == "error":
                        tb = "\n".join(msg["content"].get("traceback", []))
                        prefix = "Error during code execution:" if not interrupted \
                            else f"Python execution interrupted due to Python Tool timeout after {self.exec_timeout_s} second(s). Traceback:"
                        return prefix + "\n" + self._clean_text(tb)
                    elif t == "status" and msg["content"].get("execution_state") == "idle":
                        out = "".join(outputs)
                        if interrupted:
                            out += f"\n[Execution interrupted after {self.exec_timeout_s}s]"
                        return out

            except (WebSocketConnectionClosedException, ConnectionResetError, BrokenPipeError, WebSocketTimeoutException, AttributeError) as e:
                _LOG.error(f"Websocket communication error ({type(e).__name__}): {e}. Attempt {attempt+1}/{max_retries+1}.")
                attempt += 1
                if attempt > max_retries:
                    _LOG.error("Maximum retries exceeded after websocket errors.")
                    raise RuntimeError("Websocket connection failed after multiple retries.") from e
                time.sleep(1)

            except json.JSONDecodeError as e:
                 _LOG.error(f"Failed to decode JSON message: {e}. Message: '{msg_str[:100]}...'")
                 raise Exception(f"Received invalid JSON from kernel: {e}") from e
            except Exception as e:
                _LOG.error(f"Error during code execution: {e}")
                # Do not raise, just return the error message
                return f"Error during code execution:\n{e}"

        raise RuntimeError("Code execution failed due to unexpected loop exit.")

    def _send_execute_request(self, code: str) -> str:
        msg_id = str(uuid.uuid4())
        execute_request = {
            "header": {
                "msg_id": msg_id,
                "username": "anonymous",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.0",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
        }

        self.ws.send(json.dumps(execute_request))
        return msg_id

    def forward(self, code: str) -> str:
        output = self.run_code_raise_errors(code)
        return self._clean_text(output)

class RemoteSageMathTool(RemotePythonTool):
    name = "sagemath"
    description = textwrap.dedent("""
    Executes SageMath code in a stateful Jupyter notebook environment.
    IMPORTANT: Ensure code is ONLY SageMath code.

    Args:
        code (str): SageMath code to execute

    Returns:
        Execution result or error message

    Provided:
      - lll_cvp: solves linear (in)equalities using lattice reduction algorithms. Available as `lll_cvp` package. Source with docstrings provided at /resources, or https://raw.githubusercontent.com/maple3142/lll_cvp/refs/heads/master/lll_cvp.py.
      - coppersmith: there are .py files with coppersmith code available at /resources/coppersmith. This is if small_roots is insufficient for any coppersmith needs (i.e multivariate coppersmith).
      - flatter: very fast lattice reduction. Note - lll_cvp and other tools automatically select flatter, however if dealing with large lattices you can use the flatter function from lll_cvp to reduce lattices
    """)
    inputs = {
        "code": {
            "type": "string",
            "description": "SageMath code to execute"
        }
    }
    output_type = "string"
    tool_guidelines = textwrap.dedent("""\
    ## SageMath tool

    This is a cryptography and mathematics tool. You have access to the sagemath standard libraries to help you iterate faster. Use this tool heavily when dealing with cryptographic problems and mathematical problems.
    """).strip()
