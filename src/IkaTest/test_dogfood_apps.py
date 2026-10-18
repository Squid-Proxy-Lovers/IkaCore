"""Fast subprocess integration checks of every application's real mocked runtime."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APPS = ("research_assistant", "code_reviewer", "approval_workflow", "long_chat", "parallel_pipeline", "provider_switch")


@pytest.mark.parametrize("name", APPS)
def test_app_offline(name):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # Subprocesses use the installed package, including in CI.
    for key in ("CODEX_BEARER", "OPENAI_API_KEY", "IKA_DUMP_REQUESTS"):
        env.pop(key, None)
    completed = subprocess.run([sys.executable, "-m", f"examples.apps.{name}.main", "--offline"],
                               cwd=ROOT, env=env, capture_output=True, text=True, timeout=25)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "passed" and result["app"] == name
    if name == "long_chat":
        assert result["turns"] == 12 and result["compactions"] >= 1
    if name == "provider_switch":
        assert len(result["routes"]) == 7 and set(result["routes"].values()) == {"mocked"}


def test_parallel_sigint_cancels_before_dispatch():
    completed = subprocess.run([sys.executable, "-m", "examples.apps.parallel_pipeline.main", "--offline", "--cancel-demo"],
                               cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["cancelled"] is True
