"""Execute adoption docs, examples and applications against the installed wheel offline."""

import builtins
import contextlib
import importlib.util
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

import IkaCore
from IkaCore import IkaBaseAgent, OutputConfig
from IkaCore.testing import MockProvider, provider_response
from IkaMem.storage import mem0_storage
from IkaModel.codex import codex_auth

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("IKACORE_DOC_CHECK_OUTPUT", ROOT / "artifacts/quality/offline"))
OUTPUT.mkdir(parents=True, exist_ok=True)
assert "site-packages" in IkaCore.__file__ and not Path(IkaCore.__file__).is_relative_to(ROOT / "src")
report = {"runtime": IkaCore.__file__, "documents": [], "examples": [], "apps": []}


class OfflineMem0:
    """Replace only the optional remote Mem0 client boundary for documentation checks."""

    def __init__(self, **kwargs):
        self.values = []

    @classmethod
    def from_config(cls, config):
        return cls()

    def add(self, messages, **kwargs):
        self.values.extend(messages)

    def search(self, **kwargs):
        return {"results": [{"memory": "offline memory"}]}

    def reset(self):
        self.values.clear()


@contextlib.contextmanager
def fixture():
    seen = set()
    calls = []

    def respond(url, payload):
        assert len(calls) < 150, "fixture did not terminate"
        tools = {t.get("name") or t.get("function", {}).get("name") for t in payload.get("tools", [])}
        name = "stage_end" if "stage_end" in tools else "agent_end" if "agent_end" in tools else None
        args = {"reason": "Offline stage complete"} if name == "stage_end" else {"input": "Offline example completed"}
        if "ask_user" in tools and "ask_user" not in seen:
            name, args = "ask_user", {"question": "Which directory?"}
        else:
            for candidate, arguments in [
                ("get_pwd", {"None": "No parameters are required"}),
                ("list_files", {"directory_path": "."}),
                ("read_file", {"file_path": "fixture.txt"}),
                ("read_files", {"file_path": "fixture.txt"}),
            ]:
                if candidate in tools and candidate not in seen:
                    name, args = candidate, arguments
                    break
        seen.add(name)
        calls.append(name)
        return provider_response(url, payload, (name, args) if name else "Offline summary", len(calls))

    def blocked(*a, **kw):
        raise AssertionError("Unexpected network request")

    with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
        old = os.getcwd()
        os.chdir(temp)
        stack.callback(os.chdir, old)
        Path("fixture.txt").write_text("offline fixture")
        for name in ("API_KEY", "API_KEY2", "OPENROUTER_API_KEY", "CODEX_BEARER"):
            stack.enter_context(patch.dict(os.environ, {name: "sk-or-offline-fixture"}))
        stack.enter_context(patch.dict(os.environ, {"MODEL_ID": "deepseek-chat", "MODEL_ID2": "gpt-4o-mini"}))
        stack.enter_context(patch.object(codex_auth, "get_bearer", lambda **kw: "offline-bearer"))
        stack.enter_context(patch.object(codex_auth, "get_chatgpt_plan", lambda: "offline"))
        stack.enter_context(patch.object(builtins, "input", lambda *a: "."))
        stack.enter_context(patch.object(socket.socket, "connect", blocked))
        stack.enter_context(patch.object(socket.socket, "connect_ex", blocked))
        stack.enter_context(patch.object(mem0_storage, "Memory", OfflineMem0))
        stack.enter_context(patch.object(mem0_storage, "MemoryClient", OfflineMem0))
        stack.enter_context(patch.object(mem0_storage, "_MEM0_AVAILABLE", True))
        stack.enter_context(MockProvider(responder=respond))
        yield calls


for doc in ("README.md", "docs/USAGE.md", "docs/v2/README.md", "src/IkaMem/docs/README.md", "CHANGELOG.md"):
    blocks = list(re.finditer(r"```python\n(.*?)```", (ROOT / doc).read_text(), re.S))
    ns = {"__name__": "__documentation__"}
    with fixture() as calls:
        if doc == "docs/v2/README.md":
            ns["agent"] = IkaBaseAgent(
                "migration", prompt="Say OK", model_id="gpt-4o-mini", api_key="fixture", output=OutputConfig(cli=False)
            )
        for i, m in enumerate(blocks, 1):
            item = {"doc": doc, "block": i, "line": (ROOT / doc).read_text()[: m.start()].count("\n") + 1}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(m[1], f"{doc}:block-{i}", "exec"), ns)
                item["status"] = "passed"
            except Exception as e:
                item.update(status="failed", error=type(e).__name__ + ": " + str(e))
                traceback.print_exc()
            report["documents"].append(item)
        print(doc, len(blocks), "blocks", len(calls), "mock requests")
# API code fences are generated signatures, rather than runnable Python. Verify all
# four root __all__ inventories and the exact full generated signature document.
spec = importlib.util.spec_from_file_location("api_gen", ROOT / "scripts/generate_api_reference.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
report["api_reference_matches_installed_wheel"] = g.generate() == (ROOT / "docs/API.md").read_text()
report["api_signature_blocks"] = len(re.findall(r"```python\n", (ROOT / "docs/API.md").read_text()))
report["exports"] = {
    n: list(__import__(n, fromlist=["__all__"]).__all__) for n in ("IkaCore", "IkaModel", "IkaMem", "IkaCore.testing")
}
for file in sorted((ROOT / "examples").glob("example_*.py")):
    item = {"file": file.name}
    try:
        with fixture() as calls, contextlib.redirect_stdout(io.StringIO()) as out:
            spec = importlib.util.spec_from_file_location("example_" + file.stem, file)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            m.main()
            if file.stem == "example_workflow_advanced":
                m.run_sync()
        assert calls and "sk-or-offline-fixture" not in out.getvalue()
        item.update(status="passed", mock_requests=len(calls))
    except Exception as e:
        item.update(status="failed", error=type(e).__name__ + ": " + str(e))
        traceback.print_exc()
    report["examples"].append(item)
# Apps supply their own offline fixtures and exercise real subprocess resume.
env = dict(os.environ)
env.pop("PYTHONPATH", None)
for app in sorted((ROOT / "examples/apps").glob("*/main.py")):
    module = "examples.apps." + app.parent.name + ".main"
    proc = subprocess.run(
        [sys.executable, "-m", module, "--offline"], cwd=ROOT, env=env, text=True, capture_output=True, timeout=90
    )
    (OUTPUT / ("app-" + app.parent.name + ".log")).write_text(proc.stdout + proc.stderr)
    report["apps"].append({"app": app.parent.name, "exit": proc.returncode})
# One extra documented cancellation option.
proc = subprocess.run(
    [sys.executable, "-m", "examples.apps.parallel_pipeline.main", "--offline", "--cancel-demo"],
    cwd=ROOT,
    env=env,
    text=True,
    capture_output=True,
    timeout=90,
)
report["cancel_demo_exit"] = proc.returncode
(OUTPUT / "results.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

failed = (
    any(item["status"] != "passed" for item in report["documents"] + report["examples"])
    or any(item["exit"] for item in report["apps"])
    or report["cancel_demo_exit"]
    or not report["api_reference_matches_installed_wheel"]
)
raise SystemExit(bool(failed))
