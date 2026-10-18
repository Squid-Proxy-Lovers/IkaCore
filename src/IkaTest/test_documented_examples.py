"""Execute maintained examples and Python documentation blocks without credentials/network."""

import importlib.util
import json
import re
from pathlib import Path

import httpx
import pytest

from IkaModel import request_interface, summarization
from IkaModel.codex import codex_auth

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def offline_provider(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("API_KEY", "API_KEY2", "OPENROUTER_API_KEY", "CODEX_BEARER"):
        monkeypatch.setenv(name, "sk-or-offline-fixture")
    monkeypatch.setenv("MODEL_ID", "deepseek-chat")
    monkeypatch.setenv("MODEL_ID2", "gpt-4o-mini")
    monkeypatch.setattr(codex_auth, "get_bearer", lambda: "offline-bearer")
    monkeypatch.setattr(codex_auth, "get_chatgpt_plan", lambda: "offline")
    monkeypatch.setattr("builtins.input", lambda _: ".")
    calls = []
    seen = set()

    def respond(api_url, headers, payload, **kwargs):
        assert len(calls) < 100, "example did not terminate with fixture controls"
        tools = {tool.get("name") or tool.get("function", {}).get("name")
                 for tool in payload.get("tools", [])}
        name = "stage_end" if "stage_end" in tools else "agent_end" if "agent_end" in tools else None
        args = {"input": "Offline example completed"}
        if "ask_user" in tools and "ask_user" not in seen:
            name, args = "ask_user", {"question": "Which directory?"}
        else:
            for candidate, arguments in (
                ("get_pwd", {"None": "No parameters are required"}), ("list_files", {"directory_path": "."}),
                ("read_file", {"file_path": "fixture.txt"}),
                ("read_files", {"file_path": "fixture.txt"}),
            ):
                if candidate in tools and candidate not in seen:
                    name, args = candidate, arguments
                    break
        seen.add(name)
        calls.append(name)
        call_id = f"offline-{len(calls)}"
        encoded = json.dumps(args)
        output = [{"type": "function_call", "call_id": call_id, "name": name, "arguments": encoded}] if name else [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Offline summary"}]}]
        message = {"role": "assistant", "content": "Offline summary"}
        if name:
            message["tool_calls"] = [{"id": call_id, "type": "function",
                                      "function": {"name": name, "arguments": encoded}}]
        data = {"id": call_id, "output": output, "choices": [{"message": message}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                          "prompt_tokens": 1, "completion_tokens": 1}}
        return httpx.Response(200, json=data, request=httpx.Request("POST", api_url))

    async def async_respond(*args, **kwargs):
        return respond(*args, **kwargs)

    monkeypatch.setattr(request_interface, "api_request_retry", respond)
    monkeypatch.setattr(request_interface, "async_api_request_retry", async_respond)
    monkeypatch.setattr(summarization, "api_request_retry", respond)
    monkeypatch.setattr(summarization, "async_api_request_retry", async_respond)
    (tmp_path / "fixture.txt").write_text("offline fixture", encoding="utf-8")
    return calls


@pytest.mark.parametrize("path", sorted((ROOT / "examples").glob("example_*.py")), ids=lambda path: path.name)
def test_example_main_offline(path, offline_provider, capsys):
    spec = importlib.util.spec_from_file_location(f"offline_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    if path.stem == "example_workflow_advanced":
        module.run_sync()
    assert offline_provider
    output = capsys.readouterr().out
    assert "sk-or-offline" not in output
    assert '"error":' not in output
    assert "Error:" not in output
    if path.stem == "example_hitl":
        assert "ask_user" in offline_provider
        assert "stage_end" in offline_provider
    if path.stem == "example_codex":
        assert "authentication succeeded; plan=offline" in output


@pytest.mark.parametrize("document", ["README.md", "docs/USAGE.md"])
def test_documented_python_blocks_offline(document, offline_provider):
    namespace = {"__name__": "__documentation__"}
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / document).read_text(), re.DOTALL)
    assert blocks
    for index, block in enumerate(blocks, 1):
        exec(compile(block, f"{document}:block-{index}", "exec"), namespace)
        if "agent = IkaBaseAgent(" in block and "agent.execution()" not in block:
            result = namespace["agent"].execution()
            assert result["final_message"]
    assert offline_provider
    if document.endswith("USAGE.md"):
        assert "ask_user" in offline_provider
        assert namespace["resumed"]["final_message"]
        assert namespace["results"]["write"].final


def test_fileagent_uses_only_environment_credentials(monkeypatch, tmp_path):
    path = ROOT / "examples" / "example_fileagent.py"
    spec = importlib.util.spec_from_file_location("offline_fileagent_auth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "example_fileagent.py"))
    (tmp_path / "keys.cfg").write_text("openrouter=must-not-be-read\n")
    assert module._load_key() is None
    with pytest.raises(ValueError, match="Set OPENROUTER_API_KEY or API_KEY"):
        module.main()
    monkeypatch.setenv("API_KEY", "fallback")
    assert module._load_key() == "fallback"
    monkeypatch.setenv("OPENROUTER_API_KEY", "preferred")
    assert module._load_key() == "preferred"
