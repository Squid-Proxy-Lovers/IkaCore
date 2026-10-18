"""Resume an interrupted tool turn through the public SQLite snapshot API."""

import json
from copy import deepcopy

import httpx
import pytest

from IkaCore import IkaBaseAgent, IkaStage, OutputConfig, ProviderConfig

PROVIDERS = {
    "codex": ("gpt-5.6-sol", "https://chatgpt.com/backend-api/codex/responses"),
    "openai_responses": ("gpt-5.6-sol", "https://api.openai.com/v1/responses"),
    "anthropic": ("claude-sonnet-4-5", "https://api.anthropic.com/v1/messages"),
    "gemini": ("gemini-2.5-pro", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"),
    "openrouter": ("anthropic/claude-sonnet-4.5", "https://openrouter.ai/api/v1/chat/completions"),
    "deepseek": ("deepseek-reasoner", "https://api.deepseek.com/chat/completions"),
    "zai": ("glm-4.7", "https://api.z.ai/api/coding/paas/v4/chat/completions"),
}
OPAQUE = "opaque-snapshot-reasoning"


def response_for(provider, final=False):
    name, args = ("agent_end", {"input": "done"}) if final else ("ask_user", {"question": "Approve?"})
    call = {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    if provider in {"codex", "openai_responses"}:
        output = [{"type": "function_call", "call_id": "call_1", "name": name, "arguments": json.dumps(args)}]
        if not final:
            output.insert(0, {"type": "reasoning", "id": "rs_snapshot", "summary": [], "encrypted_content": OPAQUE})
        return {"output": output}
    if provider == "anthropic":
        blocks = [{"type": "tool_use", "id": "call_1", "name": name, "input": args}]
        if not final:
            blocks[:0] = [{"type": "thinking", "thinking": "inspect", "signature": OPAQUE},
                          {"type": "redacted_thinking", "data": OPAQUE + "-redacted"}]
        return {"content": blocks}
    if provider == "gemini":
        part = {"functionCall": {"name": name, "args": args}}
        if not final:
            part["thoughtSignature"] = OPAQUE
        return {"candidates": [{"content": {"role": "model", "parts": [part]}}]}
    message = {"content": "", "tool_calls": [call]}
    if not final:
        if provider == "openrouter":
            message["reasoning_details"] = [{"type": "reasoning.encrypted", "data": OPAQUE, "id": "rs_snapshot", "index": 0}]
        else:
            message["reasoning_content"] = OPAQUE
    return {"choices": [{"message": message}]}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_public_snapshot_resume_mid_tool_loop_preserves_reasoning(provider, tmp_path, monkeypatch, capsys):
    requests = []

    def request(url, headers, payload, **kwargs):
        requests.append(deepcopy(payload))
        return httpx.Response(200, json=response_for(provider, final=len(requests) > 1),
                              request=httpx.Request("POST", url))

    monkeypatch.setattr("IkaModel.request_interface.api_request_retry", request)
    model_id, url = PROVIDERS[provider]
    options = ProviderConfig(api_url=url, use_responses_api=provider == "openai_responses",
                             thinking_budget_tokens=1024 if provider == "anthropic" else None, max_tokens=4096)

    def create():
        return IkaBaseAgent("snapshot_fixture", "fixture", "task", model_id=model_id, api_key="fixture",
                            provider=options, Stages=[IkaStage("phase", "ask for approval", [], hitl=True)],
                            output=OutputConfig(show_usage_level0=False))

    path = str(tmp_path / "reasoning.db")
    first = create()
    store = first.enable_snapshots(path)
    try:
        interrupted = first.execution()
        assert interrupted["status"] == "awaiting_user_input"
        snapshot = store.load_snapshot(interrupted["snapshot_id"])
        assert OPAQUE in json.dumps(snapshot["state_json"])
    finally:
        first.shutdown()

    # A distinct agent/store instance has no live pending model/tool-loop state.
    resumed_agent = create()
    resumed_agent.enable_snapshots(path)
    try:
        result = resumed_agent.resume_from_snapshot(interrupted["snapshot_id"], resume_input="Yes")
        assert result["final_message"] == "done"
    finally:
        resumed_agent.shutdown()
    assert len(requests) == 2
    assert OPAQUE in json.dumps(requests[1])
    if provider in {"codex", "openai_responses"}:
        replay = requests[1]["input"]
        assert replay[1]["encrypted_content"] == OPAQUE
        assert [item["type"] for item in replay[1:4]] == ["reasoning", "function_call", "function_call_output"]
    elif provider == "anthropic":
        assistant = next(m for m in requests[1]["messages"] if m["role"] == "assistant")
        assert assistant["content"] == response_for(provider)["content"]
    elif provider == "gemini":
        assistant = next(m for m in requests[1]["contents"] if m["role"] == "model")
        assert assistant["parts"] == response_for(provider)["candidates"][0]["content"]["parts"]
    elif provider == "openrouter":
        assistant = next(m for m in requests[1]["messages"] if m["role"] == "assistant")
        assert assistant["reasoning_details"] == response_for(provider)["choices"][0]["message"]["reasoning_details"]
    output = capsys.readouterr()
    assert OPAQUE not in output.out + output.err
