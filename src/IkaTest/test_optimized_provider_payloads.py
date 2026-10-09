import copy
import json
from types import SimpleNamespace

import pytest

from IkaModel import runtime_options
from IkaModel.anthropic.claude import anthropic_fill_payload
from IkaModel.base import AgentTool, ToolArgs
from IkaModel.deepseek.deepseek import deepseek_fill_payload
from IkaModel.gemini.google import gemini_fill_payload

BUILDERS = [anthropic_fill_payload, deepseek_fill_payload, gemini_fill_payload]


def model(name="claude-sonnet-4-5", tools=None, **kw):
    return SimpleNamespace(model_id=name, system_prompt="system", max_tokens=10000, temperature=0,
                           agent_tools=tools or [], parallel_tool_calls=False, forced_tool_name=None, **kw)


def history(*entries):
    return {"system": {"message": "system"}, "first_input": {"message": "task"},
            "summary": {"message": "summary"}, "messages": {str(i): e for i, e in enumerate(entries)}}


def tool(name, required=False):
    return AgentTool(id=name, name=name, description="tool", args=ToolArgs("object", "args"), required=required)


@pytest.mark.parametrize("fill", BUILDERS)
def test_optimized_payloads_keep_user_inputs_and_valid_history_without_mutation(fill):
    h = history({"type": "hitl_input", "message": "human answer"},
                {"message": {"nested": 1}}, {"message": "assistant thought", "reasoning_content": "reason"},
                {"type": "tool", "message": "invalid-json"}, {"type": "assistant_with_tools", "message": "invalid-json"})
    inputs = [{"content": "new input"}, {"other": "value"}, "plain input"]
    before = copy.deepcopy((inputs, h))
    with runtime_options(optimize_provider_payloads=True):
        payload = fill(model(), inputs, h)
    encoded = json.dumps(payload)
    assert all(text in encoded for text in ("human answer", "assistant thought", "summary", "task", "new input", "plain input"))
    assert (inputs, h) == before


@pytest.mark.parametrize("fill", BUILDERS)
def test_optimized_builders_accept_empty_history_and_explicit_tool_override(fill):
    with runtime_options(optimize_provider_payloads=True):
        payload = fill(model(tools=[tool("included")]), [], agent_tools=[])
        populated = fill(model(), [{"role": "user", "content": "input"}], agent_tools=[tool("lookup")])
    assert "tools" not in payload
    assert "lookup" in json.dumps(populated["tools"])


@pytest.mark.parametrize("required, forced, expected", [
    ([], None, {"type": "auto", "disable_parallel_tool_use": True}),
    (["a"], None, {"type": "tool", "name": "a"}),
    (["a", "b"], None, {"type": "required", "disable_parallel_tool_use": True}),
    (["a"], "b", {"type": "tool", "name": "b"}),
])
def test_anthropic_explicit_choice_survives_optimized_policy(required, forced, expected):
    m = model(tools=[tool(n, n in required) for n in ("a", "b")])
    m.forced_tool_name = forced
    with runtime_options(optimize_provider_payloads=True):
        assert anthropic_fill_payload(m, ["input"])["tool_choice"] == expected


def test_anthropic_token_cap_parallel_prompt_and_nontext_blocks_are_preserved():
    m = model(name="claude-3-5-haiku-latest", tools=[tool("lookup")])
    m.parallel_tool_calls = True
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "fixture"}}
    with runtime_options(optimize_provider_payloads=True):
        payload = anthropic_fill_payload(m, [{"role": "user", "content": [image]}])
        blank = anthropic_fill_payload(m, [{"role": "assistant", "content": [{"type": "text", "text": " "}]}])
    assert payload["max_tokens"] == 4096
    assert image in payload["messages"][0]["content"]
    m.model_id = "claude-sonnet-4-5"
    with runtime_options(optimize_provider_payloads=True):
        assert "use_parallel_tool_calls" in anthropic_fill_payload(m, ["input"])["system"]
    assert blank["messages"] == []


def test_gemini_forced_choice_requires_known_name_and_keeps_parts():
    m = model(tools=[tool("lookup")])
    m.forced_tool_name = "lookup"
    inputs = [{"role": "user", "parts": [{"text": "task"}]}, {"role": "model", "parts": [{"text": "answer"}]}]
    with runtime_options(optimize_provider_payloads=True):
        payload = gemini_fill_payload(m, inputs, history())
        m.forced_tool_name = "absent"
        unforced = gemini_fill_payload(m, inputs, history())
    assert payload["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"] == ["lookup"]
    assert "toolConfig" not in unforced
    assert payload["contents"][-1] == inputs[-1]
    assert sum(c["parts"] == inputs[0]["parts"] for c in payload["contents"]) == 1


def test_later_stage_inputs_are_last_for_deepseek_and_gemini():
    h = history({"type": "stage_input", "message": "stage zero"}, {"message": "old answer"},
                {"type": "stage_input", "message": "task"})
    h["summary"]["message"] = ""
    with runtime_options(optimize_provider_payloads=True):
        deepseek = deepseek_fill_payload(model(name="deepseek-reasoner"), [{"role": "user", "content": "task"}], h)
        gemini = gemini_fill_payload(model(), [{"role": "user", "content": "task"}], h)
    assert deepseek["messages"][-1]["content"] == "task"
    assert gemini["contents"][-1]["parts"] == [{"text": "task"}]
    assert deepseek["thinking"] == {"type": "enabled"}
