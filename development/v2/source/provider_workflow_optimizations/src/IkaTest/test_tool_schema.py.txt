from types import SimpleNamespace

from IkaModel import tool_schema as ts
from IkaModel.base import AgentTool, ToolArgs


class SlotTool:
    __slots__ = ("name", "description", "args", "required", "parallel")

    def __init__(self):
        self.name = "slot_tool"
        self.description = "Slot tool"
        self.args = ToolArgs("input", "slot payload")
        self.required = False
        self.parallel = False


def test_build_tool_parameters_covers_input_object_stage_and_uncacheable_tools():
    agent_end = AgentTool("agent_end", "agent_end", "End", ToolArgs("input", ""))
    assert ts.build_tool_parameters(agent_end)["properties"]["input"]["description"].startswith("Final response")

    generic_input = AgentTool("input", "input_tool", "Input", ToolArgs("input", "Custom input"))
    assert ts.build_tool_parameters(generic_input) == {
        "type": "object",
        "properties": {"input": {"type": "string", "description": "Custom input"}},
        "required": ["input"],
    }

    empty_object = AgentTool("empty", "empty", "Empty", ToolArgs("object", "payload", properties={}))
    assert ts.build_tool_parameters(empty_object) == {"type": "object", "properties": {}, "required": []}

    object_root = AgentTool(
        "root",
        "root",
        "Root",
        ToolArgs("object", "payload", properties={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}),
    )
    assert ts.build_tool_parameters(object_root)["required"] == ["x"]

    scalar_schema = AgentTool(
        "scalar",
        "scalar",
        "Scalar",
        ToolArgs("object", "payload", properties={"type": "string", "description": "not object"}),
    )
    assert ts.build_tool_parameters(scalar_schema) == {"type": "object", "properties": {}, "required": []}

    stage_index = AgentTool("stage", "change_stage", "Change", ToolArgs("stage_index", "Stage index"))
    assert ts.build_tool_parameters(stage_index)["properties"]["stage_index"]["type"] == "integer"

    slot_tool = SlotTool()
    assert ts.build_tool_parameters(slot_tool)["required"] == ["input"]


def test_provider_tool_payload_renders_provider_shapes_and_cache_eviction():
    ts._PROVIDER_TOOL_CACHE.clear()
    ts._PROVIDER_TOOL_FAST_CACHE.clear()
    original_max = ts._PROVIDER_TOOL_CACHE_MAX
    ts._PROVIDER_TOOL_CACHE_MAX = 1

    try:
        tool = AgentTool(
            "lookup",
            "lookup",
            "Lookup",
            ToolArgs("object", "payload", properties={"value": {"type": "string"}, "__required__": ["value"]}),
            required=True,
            parallel=True,
        )
        assert ts.build_provider_tool_payload("openai", [tool]).tools[0]["function"]["name"] == "lookup"
        assert ts.build_provider_tool_payload("anthropic", [tool]).tools[0]["input_schema"]["required"] == ["value"]
        assert ts.build_provider_tool_payload("gemini", [tool]).tools[0]["parameters"]["required"] == ["value"]

        codex_payload = ts.build_provider_tool_payload("codex", [tool])
        assert codex_payload.tools[0]["parameters"]["additionalProperties"] is False
        assert codex_payload.names == frozenset({"lookup"})
        assert codex_payload.required_names == ("lookup",)

        other_tool = SimpleNamespace(
            name="other",
            description="Other",
            required=False,
            parallel=False,
            args=ToolArgs("input", "payload"),
        )
        ts.build_provider_tool_payload("openai", [other_tool])
        assert len(ts._PROVIDER_TOOL_CACHE) <= 1

        assert ts.build_provider_tool_payload("openai", []) == ts.ProviderToolPayload([], frozenset(), ())
    finally:
        ts._PROVIDER_TOOL_CACHE_MAX = original_max
        ts._PROVIDER_TOOL_CACHE.clear()
        ts._PROVIDER_TOOL_FAST_CACHE.clear()


def test_control_tools_never_pin_tool_choice_but_user_required_tools_do():
    from IkaModel.anthropic.claude import anthropic_fill_payload
    from IkaModel.base import AgentTool, ToolArgs
    from IkaModel.openai.openai import openai_fill_payload
    from IkaModel.openai.openai_responses import openai_responses_fill_payload
    from IkaModel.tool_schema import CONTROL_TOOL_NAMES, build_provider_tool_payload

    def tool(name, required):
        return AgentTool(name, name, f"{name} tool", ToolArgs(type="input", description="x"), required=required)

    controls = [tool(name, True) for name in sorted(CONTROL_TOOL_NAMES)]
    assert build_provider_tool_payload("openai", controls).required_names == ()

    class Model:
        model_id, max_tokens, temperature, system_prompt = "gpt-4o", 100, 0.0, ""
        parallel_tool_calls, forced_tool_name, reasoning_effort = False, None, None

    history = {"system": {"message": ""}, "first_input": {"message": ""}, "summary": {"message": ""}, "messages": {}}
    msgs = [{"role": "user", "content": "hi"}]
    only_control = [tool("lookup", False), tool("agent_end", True)]
    with_user_required = [tool("submit_report", True), tool("agent_end", True)]

    assert openai_fill_payload(Model(), msgs, history, only_control)["tool_choice"] == "auto"
    assert openai_responses_fill_payload(Model(), msgs, history, only_control)["tool_choice"] == "auto"
    assert anthropic_fill_payload(Model(), msgs, history, only_control)["tool_choice"]["type"] == "auto"

    assert openai_fill_payload(Model(), msgs, history, with_user_required)["tool_choice"] == {
        "type": "function", "function": {"name": "submit_report"},
    }
    assert anthropic_fill_payload(Model(), msgs, history, with_user_required)["tool_choice"] == {
        "type": "tool", "name": "submit_report",
    }
