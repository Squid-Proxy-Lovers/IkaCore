"""
Codex provider — payload builder + tool_choice control-tool filter tests.

Covers:
  - codex_responses_fill_payload produces the codex-required shape
    (stream:true, store:false, structured input_text content, no
    max_output_tokens, instructions field set).
  - tool_choice control-tool filter: agent_end / stage_end / change_stage
    being required=True does NOT pin tool_choice to them.
  - tool_choice variants: forced_tool_name override, single real required
    tool pins, multiple required tools → "required", none → "auto".
  - Tool parameter shape (object root) gets additionalProperties:false.
"""
import sys
from pathlib import Path

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import pytest

from IkaModel.base import BareBoneModel, AgentTool, ToolArgs
from IkaModel.codex import CODEX_API_URL, build_codex_request
from IkaModel.codex.codex_responses import codex_responses_fill_payload


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _model(*, model_id="gpt-5.4-mini", reasoning="low", system="Be terse.",
           tools=None, parallel_tool_calls=False, forced_tool_name=None):
    m = BareBoneModel(
        model_id=model_id, api_key="fake-bearer", api_url=CODEX_API_URL,
        system_prompt=system, max_tokens=200, temperature=0,
        reasoning_effort=reasoning, parallel_tool_calls=parallel_tool_calls,
        suppress_init_output=True,
    )
    if tools is not None:
        m.agent_tools = tools
    if forced_tool_name:
        m.forced_tool_name = forced_tool_name
    return m


def _mh(system):
    return {
        "system": {"message": system, "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def _tool(name, required=False):
    return AgentTool(
        id=name, name=name, description=f"{name} description",
        args=ToolArgs(
            type="object", description=f"{name} args",
            properties={
                "x": {"type": "string", "description": "x"},
                "__required__": ["x"],
            },
        ),
        required=required,
    )


def _control_tool(name):
    return AgentTool(
        id=name, name=name, description=f"control tool {name}",
        args=ToolArgs(type="input", description="x"),
        required=True, limit_calls=1,
    )


# ----------------------------------------------------------------------
# Codex-required payload shape
# ----------------------------------------------------------------------

class TestCodexPayloadShape:

    def test_stream_is_true(self):
        m = _model()
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert p["stream"] is True

    def test_store_is_false(self):
        m = _model()
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert p["store"] is False

    def test_no_max_output_tokens(self):
        """codex backend rejects max_output_tokens with 400."""
        m = _model()
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert "max_output_tokens" not in p

    def test_instructions_from_message_history(self):
        m = _model(system="System prompt here.")
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh("System prompt here."))
        assert p.get("instructions") == "System prompt here."

    def test_user_content_wrapped_as_input_text_list(self):
        m = _model()
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hello world"}], _mh(m.system_prompt))
        assert isinstance(p["input"], list)
        first = p["input"][0]
        assert first["type"] == "message"
        assert first["role"] == "user"
        assert isinstance(first["content"], list)
        assert first["content"][0] == {"type": "input_text", "text": "hello world"}

    def test_reasoning_effort_passed_through(self):
        m = _model(reasoning="xhigh")
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert p.get("reasoning") == {"effort": "xhigh"}

    def test_temperature_skipped_for_gpt5(self):
        """gpt-5 family rejects custom temperature on codex backend."""
        m = _model(model_id="gpt-5.5")
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert "temperature" not in p

    def test_tools_get_additional_properties_false(self):
        m = _model(tools=[_tool("my_tool")])
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        params = p["tools"][0]["parameters"]
        assert params["additionalProperties"] is False


# ----------------------------------------------------------------------
# tool_choice — control-tool filter (the original bug)
# ----------------------------------------------------------------------

class TestToolChoiceControlFilter:

    def test_only_agent_end_required_falls_to_auto(self):
        """The original bug: agent_end is the only required=True tool;
        tool_choice must NOT pin to it, must fall to 'auto'."""
        m = _model(tools=[_control_tool("agent_end")])
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert p["tool_choice"] == "auto"

    def test_all_control_tools_required_falls_to_auto(self):
        """stage_end + change_stage + agent_end all required=True → still auto."""
        m = _model(tools=[
            _control_tool("agent_end"),
            _control_tool("stage_end"),
            _control_tool("change_stage"),
        ])
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert p["tool_choice"] == "auto"

    def test_one_real_required_tool_pins(self):
        """User-provided required tool DOES pin tool_choice."""
        m = _model(tools=[_tool("custom", required=True), _control_tool("agent_end")])
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert p["tool_choice"] == {"type": "function", "name": "custom"}

    def test_two_real_required_tools_falls_to_required(self):
        """Multiple real required tools → tool_choice='required' (free pick)."""
        m = _model(tools=[
            _tool("a", required=True), _tool("b", required=True),
            _control_tool("agent_end"),
        ])
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert p["tool_choice"] == "required"

    def test_no_required_tools_falls_to_auto(self):
        m = _model(tools=[_tool("a"), _tool("b"), _control_tool("agent_end")])
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert p["tool_choice"] == "auto"

    def test_forced_tool_name_overrides_everything(self):
        m = _model(
            tools=[_tool("a", required=True), _tool("b")],
            forced_tool_name="b",
        )
        p = codex_responses_fill_payload(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert p["tool_choice"] == {"type": "function", "name": "b"}


# ----------------------------------------------------------------------
# build_codex_request — bearer + URL + headers contract
# ----------------------------------------------------------------------

class TestBuildCodexRequest:

    def test_returns_codex_url(self):
        m = _model()
        url, headers, _ = build_codex_request(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert url == CODEX_API_URL

    def test_authorization_bearer_header(self):
        m = _model()
        _, headers, _ = build_codex_request(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert headers["Authorization"] == "Bearer fake-bearer"

    def test_accept_event_stream(self):
        m = _model()
        _, headers, _ = build_codex_request(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert headers["Accept"] == "text/event-stream"

    def test_missing_api_key_raises_value_error(self):
        m = _model()
        m.api_key = None
        with pytest.raises(ValueError, match="bearer"):
            build_codex_request(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))

    def test_empty_api_key_raises_value_error(self):
        m = _model()
        m.api_key = ""
        with pytest.raises(ValueError, match="bearer"):
            build_codex_request(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))

    def test_forces_codex_url_if_caller_set_wrong_url(self):
        """Defensive: if caller routed here but set api_url to something else,
        we normalize back to the codex endpoint."""
        m = _model()
        m.api_url = "https://api.openai.com/v1/responses"
        url, _, _ = build_codex_request(m, [{"role": "user", "content": "hi"}], _mh(m.system_prompt))
        assert url == CODEX_API_URL


# ----------------------------------------------------------------------
# Multi-turn conversation conversion (tool result → function_call_output)
# ----------------------------------------------------------------------

class TestMultiTurnInputConversion:

    def test_assistant_with_tool_calls_emits_function_call_items(self):
        m = _model()
        messages = [
            {"role": "user", "content": "do stuff"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "my_tool", "arguments": '{"x":1}'},
                }],
            },
        ]
        p = codex_responses_fill_payload(m, messages, _mh(m.system_prompt))
        fc_items = [i for i in p["input"] if i.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["call_id"] == "call_abc"
        assert fc_items[0]["name"] == "my_tool"
        assert fc_items[0]["arguments"] == '{"x":1}'

    def test_tool_role_message_emits_function_call_output_with_matching_call_id(self):
        m = _model()
        messages = [
            {"role": "user", "content": "do stuff"},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "call_xyz", "type": "function",
                                "function": {"name": "t", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_xyz", "content": "result text"},
        ]
        p = codex_responses_fill_payload(m, messages, _mh(m.system_prompt))
        fc_outs = [i for i in p["input"] if i.get("type") == "function_call_output"]
        assert len(fc_outs) == 1
        assert fc_outs[0]["call_id"] == "call_xyz"
        assert fc_outs[0]["output"] == "result text"
