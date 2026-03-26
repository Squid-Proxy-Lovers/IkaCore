"""
Tests for IkaBaseAgent tool conversion and API schema generation:
convert IkaTools/subagents to AgentTool, build_tool_executors, and payload tools shape.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

mock_ika_mem = MagicMock()
with patch.dict("sys.modules", {"IkaMem": mock_ika_mem}):
    from IkaCore.agents import IkaBaseAgent
from IkaCore.agent_tools import SUBAGENT_TOOL_TIMEOUT_SECONDS
from IkaCore.tools import IkaTools
from IkaCore.stages import IkaStage
from IkaModel.base import AgentTool, ToolArgs, BareBoneModel
from IkaModel.chat_helpers_common import build_provider_request
from IkaModel.openai.openai import openai_fill_payload


def _minimal_agent(**kwargs):
    defaults = {"name": "TestAgent", "description": "Desc", "prompt": "Do it", "model_id": "gpt-4o", "api_key": "k"}
    defaults.update(kwargs)
    return IkaBaseAgent(**defaults)


class TestConvertToolsToAgentTools:
    def test_ika_tool_to_agent_tool(self):
        ika = IkaTools("fetch", "Fetches data", {"url": {"type": "string", "description": "URL"}}, execute_function=lambda p: "ok")
        a = _minimal_agent()
        agent_tools = a._convert_tools_to_agent_tools([ika])
        assert len(agent_tools) == 1
        t = agent_tools[0]
        assert t.name == "fetch"
        assert t.description == "Fetches data"
        assert t.args.type == "object"
        assert t.args.properties is not None
        assert "url" in t.args.properties

    def test_ika_tool_required_param(self):
        ika = IkaTools("run", "Run", {"x": {"type": "string", "description": "X", "required": True}}, execute_function=lambda p: "ok")
        a = _minimal_agent()
        agent_tools = a._convert_tools_to_agent_tools([ika])
        assert agent_tools[0].args.properties.get("__required__") == ["x"]

    def test_agent_tool_passthrough(self):
        at = AgentTool("id", "direct", "Direct", ToolArgs(type="input", description="In"))
        a = _minimal_agent()
        out = a._convert_tools_to_agent_tools([at])
        assert len(out) == 1
        assert out[0] is at

    def test_ika_tool_parallel_and_limit_calls(self):
        ika = IkaTools("p", "P", {"x": "string"}, limit_calls=2, parallel=False, execute_function=lambda p: "ok")
        a = _minimal_agent()
        agent_tools = a._convert_tools_to_agent_tools([ika])
        assert agent_tools[0].parallel is False
        assert agent_tools[0].limit_calls == 2


class TestConvertSubagentsToTools:
    def test_subagents_become_agent_tools(self):
        sub = _minimal_agent(name="Worker")
        a = _minimal_agent(subagents=[sub])
        tools = a._convert_subagents_to_tools()
        assert len(tools) == 1
        assert tools[0].name == "Worker"
        assert tools[0].args.type == "input"
        assert tools[0].required is True

    def test_convert_subagents_param_overrides_self(self):
        sub1 = _minimal_agent(name="A")
        sub2 = _minimal_agent(name="B")
        a = _minimal_agent()
        tools = a._convert_subagents_to_tools(subagents=[sub1, sub2])
        assert len(tools) == 2
        assert {t.name for t in tools} == {"A", "B"}

    def test_empty_subagents_returns_empty(self):
        a = _minimal_agent()
        assert a._convert_subagents_to_tools() == []


class TestBuildToolExecutors:
    def test_agent_end_in_executors(self):
        a = _minimal_agent()
        executors = a.build_tool_executors([])
        assert "agent_end" in executors
        out = executors["agent_end"]({"input": "Final answer here"})
        assert "success" in out.lower() or "ended" in out.lower()

    def test_agent_end_empty_input_raises(self):
        a = _minimal_agent()
        executors = a.build_tool_executors([])
        with pytest.raises(ValueError, match="empty"):
            executors["agent_end"]({"input": ""})

    def test_stage_end_executor(self):
        a = _minimal_agent()
        stage = IkaStage("S", "P", [])
        executors = a.build_tool_executors(stage.tools, stage=stage)
        assert "stage_end" in executors
        out = executors["stage_end"]({})
        assert "stage" in out.lower()

    def test_change_stage_executor(self):
        stage = IkaStage("S", "P", [], allowed_back_to=[0])
        a = _minimal_agent(Stages=[stage])
        executors = a.build_tool_executors(stage.tools, stage=stage)
        assert "change_stage" in executors
        out = executors["change_stage"]({"reason": "Need to redo"})
        assert "stage" in out.lower()

    def test_ika_tool_executor_wired(self):
        results = []
        def my_exec(params):
            results.append(params)
            return "done"
        ika = IkaTools("my_tool", "Desc", {"x": "string"}, execute_function=my_exec)
        a = _minimal_agent()
        executors = a.build_tool_executors([ika])
        assert "my_tool" in executors
        out = executors["my_tool"]({"x": "y"})
        assert out == "done"
        assert results[0] == {"x": "y"}

    def test_subagent_executor_calls_execution(self):
        sub = _minimal_agent(name="Sub")
        sub.execution = MagicMock(return_value={"final_message": "Sub result", "summary": "Sub result"})
        a = _minimal_agent(subagents=[sub])
        executors = a.build_tool_executors(a.build_simple_tools())
        out = executors["Sub"]({"input": "Do task"})
        assert "Sub result" in out
        sub.execution.assert_called_once()

    def test_subagent_executor_has_extended_timeout_override(self):
        sub = _minimal_agent(name="Sub")
        a = _minimal_agent(subagents=[sub])
        executors = a.build_tool_executors(a.build_simple_tools())
        assert getattr(executors["Sub"], "__tool_timeout__", None) == SUBAGENT_TOOL_TIMEOUT_SECONDS


class TestApiPayloadToolSchema:
    def test_agent_tools_to_openai_payload_tools(self):
        agent_end = AgentTool(
            "agent_end", "agent_end", "End with final answer",
            ToolArgs(type="input", description="Final answer"),
            required=True, limit_calls=1,
        )
        model = BareBoneModel(
            model_id="gpt-4o", api_key="k", api_url="https://api.openai.com/v1",
            suppress_init_output=True,
        )
        model.agent_tools = [agent_end]
        messages = [{"role": "user", "content": "Hi"}]
        history = {"system": {"message": ""}, "first_input": {"message": ""}, "summary": {"message": ""}, "messages": {}}
        _, headers, payload = build_provider_request("openai", model, messages, history)
        assert "tools" in payload
        tools = payload["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "agent_end"
        params = tools[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert "input" in params["properties"]
        assert "input" in params["required"]
        assert payload["tool_choice"]["function"]["name"] == "agent_end"

    def test_subagent_tool_type_input_in_payload(self):
        at = AgentTool("Sub", "Sub", "Subagent", ToolArgs(type="input", description="Task"), required=True)
        model = BareBoneModel(
            model_id="gpt-4o", api_key="k", api_url="https://api.openai.com/v1",
            suppress_init_output=True,
        )
        model.agent_tools = [at]
        payload = openai_fill_payload(model, [{"role": "user", "content": "Hi"}], None)
        assert "tools" in payload
        fn = payload["tools"][0]["function"]
        assert fn["parameters"]["properties"]["input"] is not None
        assert "input" in fn["parameters"]["required"]

    def test_object_properties_in_payload(self):
        at = AgentTool(
            "search",
            "search",
            "Search",
            ToolArgs(type="object", description="Query", properties={"query": {"type": "string", "description": "Q"}}),
            required=False,
        )
        model = BareBoneModel(
            model_id="gpt-4o", api_key="k", api_url="https://api.openai.com/v1",
            suppress_init_output=True,
        )
        model.agent_tools = [at]
        payload = openai_fill_payload(model, [{"role": "user", "content": "Hi"}], None)
        params = payload["tools"][0]["function"]["parameters"]
        assert params["properties"]["query"]["type"] == "string"
        assert payload["tool_choice"] == "auto"
