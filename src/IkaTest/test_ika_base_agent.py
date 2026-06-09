"""
Tests for IkaBaseAgent: init, validation, geturl, get_barebone, and API-facing tool/stage build.
"""
from unittest.mock import MagicMock, patch

import pytest

mock_ika_mem = MagicMock()
with patch.dict("sys.modules", {"IkaMem": mock_ika_mem}):
    from IkaCore.agents import IkaBaseAgent
from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools
from IkaModel.base import AgentTool, ToolArgs


def _minimal_agent(**kwargs):
    defaults = {
        "name": "TestAgent",
        "description": "Test agent",
        "prompt": "Do the task",
        "model_id": "gpt-4o",
        "api_key": "test-key",
    }
    defaults.update(kwargs)
    return IkaBaseAgent(**defaults)


class TestIkaBaseAgentInit:
    def test_init_minimal(self):
        a = _minimal_agent()
        assert a.name == "TestAgent"
        assert a.description == "Test agent"
        assert a.prompt == "Do the task"
        assert a.model_id == "gpt-4o"
        assert a.api_key == "test-key"
        assert a.api_url is not None
        assert a.tools == []
        assert a.Stages == []
        assert a.subagents == []
        assert a.next_agent is None
        assert a.maxsteps == 100
        assert a.message_history["system"]["message"] == ""
        assert a.max_tool_rounds == 5

    def test_init_with_system_prompt(self):
        a = _minimal_agent(system_prompt="You are helpful.")
        assert a.system_prompt == "You are helpful."
        assert a.message_history["system"]["message"] == "You are helpful."

    def test_init_api_url_default_from_model_id(self):
        a = _minimal_agent(model_id="gpt-4o")
        assert a.api_url.endswith("/v1/responses")
        a2 = _minimal_agent(model_id="claude-3-sonnet")
        assert "anthropic.com" in a2.api_url
        a3 = _minimal_agent(model_id="deepseek-chat")
        assert "deepseek.com" in a3.api_url
        a4 = _minimal_agent(model_id="gemini-1.5-pro")
        assert "generativelanguage.googleapis.com" in a4.api_url
        a5 = _minimal_agent(model_id="gpt-4o", use_responses_api=False)
        assert a5.api_url.endswith("/v1/chat/completions")

    def test_init_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name is required"):
            _minimal_agent(name="")

    def test_init_rejects_empty_description(self):
        with pytest.raises(ValueError, match="description is required"):
            _minimal_agent(description="")

    def test_init_rejects_empty_prompt(self):
        with pytest.raises(ValueError, match="prompt is required"):
            _minimal_agent(prompt="")

    def test_init_rejects_empty_model_id(self):
        with pytest.raises(ValueError, match="model_id is required"):
            _minimal_agent(model_id="")

    def test_init_rejects_empty_api_key(self):
        with pytest.raises(ValueError, match="api_key is required"):
            _minimal_agent(api_key="")

    def test_init_stages_with_subagents_raises(self):
        stage = IkaStage("s1", "Prompt", [])
        with pytest.raises(ValueError, match="Subagents or next_agent are not allowed when stages are defined"):
            _minimal_agent(Stages=[stage], subagents=[_minimal_agent(name="Sub")])

    def test_init_no_stages_both_subagents_and_next_raises(self):
        sub = _minimal_agent(name="Sub")
        nxt = _minimal_agent(name="Next")
        with pytest.raises(ValueError, match="Only one of subagents or next_agent may be set"):
            _minimal_agent(subagents=[sub], next_agent=nxt)

    def test_removed_public_args_are_not_accepted(self):
        with pytest.raises(TypeError):
            _minimal_agent(feedback_agent=_minimal_agent(name="Feedback"))
        with pytest.raises(TypeError):
            _minimal_agent(Batch=True)

    def test_logging_level_three_configures_debug_logging(self):
        logger = MagicMock()

        with patch("logging.basicConfig") as basic_config:
            with patch("logging.getLogger", return_value=logger) as get_logger:
                _minimal_agent(logging_level=3)

        basic_config.assert_called_once()
        get_logger.assert_called_once_with("IkaModel.chat_interface.chat_interface")
        logger.setLevel.assert_called_once()

    def test_shutdown_awaits_async_client_close_and_closes_logger(self):
        class AsyncClient:
            def __init__(self):
                self.closed = False

            async def aclose(self):
                self.closed = True

        a = _minimal_agent(use_async=True)
        client = AsyncClient()
        a.client = client
        a.logger = MagicMock()

        a.shutdown()

        assert client.closed is True
        assert a.client is None
        a.logger.shutdown.assert_called_once()


class TestGetUrl:
    def test_openai(self):
        url = IkaBaseAgent.geturl("gpt-4o")
        assert url.endswith("/v1/responses")

    def test_deepseek(self):
        url = IkaBaseAgent.geturl("deepseek-chat")
        assert "deepseek.com" in url

    def test_anthropic(self):
        url = IkaBaseAgent.geturl("claude-3-sonnet")
        assert "anthropic.com" in url

    def test_gemini(self):
        url = IkaBaseAgent.geturl("gemini-1.5-pro")
        assert "generativelanguage.googleapis.com" in url
        assert "gemini" in url

    def test_gemini_with_slash_model(self):
        url = IkaBaseAgent.geturl("models/gemini-2.5-pro")
        assert "openrouter.ai" in url

    def test_unknown_defaults_openai(self):
        url = IkaBaseAgent.geturl("unknown-model")
        assert url.endswith("/v1/responses")


class TestGetBarebone:
    def test_returns_barebone_with_agent_attrs(self):
        a = _minimal_agent(system_prompt="Sys", max_tokens=4096, temperature=0.3)
        tools: list = []
        b = a.get_barebone("Sys", tools)
        assert hasattr(b, "model_id") and b.model_id == a.model_id
        assert b.api_key == a.api_key
        assert b.api_url == a.api_url
        assert b.system_prompt == "Sys"
        assert b.max_tokens == 4096
        assert b.temperature == 0.3
        assert b.agent_hierarchy == [a.name]
        assert b.agent_tools == tools

    def test_content_prompt_override(self):
        a = _minimal_agent(prompt="Original")
        b = a.get_barebone("Sys", [], content_prompt_override="Override text")
        assert b.content_prompt == "Override text"

    def test_model_overrides(self):
        a = _minimal_agent(model_id="gpt-4o", max_tokens=5000)
        overrides = {"model_id": "gpt-4.1-mini", "max_tokens": 8000}
        b = a.get_barebone("Sys", [], model_overrides=overrides)
        assert b.model_id == "gpt-4.1-mini"
        assert b.max_tokens == 8000

    def test_api_url_override_explicit(self):
        a = _minimal_agent(model_id="gpt-4o")
        overrides = {"model_id": "claude-3-sonnet", "api_url": "https://api.anthropic.com/v1/messages"}
        b = a.get_barebone("Sys", [], model_overrides=overrides)
        assert b.model_id == "claude-3-sonnet"
        assert "anthropic" in b.api_url

    def test_parent_hierarchy(self):
        a = _minimal_agent()
        b = a.get_barebone("Sys", [], parent_hierarchy=["Parent", "Child"])
        assert b.agent_hierarchy == ["Parent", "Child"]

    def test_parallel_tool_calls_false_when_any_tool_not_parallel(self):
        t = AgentTool("id", "tool_a", "Desc", ToolArgs(type="object", description=""), parallel=False)
        a = _minimal_agent()
        b = a.get_barebone("Sys", [t])
        assert b.parallel_tool_calls is False

    def test_parallel_tool_calls_true_when_all_parallel(self):
        t = AgentTool("id", "tool_a", "Desc", ToolArgs(type="object", description=""), parallel=True)
        a = _minimal_agent()
        b = a.get_barebone("Sys", [t])
        assert b.parallel_tool_calls is True


class TestBuildSimpleTools:
    def test_includes_agent_end_and_converted_tools(self):
        ika_tool = IkaTools("my_tool", "Does stuff", {"x": "string"}, execute_function=lambda p: "ok")
        a = _minimal_agent(tools=[ika_tool])
        agent_tools = a.build_simple_tools()
        names = [t.name for t in agent_tools]
        assert "agent_end" in names
        assert "my_tool" in names
        agent_end = next(t for t in agent_tools if t.name == "agent_end")
        assert agent_end.required is True
        assert agent_end.limit_calls == 1

    def test_subagents_become_tools(self):
        sub = _minimal_agent(name="SubAgent")
        a = _minimal_agent(subagents=[sub])
        agent_tools = a.build_simple_tools()
        names = [t.name for t in agent_tools]
        assert "SubAgent" in names
        sub_tool = next(t for t in agent_tools if t.name == "SubAgent")
        assert sub_tool.args.type == "input"


class TestBuildStage:
    def test_stage_tools_plus_agent_end_on_last_stage(self):
        ika_tool = IkaTools("stage_tool", "Stage tool", {"x": "string"}, execute_function=lambda p: "ok")
        stage = IkaStage("Stage0", "Do step", [ika_tool])
        a = _minimal_agent(Stages=[stage])
        agent_tools = a.build_stage(stage)
        names = [t.name for t in agent_tools]
        assert "agent_end" in names
        assert "stage_tool" in names
        assert "stage_end" not in names

    def test_stage_end_removed_on_last_stage(self):
        stage = IkaStage("Last", "Final", [])
        a = _minimal_agent(Stages=[stage])
        agent_tools = a.build_stage(stage)
        names = [t.name for t in agent_tools]
        assert "agent_end" in names
        assert "stage_end" not in names

    def test_non_last_stage_keeps_stage_end(self):
        s0 = IkaStage("S0", "Step 0", [])
        s1 = IkaStage("S1", "Step 1", [])
        a = _minimal_agent(Stages=[s0, s1])
        tools_s0 = a.build_stage(s0)
        stage_end = next(t for t in tools_s0 if t.name == "stage_end")
        assert stage_end.required is False
        assert any(t.name == "agent_end" for t in a.build_stage(s1))

    def test_change_stage_is_optional_when_available(self):
        s0 = IkaStage("S0", "Step 0", [], allowed_back_to=[0])
        s1 = IkaStage("S1", "Step 1", [])
        a = _minimal_agent(Stages=[s0, s1])
        tools_s0 = a.build_stage(s0)
        change_stage = next(t for t in tools_s0 if t.name == "change_stage")
        assert change_stage.required is False
