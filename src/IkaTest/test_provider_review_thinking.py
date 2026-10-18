import pytest

from IkaModel.anthropic.claude import anthropic_fill_payload
from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.summarization import create_summary_payload


@pytest.mark.parametrize("name", ["claude-opus-4-7", "claude-opus-4-8", "claude-opus-4-7-20260416"])
def test_modern_opus_omits_sampling_for_turns_and_summaries(name):
    model = BareBoneModel(name, "fixture", "https://api.anthropic.com/v1/messages")
    assert "temperature" not in anthropic_fill_payload(model, [{"role": "user", "content": "task"}])
    summary = create_summary_payload("anthropic", name, "fixture", "conversation")[0]
    assert "temperature" not in summary


@pytest.mark.parametrize("name", ["claude-opus-4-7", "claude-opus-4-8"])
def test_modern_opus_rejects_manual_thinking_locally(name):
    model = BareBoneModel(name, "fixture", "https://api.anthropic.com/v1/messages", thinking_budget_tokens=1024)
    with pytest.raises(ValueError, match="adaptive.*thinking_budget_tokens"):
        anthropic_fill_payload(model, [])


@pytest.mark.parametrize("name", ["claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-7", "claude-opus-4-8"])
def test_capable_claude_models_use_adaptive_thinking_and_effort(name):
    model = BareBoneModel(name, "fixture", "https://api.anthropic.com/v1/messages", reasoning_effort="high")
    model.deepthinking = True
    model.agent_tools = [AgentTool("lookup", "lookup", "lookup", ToolArgs("object", "args"), required=True)]
    payload = anthropic_fill_payload(model, [{"role": "user", "content": "task"}])
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "high"}
    assert "temperature" not in payload
    assert payload["tool_choice"]["type"] == "tool"
