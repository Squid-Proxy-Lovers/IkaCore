from IkaModel.anthropic.claude import anthropic_fill_payload
from IkaModel.base import AgentTool, BareBoneModel, ToolArgs


def test_anthropic_multiple_required_tools_use_any():
    model = BareBoneModel("claude-sonnet-4-5", "fixture", "https://api.anthropic.com/v1/messages")
    model.agent_tools = [AgentTool(name, name, name, ToolArgs("object", "args"), required=True)
                         for name in ("lookup", "other")]
    choice = anthropic_fill_payload(model, [{"role": "user", "content": "task"}])["tool_choice"]
    assert choice == {"type": "any", "disable_parallel_tool_use": True}
