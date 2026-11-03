from copy import deepcopy

import pytest

from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_helpers_common import build_provider_request
from IkaModel.chat_interface.chat_request import init_message_history
from IkaModel.chat_interface.chat_response import _record_model_message


@pytest.mark.parametrize("provider,name,url", [
    ("deepseek", "deepseek-reasoner", "https://api.deepseek.com/chat/completions"),
    ("openai", "glm-4.7", "https://api.z.ai/api/coding/paas/v4/chat/completions"),
])
@pytest.mark.parametrize("turns", [1, 2])
def test_plain_assistant_replay_keeps_each_reasoning_turn_once(provider, name, url, turns):
    model = BareBoneModel(name, "fixture", url, use_responses_api=False)
    model.agent_tools = [AgentTool("lookup", "lookup", "lookup", ToolArgs("object", "args"))]
    history = init_message_history()
    history["first_input"]["message"] = "task"
    live = [{"role": "user", "content": "task"}]
    for i in range(turns):
        _record_model_message(history, "same answer", 10, f"original thought {i}")
        live.append({"role": "assistant", "content": "same answer"})
    before = deepcopy((history, live))
    payload = build_provider_request(provider, model, live, history)[2]
    assistants = [m for m in payload["messages"] if m.get("role") == "assistant"]
    assert assistants == [
        {"role": "assistant", "content": "same answer", "reasoning_content": f"original thought {i}"}
        for i in range(turns)
    ]
    assert (history, live) == before
