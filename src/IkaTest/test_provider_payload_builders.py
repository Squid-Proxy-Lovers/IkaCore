import json
from types import SimpleNamespace

from IkaModel.anthropic.claude import anthropic_fill_payload
from IkaModel.base import AgentTool, ToolArgs
from IkaModel.deepseek.deepseek import deepseek_fill_payload
from IkaModel.gemini.google import gemini_fill_payload
from IkaModel.openai.openai import openai_fill_payload
from IkaModel.openai.openai_responses import openai_responses_fill_payload


def _message_history(system: str = "system", first_input: str = "", summary: str = ""):
    return {
        "system": {"message": system, "tokens": 0},
        "first_input": {"message": first_input, "tokens": 0},
        "summary": {"message": summary, "tokens": 0},
        "messages": {},
    }


def _model(**overrides):
    defaults = {
        "model_id": "gpt-4o",
        "max_tokens": 4096,
        "temperature": 0.2,
        "system_prompt": "fallback system",
        "agent_tools": [],
        "reasoning_effort": None,
        "parallel_tool_calls": False,
        "forced_tool_name": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _tool(name: str, *, required: bool = False, parallel: bool = False):
    return AgentTool(
        id=name,
        name=name,
        description=f"{name} description",
        args=ToolArgs(
            type="object",
            description=f"{name} args",
            properties={"query": {"type": "string", "description": "query"}, "__required__": ["query"]},
        ),
        required=required,
        parallel=parallel,
    )


def test_openai_responses_converts_mixed_message_shapes_and_forced_tool_choice():
    tools = [_tool("lookup", required=True, parallel=True), _tool("other")]
    model = _model(
        agent_tools=tools,
        forced_tool_name="other",
        parallel_tool_calls=True,
        reasoning_effort="low",
    )
    history = _message_history(system="instructions", first_input="original", summary="summary")
    messages = [
        {"role": "user", "content": "original"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": "{\"query\":\"x\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": {"ok": True}},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "anthropic_call", "content": "result"}, "plain"]},
        {"role": "reviewer", "content": "nonstandard role"},
    ]

    payload = openai_responses_fill_payload(model, messages, history)

    assert payload["instructions"] == "instructions"
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["tool_choice"] == {"type": "function", "name": "other"}
    assert payload["parallel_tool_calls"] is True
    assert [item for item in payload["input"] if item.get("content") == "original"] == [
        {"type": "message", "role": "user", "content": "original"}
    ]
    assert {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"query\":\"x\"}"} in payload["input"]
    assert {"type": "function_call_output", "call_id": "call_1", "output": "{\"ok\": true}"} in payload["input"]
    assert {"type": "function_call_output", "call_id": "anthropic_call", "output": "result"} in payload["input"]
    assert {"type": "message", "role": "reviewer", "content": "nonstandard role"} in payload["input"]


def test_openai_chat_payload_deduplicates_first_input_caps_tokens_and_selects_tools():
    tools = [_tool("lookup", required=True), _tool("other")]
    model = _model(
        model_id="gpt-4o",
        max_tokens=50000,
        temperature=0.7,
        agent_tools=tools,
        forced_tool_name="other",
        parallel_tool_calls=True,
        reasoning_effort="low",
    )
    history = _message_history(system="system", first_input="first", summary="summary")
    messages = [
        {"role": "user", "content": "first"},
        {"content": "second"},
        {"role": "assistant", "content": "reply"},
        {"value": 1},
    ]

    payload = openai_fill_payload(model, messages, history)

    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "summary"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "{'value': 1}"},
    ]
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 16384
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "other"}}
    assert payload["parallel_tool_calls"] is True
    assert "reasoning_effort" not in payload


def test_openai_reasoning_model_uses_completion_token_key_and_default_temperature():
    model = _model(model_id="o4-mini", max_tokens=9000, reasoning_effort="medium")

    payload = openai_fill_payload(model, [{"role": "user", "content": "hi"}], _message_history(system=""))

    assert payload["max_completion_tokens"] == 8192
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    assert payload["reasoning_effort"] == "medium"


def test_deepseek_payload_replays_valid_history_skips_bad_history_and_caps_tokens():
    tool = _tool("lookup")
    model = _model(model_id="deepseek-reasoner", max_tokens=50000, agent_tools=[tool])
    history = _message_history(system="system", first_input="first", summary="summary")
    history["messages"]["assistant"] = {
        "message": "answer",
        "reasoning_content": "thoughts",
    }
    history["messages"]["bad_tool"] = {"type": "tool", "message": "{bad json"}
    history["messages"]["tool"] = {"type": "tool", "message": json.dumps({"role": "tool", "content": "ok"})}

    payload = deepseek_fill_payload(
        model,
        [{"role": "user", "content": "first"}, {"content": "next"}],
        history,
    )

    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "summary"},
        {"role": "assistant", "content": "answer", "reasoning_content": "thoughts"},
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    assert payload["max_tokens"] == 32768
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["tools"][0]["function"]["name"] == "lookup"


def test_anthropic_payload_replays_history_normalizes_assistant_tools_and_controls_parallel_use():
    tool = _tool("search")
    model = _model(
        model_id="claude-sonnet-4-6",
        agent_tools=[tool],
        parallel_tool_calls=False,
    )
    history = _message_history(system="", first_input="first", summary="summary")
    history["messages"]["assistant_tools"] = {
        "type": "assistant_with_tools",
        "message": json.dumps({"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {}}]}),
    }
    history["messages"]["tool"] = {
        "type": "tool",
        "message": json.dumps({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}),
    }

    payload = anthropic_fill_payload(model, [{"role": "user", "content": "first"}, {"role": "assistant", "content": []}], history)

    assert payload["system"] == "fallback system"
    assert payload["messages"][0] == {"role": "user", "content": "first"}
    assert payload["messages"][2]["content"][0] == {"type": "text", "text": " "}
    assert payload["messages"][3]["content"][0]["type"] == "tool_result"
    assert payload["messages"][4]["content"] == [{"type": "text", "text": " "}]
    assert payload["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_anthropic_payload_appends_parallel_prompt_for_supported_models():
    model = _model(model_id="claude-opus-4", parallel_tool_calls=True)

    payload = anthropic_fill_payload(model, [{"role": "user", "content": "hi"}], _message_history(system="base"))

    assert payload["system"].startswith("base")
    assert "<use_parallel_tool_calls>" in payload["system"]


def test_gemini_payload_handles_history_fallback_dedup_and_forced_tool_config():
    tool = _tool("lookup")
    model = _model(
        model_id="gemini-2.5-pro",
        agent_tools=[tool],
        forced_tool_name="lookup",
        max_tokens=256,
    )
    history = _message_history(system="gemini system", first_input="first", summary="summary")
    history["messages"]["bad_json"] = {"type": "assistant_with_tools", "message": "{bad json"}
    messages = [
        {"role": "user", "parts": [{"text": "first"}]},
        {"role": "user", "content": "next"},
    ]

    payload = gemini_fill_payload(model, messages, history)

    assert payload["systemInstruction"] == {"parts": [{"text": "gemini system"}]}
    assert payload["contents"][0] == {"role": "user", "parts": [{"text": "first"}]}
    assert payload["contents"][2] == {"role": "model", "parts": [{"text": "{bad json"}]}
    assert payload["contents"][-1] == {"role": "user", "parts": [{"text": "next"}]}
    assert payload["generationConfig"]["maxOutputTokens"] == 256
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    assert payload["toolConfig"] == {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": ["lookup"],
        }
    }
