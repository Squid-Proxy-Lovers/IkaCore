# pyright: strict
# pyright: reportUnusedFunction=false

"""
Payload builder for the codex Responses endpoint
(``https://chatgpt.com/backend-api/codex/responses``).

Mirrors the role of ``openai/openai_responses.py`` but produces a payload
shaped for the codex backend. Differences from the generic OpenAI Responses
payload:

- ``stream`` is always True (the codex backend rejects non-streaming).
- ``store`` is always False (we manage history client-side; no server-side
  ``previous_response_id`` plumbing in IkaCore yet).
- ``content`` items use the structured ``[{"type":"input_text","text":...}]``
  list shape. The codex endpoint accepts the plain-string shape too on the
  generic OpenAI endpoint, but the list shape is the canonical Codex CLI
  shape and works everywhere — there's no upside to the string form here.
- Only codex-known model slugs are passed through verbatim; callers using
  anything else get a clear error before the request hits the wire.

Reasoning effort, tool definitions, parallel_tool_calls, etc. follow the
standard Responses API conventions.
"""

from __future__ import annotations

import json
from typing import Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value

from ..base import AgentTool, BareBoneModel
from ..model_metadata import CODEX_KNOWN_MODELS as CODEX_KNOWN_MODELS
from ..tool_schema import build_provider_tool_payload


def _wrap_user_content(content: object) -> list[JsonDict]:
    """Convert a string or pre-wrapped list into the input_text content shape."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, list):
        # Already a list — assume the caller knows what they're doing, but
        # rewrap any bare strings inside.
        out: list[JsonDict] = []
        for block in cast(list[object], content):
            if isinstance(block, str):
                out.append({"type": "input_text", "text": block})
            elif isinstance(block, dict):
                out.append(cast(JsonDict, block))
            else:
                out.append({"type": "input_text", "text": str(block)})
        return out
    return [{"type": "input_text", "text": str(content)}]


def _wrap_assistant_content(content: object) -> list[JsonDict]:
    """Assistant turns use ``output_text`` blocks in the Responses API."""
    if isinstance(content, str):
        return [{"type": "output_text", "text": content}]
    if isinstance(content, list):
        out: list[JsonDict] = []
        for block in cast(list[object], content):
            if isinstance(block, str):
                out.append({"type": "output_text", "text": block})
            elif isinstance(block, dict):
                out.append(cast(JsonDict, block))
            else:
                out.append({"type": "output_text", "text": str(block)})
        return out
    return [{"type": "output_text", "text": str(content)}]


def _default_message_history() -> JsonDict:
    return {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def _first_message_matches(messages: list[JsonDict], first_input_content: str) -> bool:
    if not first_input_content or not messages:
        return False
    first_msg = messages[0]
    content = first_msg.get("content", "")
    return isinstance(content, str) and content == first_input_content


def _append_codex_context(input_items: list[JsonDict], message_history: JsonDict) -> None:
    summary_msg = string_value(history_section(message_history, "summary").get("message"))
    if summary_msg:
        input_items.append({
            "type": "message",
            "role": "assistant",
            "content": _wrap_assistant_content(summary_msg),
        })

    first_input_content = string_value(history_section(message_history, "first_input").get("message"))
    if first_input_content:
        input_items.append({
            "type": "message",
            "role": "user",
            "content": _wrap_user_content(first_input_content),
        })


def _codex_tool_output_item(message: JsonDict) -> JsonDict:
    content = message.get("content", "")
    return {
        "type": "function_call_output",
        "call_id": string_value(message.get("tool_call_id")),
        "output": content if isinstance(content, str) else json.dumps(content),
    }


def _codex_assistant_items(message: JsonDict) -> list[JsonDict]:
    content = message.get("content", "")
    raw_tool_calls = message.get("tool_calls")
    tool_calls = cast(list[object], raw_tool_calls) if isinstance(raw_tool_calls, list) else []
    if not tool_calls:
        return [{"type": "message", "role": "assistant", "content": _wrap_assistant_content(content)}] if content else []

    items: list[JsonDict] = []
    for tool_call in tool_calls:
        tool_call_data = json_dict(tool_call)
        fn = json_dict(tool_call_data.get("function"))
        items.append({
            "type": "function_call",
            "call_id": string_value(tool_call_data.get("id")),
            "name": string_value(fn.get("name")),
            "arguments": string_value(fn.get("arguments"), "{}"),
        })
    return items


def _codex_user_items(message: JsonDict) -> list[JsonDict]:
    content = message.get("content", "")
    if not isinstance(content, list):
        return [{"type": "message", "role": "user", "content": _wrap_user_content(content or str(message))}]

    items: list[JsonDict] = []
    for block in cast(list[object], content):
        block_data = cast(JsonDict, block) if isinstance(block, dict) else {}
        if block_data.get("type") == "tool_result":
            items.append({
                "type": "function_call_output",
                "call_id": string_value(block_data.get("tool_use_id")),
                "output": block_data.get("content", ""),
            })
        else:
            items.append({
                "type": "message",
                "role": "user",
                "content": _wrap_user_content(block if isinstance(block, str) else json.dumps(block)),
            })
    return items


def _codex_items_for_message(message: object) -> list[JsonDict]:
    if not isinstance(message, dict):
        return [{"type": "message", "role": "user", "content": _wrap_user_content(str(message))}]

    message_data = cast(JsonDict, message)
    role = string_value(message_data.get("role"), "user")
    if role == "tool":
        return [_codex_tool_output_item(message_data)]
    if role == "assistant":
        return _codex_assistant_items(message_data)
    if role == "user":
        return _codex_user_items(message_data)
    content = message_data.get("content", "")
    return [{"type": "message", "role": "user", "content": _wrap_user_content(content or str(message_data))}]


def _build_codex_input_items(messages: list[JsonDict], message_history: JsonDict) -> list[JsonDict]:
    input_items: list[JsonDict] = []
    _append_codex_context(input_items, message_history)
    first_input_content = string_value(history_section(message_history, "first_input").get("message"))
    skip_first = _first_message_matches(messages, first_input_content)
    for index, message in enumerate(messages):
        if skip_first and index == 0:
            continue
        input_items.extend(_codex_items_for_message(message))
    return input_items


def _required_non_control_tools(agent_tools: list[AgentTool]) -> list[AgentTool]:
    control_tools = {"agent_end", "stage_end", "change_stage"}
    return [
        tool for tool in agent_tools
        if getattr(tool, "required", False) and tool.name not in control_tools
    ]


def _apply_codex_tools(payload: JsonDict, model: BareBoneModel, agent_tools: list[AgentTool]) -> None:
    tool_payload = build_provider_tool_payload("codex", agent_tools)
    payload["tools"] = tool_payload.tools

    forced_tool_name = getattr(model, "forced_tool_name", None)
    if forced_tool_name and forced_tool_name in tool_payload.names:
        payload["tool_choice"] = {"type": "function", "name": forced_tool_name}
    else:
        required_tools = _required_non_control_tools(agent_tools)
        if len(required_tools) == 1:
            payload["tool_choice"] = {"type": "function", "name": required_tools[0].name}
        elif len(required_tools) > 1:
            payload["tool_choice"] = "required"
        else:
            payload["tool_choice"] = "auto"

    payload["parallel_tool_calls"] = bool(getattr(model, "parallel_tool_calls", False))


def codex_responses_fill_payload(
    model: BareBoneModel,
    messages: list[JsonDict],
    message_history: Optional[JsonDict] = None,
    agent_tools: Optional[list[AgentTool]] = None,
) -> JsonDict:
    """Build a request body for the codex Responses endpoint."""
    message_history = message_history or _default_message_history()
    instructions = string_value(history_section(message_history, "system").get("message")) or model.system_prompt or ""

    payload: JsonDict = {
        "model": model.model_id,
        "input": _build_codex_input_items(messages, message_history),
        "stream": True,   # codex backend rejects stream:false; see chat_helpers_codex
        "store": False,
    }
    if instructions:
        payload["instructions"] = instructions

    # NOTE: the codex backend rejects "max_output_tokens" with 400 (it relies on
    # plan-level budgets instead of per-request caps). Omit it entirely.

    # Reasoning effort. Codex models accept low|medium|high|xhigh.
    if getattr(model, "reasoning_effort", None):
        payload["reasoning"] = {"effort": model.reasoning_effort}

    agent_tools = model.agent_tools if agent_tools is None else agent_tools

    # Tools.
    if agent_tools:
        _apply_codex_tools(payload, model, agent_tools)

    return payload
