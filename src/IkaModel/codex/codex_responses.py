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
from typing import Any, Dict, List, Optional

from ..openai.openai_responses import _build_parameters


# Codex backend model slugs (per codex-rs/models-manager/models.json).
# Adding new entries is harmless — the backend will reject unknowns with a
# clear 400, but we surface a friendlier message client-side when we know.
CODEX_KNOWN_MODELS = {
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.2",
    "gpt-5.2-codex",
}


def _wrap_user_content(content: Any) -> List[Dict[str, Any]]:
    """Convert a string or pre-wrapped list into the input_text content shape."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, list):
        # Already a list — assume the caller knows what they're doing, but
        # rewrap any bare strings inside.
        out = []
        for block in content:
            if isinstance(block, str):
                out.append({"type": "input_text", "text": block})
            elif isinstance(block, dict):
                out.append(block)
            else:
                out.append({"type": "input_text", "text": str(block)})
        return out
    return [{"type": "input_text", "text": str(content)}]


def _wrap_assistant_content(content: Any) -> List[Dict[str, Any]]:
    """Assistant turns use ``output_text`` blocks in the Responses API."""
    if isinstance(content, str):
        return [{"type": "output_text", "text": content}]
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, str):
                out.append({"type": "output_text", "text": block})
            elif isinstance(block, dict):
                out.append(block)
            else:
                out.append({"type": "output_text", "text": str(block)})
        return out
    return [{"type": "output_text", "text": str(content)}]


def codex_responses_fill_payload(
    model,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a request body for the codex Responses endpoint."""
    message_history = message_history or {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }

    # System prompt → top-level instructions (required by codex backend).
    instructions = message_history.get("system", {}).get("message") or model.system_prompt or ""

    input_items: List[Dict[str, Any]] = []

    # Summary of older context (after compaction) — emit as an assistant turn.
    summary_msg = message_history.get("summary", {}).get("message")
    if summary_msg:
        input_items.append({
            "type": "message",
            "role": "assistant",
            "content": _wrap_assistant_content(summary_msg),
        })

    # Original first input (preserved across compactions). Dedup against
    # messages[0] when they match verbatim so we don't double-prompt.
    first_input_content = message_history.get("first_input", {}).get("message", "")
    skip_first = False
    if first_input_content and messages:
        first_msg = messages[0]
        if isinstance(first_msg, dict):
            cmp = first_msg.get("content", "")
        else:
            cmp = str(first_msg)
        if isinstance(cmp, str) and cmp == first_input_content:
            skip_first = True

    if first_input_content:
        input_items.append({
            "type": "message",
            "role": "user",
            "content": _wrap_user_content(first_input_content),
        })

    # Append the running conversation.
    for i, msg in enumerate(messages):
        if skip_first and i == 0:
            continue

        if not isinstance(msg, dict):
            input_items.append({
                "type": "message",
                "role": "user",
                "content": _wrap_user_content(str(msg)),
            })
            continue

        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": content if isinstance(content, str) else json.dumps(content),
            })
        elif role == "assistant":
            tool_calls = msg.get("tool_calls", []) or []
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    })
            else:
                if content:
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": _wrap_assistant_content(content),
                    })
        elif role == "user":
            if isinstance(content, list):
                # Anthropic-style content list (may include tool_result blocks).
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": block.get("tool_use_id", ""),
                            "output": block.get("content", ""),
                        })
                    else:
                        input_items.append({
                            "type": "message",
                            "role": "user",
                            "content": _wrap_user_content(
                                block if isinstance(block, str) else json.dumps(block)
                            ),
                        })
            else:
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": _wrap_user_content(content or str(msg)),
                })
        else:
            # Unknown role — coerce to user so we don't lose the content.
            input_items.append({
                "type": "message",
                "role": "user",
                "content": _wrap_user_content(content or str(msg)),
            })

    payload: Dict[str, Any] = {
        "model": model.model_id,
        "input": input_items,
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

    # Tools.
    if getattr(model, "agent_tools", None):
        tools = []
        tool_names = set()
        for tool in model.agent_tools:
            tool_names.add(tool.name)
            parameters = _build_parameters(tool)
            # Ensure additionalProperties: false at object roots so callers that
            # opt into strict mode later don't trip the codex backend.
            if isinstance(parameters, dict) and parameters.get("type") == "object":
                parameters.setdefault("additionalProperties", False)
            tools.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            })
        payload["tools"] = tools

        # Control tools (agent_end / stage_end / change_stage) are registered
        # by IkaBaseAgent with required=True so the loop knows they exist. If
        # they're the only "required" tools we see, we must NOT pin tool_choice
        # to one of them — that forces the model to terminate on turn 0 before
        # it gets a chance to call real tools. Only let *non-control* required
        # tools drive the forced-choice path.
        _CONTROL_TOOLS = {"agent_end", "stage_end", "change_stage"}

        forced_tool_name = getattr(model, "forced_tool_name", None)
        if forced_tool_name and forced_tool_name in tool_names:
            payload["tool_choice"] = {"type": "function", "name": forced_tool_name}
        else:
            required_tools = [
                t for t in model.agent_tools
                if getattr(t, "required", False) and t.name not in _CONTROL_TOOLS
            ]
            if len(required_tools) == 1:
                payload["tool_choice"] = {"type": "function", "name": required_tools[0].name}
            elif len(required_tools) > 1:
                payload["tool_choice"] = "required"
            else:
                payload["tool_choice"] = "auto"

        payload["parallel_tool_calls"] = bool(getattr(model, "parallel_tool_calls", False))

    return payload
