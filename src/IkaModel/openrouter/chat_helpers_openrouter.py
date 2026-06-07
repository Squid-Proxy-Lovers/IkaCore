# pyright: strict
# pyright: reportUnusedFunction=false

import json
import logging
import uuid
from typing import Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value

from ..base import BareBoneModel
from ..request_interface import agent_tools_for_payload
from .openrouter import openrouter_fill_payload

LOG = logging.getLogger(__name__)
ProviderRequest = tuple[str, dict[str, str], JsonDict]
ProviderRound = tuple[str, Optional[str], list[JsonDict], int]
MessageList = list[JsonDict]


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def build_openrouter_request(
    barebone_model: BareBoneModel,
    messages: MessageList,
    message_history: JsonDict
) -> ProviderRequest:
    """Build OpenRouter API request (URL, headers, payload)."""
    # Extract OpenRouter-specific config if present
    plugins = getattr(barebone_model, 'openrouter_plugins', None)
    response_format = getattr(barebone_model, 'openrouter_response_format', None)

    payload = openrouter_fill_payload(
        barebone_model, messages, message_history,
        plugins=plugins, response_format=response_format,
        agent_tools=agent_tools_for_payload(barebone_model),
    )

    # Build headers
    headers = {
        "Authorization": f"Bearer {barebone_model.api_key}",
        "Content-Type": "application/json"
    }

    # Add optional attribution headers
    http_referer = getattr(barebone_model, 'http_referer', None)
    if isinstance(http_referer, str):
        headers["HTTP-Referer"] = http_referer
    x_title = getattr(barebone_model, 'x_title', None)
    if isinstance(x_title, str):
        headers["X-Title"] = x_title

    return barebone_model.api_url, headers, payload


def parse_openrouter_response(data: JsonDict, model_id: str) -> ProviderRound:
    """Parse OpenRouter API response (OpenAI-compatible with cost field)."""
    # OpenRouter can return 200 with an error body when the upstream provider fails (e.g. 502)
    if "error" in data:
        err = json_dict(data.get("error"))
        msg = string_value(err.get("message"), "Unknown error")
        meta = json_dict(err.get("metadata"))
        raw = string_value(meta.get("raw"))
        provider = string_value(meta.get("provider_name"))
        detail = f"{msg}"
        if provider:
            detail += f" (provider: {provider})"
        if raw:
            detail += f" — {raw.strip()}"
        raise ValueError(f"OpenRouter returned error: {detail}")
    if "choices" not in data or not data["choices"]:
        raise ValueError("OpenRouter response missing 'choices'")
    choices = cast(list[JsonDict], data["choices"])
    message_obj = json_dict(choices[0].get("message"))
    content = string_value(message_obj.get("content"))
    raw_tool_calls: object = message_obj.get("tool_calls", []) or []
    tool_calls = cast(list[JsonDict], raw_tool_calls) if isinstance(raw_tool_calls, list) else []

    # Extract usage info
    usage = json_dict(data.get("usage"))
    tokens = _token_count(usage.get("total_tokens"))
    cost = usage.get("cost", 0.0)  # OpenRouter-specific

    # Log cost for tracking
    if isinstance(cost, (int, float)) and cost > 0:
        LOG.debug(f"OpenRouter request cost: ${cost:.6f}")

    # Check for reasoning content (o1/o3 models + OpenRouter reasoning field)
    reasoning_value = message_obj.get("reasoning_content") or message_obj.get("reasoning")
    reasoning_content = string_value(reasoning_value) if reasoning_value else None

    # Log GLM reasoning to file for analysis
    if reasoning_content and "glm" in (model_id or "").lower():
        import os as _os
        log_path = _os.environ.get("GLM_REASONING_LOG")
        if log_path:
            try:
                with open(log_path, "a") as f:
                    f.write(f"\n===== {model_id} =====\n")
                    f.write(reasoning_content[:4000])
                    f.write(f"\n---\ntool_calls: {len(tool_calls)}\n")
                    f.write("=" * 40 + "\n")
            except (OSError, TypeError, ValueError):
                pass

    return content, reasoning_content, tool_calls, tokens


def append_openrouter_tool_messages(
    messages: MessageList,
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: list[JsonDict],
    tool_messages: MessageList,
    tokens: int,
    repeated_warning_msg: str = ""
) -> None:
    """Append tool messages to conversation history (OpenAI-compatible)."""
    assistant_msg: JsonDict = {"role": "assistant", "content": content}
    if reasoning_content:
        assistant_msg["reasoning_content"] = reasoning_content
    if executed_tool_call_list:
        assistant_msg["tool_calls"] = executed_tool_call_list

    messages.append(assistant_msg)
    messages.extend(tool_messages)

    if repeated_warning_msg:
        messages.append({"role": "user", "content": repeated_warning_msg})

    # Update message history
    if executed_tool_call_list:
        msg_id = str(uuid.uuid4())
        history_section(message_history, "messages")[msg_id] = {
            "message": json.dumps(assistant_msg),
            "tokens": tokens,
            "type": "assistant_with_tools",
        }

    for tool_msg in tool_messages:
        msg_id = str(uuid.uuid4())
        history_section(message_history, "messages")[msg_id] = {
            "message": json.dumps(tool_msg),
            "tokens": 0,
            "type": "tool",
        }
