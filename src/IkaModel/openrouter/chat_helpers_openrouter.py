import json
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

from .openrouter import openrouter_fill_payload
from ..request_interface import _apply_tools_filter_for_payload, _restore_tools_after_payload

LOG = logging.getLogger(__name__)


def build_openrouter_request(
    barebone_model: Any,
    messages: List[dict],
    message_history: dict
) -> Tuple[str, Dict[str, str], dict]:
    """Build OpenRouter API request (URL, headers, payload)."""
    _apply_tools_filter_for_payload(barebone_model)

    # Extract OpenRouter-specific config if present
    plugins = getattr(barebone_model, 'openrouter_plugins', None)
    response_format = getattr(barebone_model, 'openrouter_response_format', None)

    payload = openrouter_fill_payload(
        barebone_model, messages, message_history,
        plugins=plugins, response_format=response_format
    )
    _restore_tools_after_payload(barebone_model)

    # Build headers
    headers = {
        "Authorization": f"Bearer {barebone_model.api_key}",
        "Content-Type": "application/json"
    }

    # Add optional attribution headers
    if hasattr(barebone_model, 'http_referer'):
        headers["HTTP-Referer"] = barebone_model.http_referer
    if hasattr(barebone_model, 'x_title'):
        headers["X-Title"] = barebone_model.x_title

    return barebone_model.api_url, headers, payload


def parse_openrouter_response(data: dict, model_id: str) -> Tuple[str, Optional[str], List[dict], int]:
    """Parse OpenRouter API response (OpenAI-compatible with cost field)."""
    # OpenRouter can return 200 with an error body when the upstream provider fails (e.g. 502)
    if "error" in data:
        err = data["error"]
        msg = err.get("message", "Unknown error")
        meta = err.get("metadata", {}) or {}
        raw = meta.get("raw", "")
        provider = meta.get("provider_name", "")
        detail = f"{msg}"
        if provider:
            detail += f" (provider: {provider})"
        if raw:
            detail += f" — {raw.strip()}"
        raise ValueError(f"OpenRouter returned error: {detail}")
    if "choices" not in data or not data["choices"]:
        raise ValueError("OpenRouter response missing 'choices'")
    message_obj = data["choices"][0]["message"]
    content = message_obj.get("content") or ""
    tool_calls = message_obj.get("tool_calls", []) or []

    # Extract usage info
    usage = data.get("usage", {})
    tokens = usage.get("total_tokens", 0)
    cost = usage.get("cost", 0.0)  # OpenRouter-specific

    # Log cost for tracking
    if cost > 0:
        LOG.debug(f"OpenRouter request cost: ${cost:.6f}")

    # Check for reasoning content (o1/o3 models + OpenRouter reasoning field)
    reasoning_content = message_obj.get("reasoning_content") or message_obj.get("reasoning")

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
            except Exception:
                pass

    return content, reasoning_content, tool_calls, tokens


def append_openrouter_tool_messages(
    messages: List[dict],
    message_history: dict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: List[dict],
    tool_messages: List[dict],
    tokens: int,
    repeated_warning_msg: str = ""
) -> None:
    """Append tool messages to conversation history (OpenAI-compatible)."""
    assistant_msg = {"role": "assistant", "content": content}
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
        message_history["messages"][msg_id] = {
            "message": json.dumps(assistant_msg),
            "tokens": tokens,
            "type": "assistant_with_tools",
        }

    for tool_msg in tool_messages:
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {
            "message": json.dumps(tool_msg),
            "tokens": 0,
            "type": "tool",
        }
