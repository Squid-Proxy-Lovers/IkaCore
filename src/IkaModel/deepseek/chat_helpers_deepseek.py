import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..request_interface import agent_tools_for_payload
from .deepseek import deepseek_fill_payload


def build_deepseek_request(
    barebone_model: Any,
    messages: List[dict],
    message_history: dict
) -> Tuple[str, Dict[str, str], dict]:
    payload = deepseek_fill_payload(
        barebone_model,
        messages,
        message_history,
        agent_tools=agent_tools_for_payload(barebone_model),
    )
    
    api_url = barebone_model.api_url
    headers = {
        "Authorization": f"Bearer {barebone_model.api_key}",
        "Content-Type": "application/json"
    }
    
    return api_url, headers, payload


def _unwrap_double_encoded_args(args_str: str) -> str:
    """DeepSeek V4-flash double-encodes nested object/array tool-call args.

    When a tool schema has a parameter typed as `object` (or `array`), V4
    sometimes ships it as a JSON-encoded *string* rather than a real nested
    object — e.g. for submit_trust_model:

        # what V4 sends:
        {"dispatch_id": "...", "model": "{\\"product_type\\": \\"...\\"}"}
        # what the tool expects:
        {"dispatch_id": "...", "model": {"product_type": "..."}}

    Tool handlers then reject "must be an object" and the model burns turns
    re-trying with smaller string-wrapped variants until it timeouts. Fix it
    once at the provider boundary: parse the outer args, walk one level of
    values, and unwrap any string that round-trips through json.loads into a
    dict/list. Re-serialize so downstream code (which expects the OpenAI
    spec — `arguments` is a JSON string) is unchanged.

    Strings that don't parse as JSON, or parse to scalars (numbers, bools),
    are left alone. Only dict/list payloads get unwrapped, which matches the
    schema-shape error the model is actually trying to satisfy.
    """
    if not args_str or not isinstance(args_str, str):
        return args_str
    try:
        parsed = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return args_str
    if not isinstance(parsed, dict):
        return args_str
    changed = False
    for k, v in list(parsed.items()):
        if not isinstance(v, str):
            continue
        # Cheap pre-check: only attempt re-parse on values that LOOK like
        # encoded JSON — avoids wasted json.loads on every short string.
        stripped = v.lstrip()
        if not stripped or stripped[0] not in "{[":
            continue
        try:
            inner = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(inner, (dict, list)):
            parsed[k] = inner
            changed = True
    return json.dumps(parsed) if changed else args_str


def parse_deepseek_response(data: dict, model_id: str) -> Tuple[str, Optional[str], List[dict], int]:
    message_obj = data["choices"][0]["message"]
    content = message_obj.get("content") or ""
    tool_calls = message_obj.get("tool_calls", []) or []
    tokens = data.get("usage", {}).get("total_tokens", 0)
    reasoning_content = message_obj.get("reasoning_content")

    # Repair V4-flash's double-encoded nested object/array args in place so
    # downstream tool dispatch sees the canonical shape.
    for tc in tool_calls:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict):
            continue
        fn["arguments"] = _unwrap_double_encoded_args(fn.get("arguments", "{}"))

    return content, reasoning_content, tool_calls, tokens


def append_deepseek_tool_messages(
    messages: List[dict],
    message_history: dict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: List[dict],
    tool_messages: List[dict],
    tokens: int,
    repeated_warning_msg: str = ""
) -> None:
    assistant_msg = {"role": "assistant", "content": content}
    if reasoning_content:
        assistant_msg["reasoning_content"] = reasoning_content
    if executed_tool_call_list:
        assistant_msg["tool_calls"] = executed_tool_call_list
    
    messages.append(assistant_msg)
    messages.extend(tool_messages)
    
    if repeated_warning_msg:
        messages.append({"role": "user", "content": repeated_warning_msg})
    
    if executed_tool_call_list:
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {  # type: ignore
            "message": json.dumps(assistant_msg),
            "tokens": tokens,
            "type": "assistant_with_tools",
        }
    
    for tool_msg in tool_messages:
        msg_id = str(uuid.uuid4())
        message_history["messages"][msg_id] = {  # type: ignore
            "message": json.dumps(tool_msg),
            "tokens": 0,
            "type": "tool",
        }
