# pyright: strict
# pyright: reportUnusedFunction=false

import json
import uuid
from typing import Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict, history_section, json_dict, string_value

from ..base import BareBoneModel
from ..request_interface import agent_tools_for_payload
from .deepseek import deepseek_fill_payload

ProviderRequest = tuple[str, dict[str, str], JsonDict]
ProviderRound = tuple[str, Optional[str], list[JsonDict], int]
MessageList = list[JsonDict]


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def build_deepseek_request(
    barebone_model: BareBoneModel,
    messages: MessageList,
    message_history: JsonDict
) -> ProviderRequest:
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
    if not args_str:
        return args_str
    try:
        parsed = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return args_str
    if not isinstance(parsed, dict):
        return args_str
    parsed_data = cast(JsonDict, parsed)
    changed = False
    for k, v in list(parsed_data.items()):
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
            parsed_data[k] = inner
            changed = True
    return json.dumps(parsed_data) if changed else args_str


def parse_deepseek_response(data: JsonDict, model_id: str) -> ProviderRound:
    choices = cast(list[JsonDict], data["choices"])
    message_obj = json_dict(choices[0].get("message"))
    content = string_value(message_obj.get("content"))
    raw_tool_calls: object = message_obj.get("tool_calls", []) or []
    tool_calls = cast(list[JsonDict], raw_tool_calls) if isinstance(raw_tool_calls, list) else []
    tokens = _token_count(json_dict(data.get("usage")).get("total_tokens"))
    reasoning_value = message_obj.get("reasoning_content")
    reasoning_content = string_value(reasoning_value) if reasoning_value else None

    # Repair V4-flash's double-encoded nested object/array args in place so
    # downstream tool dispatch sees the canonical shape.
    for tc in tool_calls:
        fn = json_dict(tc.get("function"))
        if not fn:
            continue
        fn["arguments"] = _unwrap_double_encoded_args(string_value(fn.get("arguments"), "{}"))

    return content, reasoning_content, tool_calls, tokens


def append_deepseek_tool_messages(
    messages: MessageList,
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: list[JsonDict],
    tool_messages: MessageList,
    tokens: int,
    repeated_warning_msg: str = ""
) -> None:
    assistant_msg: JsonDict = {"role": "assistant", "content": content}
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
