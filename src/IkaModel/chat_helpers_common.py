from typing import Any, Callable, Dict, List, Optional, Tuple

from .chat_helpers_deepseek import build_deepseek_request, parse_deepseek_response, append_deepseek_tool_messages
from .chat_helpers_openai import build_openai_request, parse_openai_response, append_openai_tool_messages
from .chat_helpers_anthropic import build_anthropic_request, parse_anthropic_response, append_anthropic_tool_messages
from .chat_helpers_gemini import build_gemini_request, parse_gemini_response, append_gemini_tool_messages


def build_provider_request(
    provider: str,
    barebone_model: Any,
    messages: List[dict],
    message_history: dict
) -> Tuple[str, Dict[str, str], dict]:
    if provider == "deepseek":
        return build_deepseek_request(barebone_model, messages, message_history)
    elif provider == "openai":
        return build_openai_request(barebone_model, messages, message_history)
    elif provider == "anthropic":
        return build_anthropic_request(barebone_model, messages, message_history)
    elif provider == "gemini":
        return build_gemini_request(barebone_model, messages, message_history)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def parse_provider_response(
    provider: str,
    data: dict,
    model_id: str
) -> Tuple[str, Optional[str], List[dict], int]:
    if provider == "deepseek":
        return parse_deepseek_response(data, model_id)
    elif provider == "openai":
        return parse_openai_response(data, model_id)
    elif provider == "anthropic":
        return parse_anthropic_response(data, model_id)
    elif provider == "gemini":
        return parse_gemini_response(data, model_id)
    else:
        return "", None, [], 0


def append_provider_tool_messages(
    provider: str,
    messages: List[dict],
    message_history: dict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: List[dict],
    tool_messages: List[dict],
    tool_results: List[str],
    tokens: int,
    repeated_warning_msg: str = "",
    format_gemini_results_fn: Optional[Callable] = None
) -> None:
    if provider == "deepseek":
        append_deepseek_tool_messages(
            messages, message_history, content, reasoning_content,
            executed_tool_call_list, tool_messages, tokens, repeated_warning_msg
        )
    elif provider == "openai":
        append_openai_tool_messages(
            messages, message_history, content, reasoning_content,
            executed_tool_call_list, tool_messages, tokens, repeated_warning_msg
        )
    elif provider == "anthropic":
        append_anthropic_tool_messages(
            messages, message_history, content, reasoning_content,
            executed_tool_call_list, tool_messages, tokens, repeated_warning_msg
        )
    elif provider == "gemini":
        append_gemini_tool_messages(
            messages, message_history, content, reasoning_content,
            executed_tool_call_list, tool_results, tokens, repeated_warning_msg,
            format_gemini_results_fn
        )
