from collections.abc import Callable
from typing import Any, Optional

from IkaCore.agent_runtime_payloads import JsonDict

from .anthropic.chat_helpers_anthropic import (
    append_anthropic_tool_messages,
    build_anthropic_request,
    parse_anthropic_response,
)
from .codex.chat_helpers_codex import (
    append_codex_tool_messages,
    build_codex_request,
    parse_codex_response,
)
from .deepseek.chat_helpers_deepseek import (
    append_deepseek_tool_messages,
    build_deepseek_request,
    parse_deepseek_response,
)
from .gemini.chat_helpers_gemini import (
    append_gemini_tool_messages,
    build_gemini_request,
    parse_gemini_response,
)
from .openai.chat_helpers_openai import (
    append_openai_responses_tool_messages,
    append_openai_tool_messages,
    build_openai_request,
    build_openai_responses_request,
    parse_openai_response,
    parse_openai_responses_response,
)
from .openrouter.chat_helpers_openrouter import (
    append_openrouter_tool_messages,
    build_openrouter_request,
    parse_openrouter_response,
)

BuildRequestFn = Callable[[Any, list[JsonDict], JsonDict], tuple[str, dict[str, str], JsonDict]]
ParseResponseFn = Callable[[JsonDict, str], tuple[str, Optional[str], list[JsonDict], int]]
GeminiResultFormatterFn = Callable[[list[JsonDict], list[str]], list[JsonDict]]
ToolAppenderFn = Callable[
    [
        list[JsonDict],
        JsonDict,
        str,
        Optional[str],
        list[JsonDict],
        list[JsonDict],
        list[str],
        int,
        str,
        Optional[GeminiResultFormatterFn],
    ],
    None,
]
StandardToolAppenderFn = Callable[
    [list[JsonDict], JsonDict, str, Optional[str], list[JsonDict], list[JsonDict], int, str],
    None,
]


PROVIDER_REQUEST_BUILDERS: dict[str, BuildRequestFn] = {
    "anthropic": build_anthropic_request,
    "codex": build_codex_request,
    "deepseek": build_deepseek_request,
    "gemini": build_gemini_request,
    "openai": build_openai_request,
    "openai_responses": build_openai_responses_request,
    "openrouter": build_openrouter_request,
}

PROVIDER_RESPONSE_PARSERS: dict[str, ParseResponseFn] = {
    "anthropic": parse_anthropic_response,
    "codex": parse_codex_response,
    "deepseek": parse_deepseek_response,
    "gemini": parse_gemini_response,
    "openai": parse_openai_response,
    "openai_responses": parse_openai_responses_response,
    "openrouter": parse_openrouter_response,
}


def _standard_tool_appender(append_fn: StandardToolAppenderFn) -> ToolAppenderFn:
    def append(
        messages: list[JsonDict],
        message_history: JsonDict,
        content: str,
        reasoning_content: Optional[str],
        executed_tool_call_list: list[JsonDict],
        tool_messages: list[JsonDict],
        tool_results: list[str],
        tokens: int,
        repeated_warning_msg: str = "",
        format_gemini_results_fn: Optional[GeminiResultFormatterFn] = None,
    ) -> None:
        append_fn(
            messages, message_history, content, reasoning_content,
            executed_tool_call_list, tool_messages, tokens, repeated_warning_msg
        )

    return append


def _append_gemini_provider_tools(
    messages: list[JsonDict],
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: list[JsonDict],
    tool_messages: list[JsonDict],
    tool_results: list[str],
    tokens: int,
    repeated_warning_msg: str = "",
    format_gemini_results_fn: Optional[GeminiResultFormatterFn] = None,
) -> None:
    append_gemini_tool_messages(
        messages, message_history, content, reasoning_content,
        executed_tool_call_list, tool_results, tokens, repeated_warning_msg,
        format_gemini_results_fn
    )


PROVIDER_TOOL_APPENDERS: dict[str, ToolAppenderFn] = {
    "anthropic": _standard_tool_appender(append_anthropic_tool_messages),
    "codex": _standard_tool_appender(append_codex_tool_messages),
    "deepseek": _standard_tool_appender(append_deepseek_tool_messages),
    "gemini": _append_gemini_provider_tools,
    "openai": _standard_tool_appender(append_openai_tool_messages),
    "openai_responses": _standard_tool_appender(append_openai_responses_tool_messages),
    "openrouter": _standard_tool_appender(append_openrouter_tool_messages),
}


def build_provider_request(
    provider: str,
    barebone_model: Any,
    messages: list[JsonDict],
    message_history: JsonDict
) -> tuple[str, dict[str, str], JsonDict]:
    builder = PROVIDER_REQUEST_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return builder(barebone_model, messages, message_history)


def parse_provider_response(
    provider: str,
    data: JsonDict,
    model_id: str
) -> tuple[str, Optional[str], list[JsonDict], int]:
    parser = PROVIDER_RESPONSE_PARSERS.get(provider)
    if parser is None:
        return "", None, [], 0
    return parser(data, model_id)


def append_provider_tool_messages(
    provider: str,
    messages: list[JsonDict],
    message_history: JsonDict,
    content: str,
    reasoning_content: Optional[str],
    executed_tool_call_list: list[JsonDict],
    tool_messages: list[JsonDict],
    tool_results: list[str],
    tokens: int,
    repeated_warning_msg: str = "",
    format_gemini_results_fn: Optional[GeminiResultFormatterFn] = None
) -> None:
    appender = PROVIDER_TOOL_APPENDERS.get(provider)
    if appender is None:
        return
    appender(
        messages,
        message_history,
        content,
        reasoning_content,
        executed_tool_call_list,
        tool_messages,
        tool_results,
        tokens,
        repeated_warning_msg,
        format_gemini_results_fn,
    )
