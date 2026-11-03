# pyright: strict
# pyright: reportUnusedFunction=false

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Optional, cast

import httpx

from IkaCore.agent_runtime_payloads import JsonDict, history_message_text, history_section, json_dict, string_value

from .base import BareBoneModel
from .model_metadata import (
    OPENAI_CHAT_COMPLETIONS_URL,
    get_api_url_for_model,
    get_summary_model_for_provider,
    supports_custom_temperature,
    uses_openai_max_completion_tokens,
)
from .request_interface import api_request_retry, async_api_request_retry, get_provider

_LOG = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(relative_path: str, fallback: str) -> str:
    try:
        from src.resources import read_text as read_ika_resource

        return read_ika_resource(f"IkaCore/src/IkaModel/prompts/{relative_path}").strip()
    except (ImportError, ModuleNotFoundError, OSError):
        pass
    path = _PROMPTS_DIR / relative_path
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return fallback


SUMMARY_PROMPT = _load_prompt(
    "default_summary_system.txt",
    "Please summarize the following conversation history concisely, preserving key information and context.",
)
DEFAULT_SUMMARY_USER_PREFIX = _load_prompt(
    "default_summary_user_prefix.txt",
    "Please summarize the following conversation history:\n\n",
)
FORCE_ANSWER_SYSTEM = _load_prompt(
    "force_answer_system.txt",
    "You are a summarization assistant. Given a conversation and the original task, produce a single final response "
    "that directly answers the original prompt. Do not summarize loosely; answer as if the task had been completed. "
    "Output only the final answer the agent would give, with no meta-commentary.",
)
FORCE_ANSWER_USER_PREFIX = _load_prompt(
    "force_answer_user_prefix.txt",
    "Original task and conversation:\n\n",
)
WHAT_REMAINS_SYSTEM = _load_prompt(
    "what_remains_system.txt",
    "You are a triage assistant for incomplete agent runs. Output a short actionable brief in three parts: "
    "DONE, REMAINING, and NEXT. No preamble. Output only the three-part brief.",
)
WHAT_REMAINS_USER_PREFIX = _load_prompt(
    "what_remains_user_prefix.txt",
    "Below is the agent run that hit its step limit. Extract DONE / REMAINING / NEXT as specified.\n\n",
)


def _get_prompts_for_kind(prompt_kind: str) -> tuple[str, str]:
    if prompt_kind == "force_answer":
        return FORCE_ANSWER_SYSTEM, FORCE_ANSWER_USER_PREFIX
    if prompt_kind == "what_remains":
        return WHAT_REMAINS_SYSTEM, WHAT_REMAINS_USER_PREFIX
    return SUMMARY_PROMPT, DEFAULT_SUMMARY_USER_PREFIX


def _normalize_provider_for_summary(provider: str) -> str:
    """Normalize provider for summarization.

    Historically summarization always used Chat Completions, so ``openai_responses``
    was collapsed onto ``openai``. The codex backend has no Chat Completions
    surface — it speaks Responses-only — so ``codex`` stays as itself and gets
    its own payload/parse paths below.
    """
    if provider == "openai_responses":
        return "openai"
    return provider


def get_summary_model(provider: str) -> tuple[Optional[str], Optional[str]]:
    provider = _normalize_provider_for_summary(provider)
    return get_summary_model_for_provider(provider)


def _chat_summary_messages(sys_prompt: str, user_content: str) -> list[JsonDict]:
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]


def _bearer_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _build_deepseek_summary_payload(
    model_name: str,
    api_key: str,
    sys_prompt: str,
    user_content: str,
) -> tuple[JsonDict, dict[str, str]]:
    return {
        "model": model_name,
        "messages": _chat_summary_messages(sys_prompt, user_content),
        "temperature": 0.3,
        "max_tokens": 2000,
        "stream": False,
    }, _bearer_headers(api_key)


def _build_openai_like_summary_payload(
    model_name: str,
    api_key: str,
    sys_prompt: str,
    user_content: str,
) -> tuple[JsonDict, dict[str, str]]:
    token_key = "max_completion_tokens" if uses_openai_max_completion_tokens(model_name) else "max_tokens"
    payload: JsonDict = {
        "model": model_name,
        "messages": _chat_summary_messages(sys_prompt, user_content),
        token_key: 2000,
    }
    if supports_custom_temperature(model_name):
        payload["temperature"] = 0.3
    return payload, _bearer_headers(api_key)


def _build_anthropic_summary_payload(
    model_name: str,
    api_key: str,
    sys_prompt: str,
    user_content: str,
) -> tuple[JsonDict, dict[str, str]]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    return {
        "model": model_name,
        "max_tokens": 2000,
        "system": sys_prompt,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.3,
    }, headers


def _build_gemini_summary_payload(
    model_name: str,
    api_key: str,
    sys_prompt: str,
    user_content: str,
) -> tuple[JsonDict, dict[str, str]]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    return {
        "contents": [{
            "parts": [{"text": f"{sys_prompt}\n\n{user_content}"}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2000,
        },
    }, headers


def _build_codex_summary_payload(
    model_name: str,
    api_key: str,
    sys_prompt: str,
    user_content: str,
) -> tuple[JsonDict, dict[str, str]]:
    headers = _bearer_headers(api_key)
    headers["Accept"] = "text/event-stream"
    return {
        "model": model_name,
        "instructions": sys_prompt,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": user_content}],
        }],
        "stream": True,
        "store": False,
    }, headers


SummaryPayloadBuilder = Callable[[str, str, str, str], tuple[JsonDict, dict[str, str]]]
SUMMARY_PAYLOAD_BUILDERS: dict[str, SummaryPayloadBuilder] = {
    "anthropic": _build_anthropic_summary_payload,
    "codex": _build_codex_summary_payload,
    "deepseek": _build_deepseek_summary_payload,
    "gemini": _build_gemini_summary_payload,
    "openai": _build_openai_like_summary_payload,
    "openrouter": _build_openai_like_summary_payload,
}


def create_summary_payload(
    provider: str,
    model_name: str,
    api_key: str,
    conversation_text: str,
    system_prompt: Optional[str] = None,
    user_prompt_prefix: Optional[str] = None,
) -> tuple[JsonDict, dict[str, str]]:
    sys_prompt = system_prompt if system_prompt is not None else SUMMARY_PROMPT
    user_prefix = user_prompt_prefix if user_prompt_prefix is not None else DEFAULT_SUMMARY_USER_PREFIX
    user_content = f"{user_prefix}{conversation_text}"

    provider = _normalize_provider_for_summary(provider)
    builder = SUMMARY_PAYLOAD_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(f"Unsupported provider: {provider}")

    return builder(model_name, api_key, sys_prompt, user_content)


def _parse_chat_summary_response(data: JsonDict) -> str:
    return data["choices"][0]["message"]["content"]


def _parse_anthropic_summary_response(data: JsonDict) -> str:
    content_blocks: object = data.get("content", [])
    blocks = cast(list[object], content_blocks) if isinstance(content_blocks, list) else []
    text_parts = [
        string_value(block.get("text"))
        for block in (json_dict(item) for item in blocks)
        if block.get("type") == "text"
    ]
    return "".join(text_parts)


def _parse_gemini_summary_response(data: JsonDict) -> str:
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _parse_codex_summary_response(data: JsonDict) -> str:
    parts: list[str] = []
    output_items: object = data.get("output", []) or []
    items = cast(list[object], output_items) if isinstance(output_items, list) else []
    for item in items:
        item_data = json_dict(item)
        if item_data.get("type") == "message":
            content_blocks: object = item_data.get("content", []) or []
            blocks = cast(list[object], content_blocks) if isinstance(content_blocks, list) else []
            for block in blocks:
                block_data = json_dict(block)
                if block_data.get("type") == "output_text":
                    parts.append(string_value(block_data.get("text")))
    return "".join(parts) or string_value(data.get("output_text"))


SummaryResponseParser = Callable[[JsonDict], str]
SUMMARY_RESPONSE_PARSERS: dict[str, SummaryResponseParser] = {
    "anthropic": _parse_anthropic_summary_response,
    "codex": _parse_codex_summary_response,
    "deepseek": _parse_chat_summary_response,
    "gemini": _parse_gemini_summary_response,
    "openai": _parse_chat_summary_response,
    "openrouter": _parse_chat_summary_response,
}


def parse_summary_response(provider: str, response: httpx.Response) -> str:
    data = json_dict(response.json())
    provider = _normalize_provider_for_summary(provider)
    parser = SUMMARY_RESPONSE_PARSERS.get(provider)
    if parser is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return parser(data)


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _usage_total_tokens(data: JsonDict) -> int:
    usage = json_dict(data.get("usage"))
    return _token_count(usage.get("total_tokens"))


def _usage_anthropic_tokens(data: JsonDict) -> int:
    usage = json_dict(data.get("usage"))
    return _token_count(usage.get("input_tokens")) + _token_count(usage.get("output_tokens"))


def _usage_gemini_tokens(data: JsonDict) -> int:
    usage = json_dict(data.get("usageMetadata"))
    return _token_count(usage.get("totalTokenCount"))


def _usage_codex_tokens(data: JsonDict) -> int:
    usage = json_dict(data.get("usage"))
    return _token_count(usage.get(
        "total_tokens",
        (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0),
    ))


SummaryTokenExtractor = Callable[[JsonDict], int]
SUMMARY_TOKEN_EXTRACTORS: dict[str, SummaryTokenExtractor] = {
    "anthropic": _usage_anthropic_tokens,
    "codex": _usage_codex_tokens,
    "deepseek": _usage_total_tokens,
    "gemini": _usage_gemini_tokens,
    "openai": _usage_total_tokens,
    "openrouter": _usage_total_tokens,
}


def extract_summary_tokens(provider: str, data: JsonDict) -> int:
    provider = _normalize_provider_for_summary(provider)
    extractor = SUMMARY_TOKEN_EXTRACTORS.get(provider)
    return extractor(data) if extractor else 0


def _resolve_summary_target(
    barebone_model: BareBoneModel,
    provider: str,
    use_same_model: bool,
) -> tuple[Optional[str], Optional[str]]:
    if not use_same_model:
        return get_summary_model(provider)

    model_name = barebone_model.model_id
    api_url = barebone_model.api_url
    if not api_url:
        if provider == "codex":
            _, api_url = get_summary_model(provider)
        else:
            api_url = get_api_url_for_model(model_name, use_responses_api=False)
    elif "responses" in api_url and provider != "codex":
        api_url = OPENAI_CHAT_COMPLETIONS_URL

    return model_name, api_url


def _write_summary_to_history(message_history: JsonDict, summary: str, summary_tokens: int) -> None:
    summary_section = history_section(message_history, "summary")
    summary_section["message"] = f"[SUMMARY]\n{summary}"
    summary_section["tokens"] = summary_tokens
    message_history["messages"] = {}
    _LOG.info(
        "Message history summarized. Kept: system prompt, first input, and summary. "
        "Cleared all other messages."
    )


def get_conversation_text(message_history: JsonDict) -> str:
    parts: list[str] = []
    first_input = history_message_text(message_history, "first_input")
    if first_input:
        parts.append(first_input)
    summary = history_message_text(message_history, "summary")
    if summary:
        parts.append(summary)
    for message_entry in history_section(message_history, "messages").values():
        message = json_dict(message_entry).get("message")
        if message:
            parts.append(string_value(message))
    return "\n".join(parts)


def run_summarization(
    barebone_model: BareBoneModel,
    message_history: JsonDict,
    prompt_kind: str = "default",
    write_to_history: bool = True,
    use_same_model: bool = True,
    client: Optional[httpx.Client] = None,
) -> str:
    if not history_message_text(message_history, "first_input") and not history_section(message_history, "messages"):
        return ""

    conversation_text = get_conversation_text(message_history)
    system_prompt, user_prompt_prefix = _get_prompts_for_kind(prompt_kind)

    provider = _normalize_provider_for_summary(
        get_provider(barebone_model.model_id, barebone_model.api_url)
    )

    model_name, api_url = _resolve_summary_target(barebone_model, provider, use_same_model)
    if not model_name or not api_url:
        _LOG.warning(f"Could not determine summarization model for provider: {provider}")
        return ""

    payload, headers = create_summary_payload(
        provider,
        model_name,
        barebone_model.api_key,
        conversation_text,
        system_prompt=system_prompt,
        user_prompt_prefix=user_prompt_prefix,
    )

    try:
        response = api_request_retry(api_url, headers, payload, max_retries=3, wait_seconds=10, client=client)
        response.raise_for_status()
        data = json_dict(response.json())
        summary = parse_summary_response(provider, response)
        summary_tokens = extract_summary_tokens(provider, data)

        if write_to_history:
            _write_summary_to_history(message_history, summary, summary_tokens)

        return summary
    except httpx.HTTPError as e:
        _LOG.error(f"Failed to summarize message history: {e}")
        return ""
    except (RuntimeError, ValueError, TypeError, KeyError) as e:
        _LOG.error(f"Unexpected error during summarization: {e}")
        return ""


def summarise_message_history(
    barebone_model: BareBoneModel,
    message_history: JsonDict,
    client: Optional[httpx.Client] = None,
) -> str:
    return run_summarization(
        barebone_model,
        message_history,
        prompt_kind="default",
        write_to_history=True,
        client=client,
    )


async def async_summarise_message_history(
    barebone_model: BareBoneModel,
    message_history: JsonDict,
    client: Optional[httpx.AsyncClient] = None,
    use_same_model: bool = True,
    prompt_kind: str = "default",
    write_to_history: bool = True,
) -> str:
    if not history_message_text(message_history, "first_input") and not history_section(message_history, "messages"):
        return ""

    conversation_text = get_conversation_text(message_history)
    system_prompt, user_prompt_prefix = _get_prompts_for_kind(prompt_kind)

    provider = _normalize_provider_for_summary(
        get_provider(barebone_model.model_id, barebone_model.api_url)
    )

    model_name, api_url = _resolve_summary_target(barebone_model, provider, use_same_model)
    if not model_name or not api_url:
        _LOG.warning(f"Could not determine summarization model for provider: {provider}")
        return ""

    payload, headers = create_summary_payload(
        provider,
        model_name,
        barebone_model.api_key,
        conversation_text,
        system_prompt=system_prompt,
        user_prompt_prefix=user_prompt_prefix,
    )

    try:
        response = await async_api_request_retry(api_url, headers, payload, max_retries=3, wait_seconds=10, client=client)
        response.raise_for_status()
        data = json_dict(response.json())
        summary = parse_summary_response(provider, response)
        summary_tokens = extract_summary_tokens(provider, data)

        if write_to_history:
            _write_summary_to_history(message_history, summary, summary_tokens)

        return summary
    except httpx.HTTPError as e:
        _LOG.error(f"Failed to summarize message history: {e}")
        return ""
    except (RuntimeError, ValueError, TypeError, KeyError) as e:
        _LOG.error(f"Unexpected error during summarization: {e}")
        return ""
