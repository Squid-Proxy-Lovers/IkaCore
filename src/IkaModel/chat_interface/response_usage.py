"""Provider usage extraction helpers."""

# pyright: strict

from __future__ import annotations

from typing import Any, Callable, TypeAlias, cast

UsageDict: TypeAlias = dict[str, Any]
UsageExtractorFn = Callable[[UsageDict, UsageDict], UsageDict]


def _as_usage_dict(value: Any) -> UsageDict:
    if isinstance(value, dict):
        return cast(UsageDict, value)
    return {}


def _empty_usage() -> UsageDict:
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "input_cached_tokens": 0}
    return usage


def _extract_chat_usage(raw_usage: UsageDict, data: UsageDict) -> UsageDict:
    usage = _empty_usage()
    usage["input_tokens"] = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
    usage["output_tokens"] = raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0))
    usage["total_tokens"] = raw_usage.get("total_tokens", usage["input_tokens"] + usage["output_tokens"])
    ptd = _as_usage_dict(raw_usage.get("prompt_tokens_details"))
    cached = ptd.get("cached_tokens", 0) or raw_usage.get("prompt_cache_hit_tokens", 0) or 0
    usage["input_cached_tokens"] = cached
    return usage


def _extract_responses_usage(raw_usage: UsageDict, data: UsageDict) -> UsageDict:
    usage = _empty_usage()
    usage["input_tokens"] = raw_usage.get("input_tokens", 0)
    usage["output_tokens"] = raw_usage.get("output_tokens", 0)
    usage["total_tokens"] = raw_usage.get("total_tokens", usage["input_tokens"] + usage["output_tokens"])
    itd = _as_usage_dict(raw_usage.get("input_tokens_details"))
    usage["input_cached_tokens"] = itd.get("cached_tokens", 0) or 0
    return usage


def _extract_anthropic_usage(raw_usage: UsageDict, data: UsageDict) -> UsageDict:
    usage = _empty_usage()
    usage["input_tokens"] = raw_usage.get("input_tokens", 0)
    usage["output_tokens"] = raw_usage.get("output_tokens", 0)
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _extract_gemini_usage(raw_usage: UsageDict, data: UsageDict) -> UsageDict:
    usage = _empty_usage()
    meta = _as_usage_dict(data.get("usageMetadata")) or raw_usage
    usage["input_tokens"] = meta.get("promptTokenCount", 0)
    usage["output_tokens"] = meta.get("candidatesTokenCount", meta.get("totalTokenCount", 0))
    usage["total_tokens"] = meta.get("totalTokenCount", usage["input_tokens"] + usage["output_tokens"])
    usage["input_cached_tokens"] = meta.get("cachedContentTokenCount", 0)
    return usage


USAGE_EXTRACTORS: dict[str, UsageExtractorFn] = {
    "anthropic": _extract_anthropic_usage,
    "codex": _extract_responses_usage,
    "deepseek": _extract_chat_usage,
    "gemini": _extract_gemini_usage,
    "openai": _extract_chat_usage,
    "openai_responses": _extract_responses_usage,
    "openrouter": _extract_chat_usage,
}

def extract_usage(provider: str, data: UsageDict) -> UsageDict:
    raw_usage = _as_usage_dict(data.get("usage"))
    extractor = USAGE_EXTRACTORS.get(provider)
    if extractor is None:
        usage = _empty_usage()
        usage["total_tokens"] = raw_usage.get("total_tokens", 0)
        return usage
    return extractor(raw_usage, data)
