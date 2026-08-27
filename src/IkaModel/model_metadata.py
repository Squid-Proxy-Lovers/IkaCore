# pyright: strict

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, Optional, cast

from .codex_constants import CODEX_API_URL

DEFAULT_MAX_TOKENS = 128000

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
_MODEL_METADATA_RESOURCE = "data/model_metadata.json"


def _load_model_metadata() -> dict[str, Any]:
    try:
        from src.resources import read_text as read_ika_resource

        raw = read_ika_resource(f"IkaCore/src/IkaModel/{_MODEL_METADATA_RESOURCE}")
    except (ImportError, ModuleNotFoundError, OSError):
        try:
            raw = files(__package__ or "IkaModel").joinpath(_MODEL_METADATA_RESOURCE).read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise RuntimeError(f"missing model metadata resource: {_MODEL_METADATA_RESOURCE}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid model metadata JSON: {_MODEL_METADATA_RESOURCE}") from e
    if not isinstance(data, dict):
        raise RuntimeError("model metadata must be a JSON object")
    return cast(dict[str, Any], data)


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"model metadata field '{key}' must be a non-empty object")
    return cast(dict[str, Any], value)


def _require_list(value: Any, key: str, expected_len: int) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"metadata for '{key}' must be a {expected_len}-item list")
    typed_value = cast(list[Any], value)
    if len(typed_value) != expected_len:
        raise RuntimeError(f"metadata for '{key}' must be a {expected_len}-item list")
    return typed_value


def _load_string_set(data: dict[str, Any], key: str) -> frozenset[str]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise RuntimeError(f"model metadata field '{key}' must be a list")
    return frozenset(str(item) for item in cast(list[Any], value))


def _load_tokenmax_mapping(data: dict[str, Any]) -> dict[str, int]:
    raw = _require_mapping(data, "tokenmax_mapping")
    out: dict[str, int] = {}
    for key, value in raw.items():
        if not key:
            raise RuntimeError("token metadata keys must be non-empty strings")
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"token metadata for '{key}' must be a positive integer")
        out[key] = value
    return out


def _load_model_costs(data: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    raw = _require_mapping(data, "model_costs")
    out: dict[str, tuple[float, float, float]] = {}
    for key, value in raw.items():
        if not key:
            raise RuntimeError("cost metadata keys must be non-empty strings")
        raw_costs = _require_list(value, key, 3)
        costs = (float(raw_costs[0]), float(raw_costs[1]), float(raw_costs[2]))
        if any(v < 0 for v in costs):
            raise RuntimeError(f"cost metadata for '{key}' cannot contain negative values")
        out[key] = costs
    return out


def _load_api_urls(data: dict[str, Any]) -> dict[str, str]:
    raw = _require_mapping(data, "api_url_by_provider")
    out = {str(key): str(value) for key, value in raw.items()}
    if out.get("codex") != CODEX_API_URL:
        raise RuntimeError("codex API URL in model metadata is out of sync with CODEX_API_URL")
    return out


def _load_summary_models(data: dict[str, Any]) -> dict[str, tuple[str, str]]:
    raw = _require_mapping(data, "summary_model_by_provider")
    out: dict[str, tuple[str, str]] = {}
    for key, value in raw.items():
        raw_summary = _require_list(value, key, 2)
        model, url = raw_summary
        out[str(key)] = (str(model), str(url))
    return out


_MODEL_DATA = _load_model_metadata()
API_URL_BY_PROVIDER = _load_api_urls(_MODEL_DATA)
SUMMARY_MODEL_BY_PROVIDER = _load_summary_models(_MODEL_DATA)
CODEX_KNOWN_MODELS = _load_string_set(_MODEL_DATA, "codex_known_models")
TOKENMAX_MAPPING = _load_tokenmax_mapping(_MODEL_DATA)
MODEL_COSTS = _load_model_costs(_MODEL_DATA)

TOKEN_FALLBACKS = {
    "anthropic": TOKENMAX_MAPPING["claude-sonnet-4"],
    "deepseek": TOKENMAX_MAPPING["deepseek-chat"],
    "gemini": TOKENMAX_MAPPING["gemini-1.5-pro"],
    "llama": 131072,
    "openai": TOKENMAX_MAPPING["gpt-4o"],
    "qwen": 131072,
}

TOKEN_FALLBACK_RULES = (
    (("gpt-4", "gpt-4o"), "openai"),
    (("claude",), "anthropic"),
    (("gemini",), "gemini"),
    (("deepseek",), "deepseek"),
    (("llama",), "llama"),
    (("qwen",), "qwen"),
)

OPENAI_MODEL_MARKERS = ("gpt", "o1", "o3", "o4")
MODEL_PROVIDER_RULES = (
    (("deepseek",), "deepseek"),
    (OPENAI_MODEL_MARKERS, "openai"),
    (("claude",), "anthropic"),
    (("gemini",), "gemini"),
)

URL_PROVIDER_CONTAINS = (
    ("chatgpt.com/backend-api/codex", "codex"),
    ("openrouter.ai", "openrouter"),
    ("deepseek.com", "deepseek"),
    ("generativelanguage.googleapis.com", "gemini"),
    ("anthropic.com", "anthropic"),
)

URL_PROVIDER_SUFFIXES = (
    ("/v1/responses", "openai_responses"),
    ("/v1/chat/completions", "openai"),
)

MODEL_COST_ALIASES = {
    "deepseek": "deepseek-chat",
    "deepseek-v4-reasoner": "deepseek-reasoner",
    "deepseek-v4-thinking": "deepseek-reasoner",
}

OPENAI_MAX_COMPLETION_TOKEN_MARKERS = ("gpt-4.1", "gpt-5", "o1", "o3", "o4")
OPENAI_DEFAULT_TEMPERATURE_ONLY_MARKERS = ("gpt-5", "o1", "o3", "o4")
OPENAI_CHAT_COMPLETION_OUTPUT_CAPS = (
    (("gpt-4o-mini",), 16384),
    (("gpt-4o",), 16384),
    (("gpt-3.5",), 4096),
    (("gpt-4", "turbo"), 4096),
)
OPENROUTER_FORCED_TOOL_CHOICE_MARKERS = (
    "gpt-",
    "openai/",
    "z-ai/glm-5.1",
    "minimax/",
    "kimi",
)
OPENROUTER_REASONING_EXCLUDE_MARKERS = ("qwen3.5", "qwen3.6", "mimo", "glm-5-turbo")

TOKEN_KEYS_BY_LENGTH = tuple(sorted(TOKENMAX_MAPPING, key=len, reverse=True))
MODEL_COST_KEYS_BY_LENGTH = tuple(sorted(MODEL_COSTS, key=len, reverse=True))


def get_openai_url(use_responses_api: bool = True) -> str:
    return OPENAI_RESPONSES_URL if use_responses_api else OPENAI_CHAT_COMPLETIONS_URL


@lru_cache(maxsize=4096)
def normalize_model_id(model_id: str) -> str:
    return (model_id or "").lower().strip()


def strip_provider_prefix(model_id: str) -> str:
    return model_id.split("/", 1)[1] if "/" in model_id else model_id


def _matches_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def get_summary_model_for_provider(provider: str) -> tuple[Optional[str], Optional[str]]:
    return SUMMARY_MODEL_BY_PROVIDER.get(provider.lower(), (None, None))


@lru_cache(maxsize=4096)
def uses_openai_max_completion_tokens(model_id: str) -> bool:
    return _matches_any(normalize_model_id(model_id), OPENAI_MAX_COMPLETION_TOKEN_MARKERS)


@lru_cache(maxsize=4096)
def supports_custom_temperature(model_id: str) -> bool:
    return not _matches_any(normalize_model_id(model_id), OPENAI_DEFAULT_TEMPERATURE_ONLY_MARKERS)


@lru_cache(maxsize=4096)
def cap_openai_chat_completion_tokens(model_id: str, max_tokens: int) -> int:
    model_id_lower = normalize_model_id(model_id)
    for markers, cap in OPENAI_CHAT_COMPLETION_OUTPUT_CAPS:
        if all(marker in model_id_lower for marker in markers):
            return min(max_tokens, cap)
    return min(max_tokens, 8192)


@lru_cache(maxsize=4096)
def is_anthropic_haiku_model(model_id: str) -> bool:
    return "haiku" in normalize_model_id(model_id)


@lru_cache(maxsize=4096)
def supports_anthropic_parallel_tool_use(model_id: str) -> bool:
    model_id_lower = normalize_model_id(model_id)
    return any(marker in model_id_lower for marker in ("opus-4", "sonnet-4", "claude-4"))


@lru_cache(maxsize=4096)
def openrouter_supports_forced_tool_choice(model_id: str) -> bool:
    return _matches_any(normalize_model_id(model_id), OPENROUTER_FORCED_TOOL_CHOICE_MARKERS)


@lru_cache(maxsize=4096)
def openrouter_should_exclude_reasoning_for_tools(model_id: str) -> bool:
    return _matches_any(normalize_model_id(model_id), OPENROUTER_REASONING_EXCLUDE_MARKERS)


@lru_cache(maxsize=4096)
def get_max_tokens_for_model(model_id: str) -> int:
    model_id_lower = normalize_model_id(model_id)

    for key in TOKEN_KEYS_BY_LENGTH:
        if key in model_id_lower:
            return TOKENMAX_MAPPING[key]

    for markers, fallback_key in TOKEN_FALLBACK_RULES:
        if _matches_any(model_id_lower, markers):
            return TOKEN_FALLBACKS[fallback_key]

    return DEFAULT_MAX_TOKENS


@lru_cache(maxsize=4096)
def get_provider_for_model(
    model_id: str,
    api_url: Optional[str] = None,
    use_responses_api: bool = True,
) -> str:
    if api_url:
        api_url_lower = api_url.lower()
        stripped_url = api_url_lower.rstrip("/")

        for needle, provider in URL_PROVIDER_CONTAINS:
            if needle in api_url_lower:
                return provider

        for suffix, provider in URL_PROVIDER_SUFFIXES:
            if stripped_url.endswith(suffix):
                return provider

        if "openai.com" in api_url_lower:
            return "openai_responses" if use_responses_api else "openai"

    model_id_lower = normalize_model_id(model_id)

    if "/" in model_id_lower:
        return "openrouter"

    if model_id_lower.endswith("-codex"):
        return "codex"

    for markers, provider in MODEL_PROVIDER_RULES:
        if _matches_any(model_id_lower, markers):
            return "openai_responses" if provider == "openai" and use_responses_api else provider

    return "openai_responses" if use_responses_api else "openai"


@lru_cache(maxsize=4096)
def get_api_url_for_model(model_id: str, use_responses_api: bool = True) -> str:
    model_id_lower = normalize_model_id(model_id)

    if "/" in model_id_lower:
        return API_URL_BY_PROVIDER["openrouter"]

    if model_id_lower.endswith("-codex"):
        return API_URL_BY_PROVIDER["codex"]

    for markers, provider in MODEL_PROVIDER_RULES:
        if not _matches_any(model_id_lower, markers):
            continue
        if provider == "openai":
            return get_openai_url(use_responses_api)
        if provider == "gemini":
            return f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
        return API_URL_BY_PROVIDER[provider]

    return get_openai_url(use_responses_api)


@lru_cache(maxsize=4096)
def _model_cost_candidates(model_id: str) -> tuple[str, ...]:
    candidates = [model_id]

    if "anthropic." in model_id:
        candidates.append(model_id.split("anthropic.", 1)[1])
    if "." in model_id:
        candidates.append(model_id.split(".")[-1])
    for delimiter in ("@", ":"):
        if delimiter in model_id:
            candidates.append(model_id.split(delimiter, 1)[0])

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = MODEL_COST_ALIASES.get(candidate.strip(), candidate.strip())
        if candidate and candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)
    return tuple(normalized)


@lru_cache(maxsize=4096)
def resolve_model_cost(model_id: str) -> Optional[tuple[float, float, float]]:
    normalized_id = normalize_model_id(model_id)
    if not normalized_id:
        return None

    normalized_id = MODEL_COST_ALIASES.get(strip_provider_prefix(normalized_id), strip_provider_prefix(normalized_id))

    result = MODEL_COSTS.get(normalized_id)
    if result:
        return result

    candidates = _model_cost_candidates(normalized_id)
    for candidate in candidates:
        result = MODEL_COSTS.get(candidate)
        if result:
            return result

    for candidate in candidates:
        for key in MODEL_COST_KEYS_BY_LENGTH:
            if candidate.startswith(key):
                return MODEL_COSTS[key]

    return None
