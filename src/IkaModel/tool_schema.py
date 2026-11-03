# pyright: strict

from __future__ import annotations

from typing import Any, NamedTuple, cast

JsonSchema = dict[str, Any]


class ProviderToolPayload(NamedTuple):
    tools: list[JsonSchema]
    names: frozenset[str]
    required_names: tuple[str, ...]


_PROVIDER_TOOL_CACHE: dict[tuple[Any, ...], ProviderToolPayload] = {}
_PROVIDER_TOOL_FAST_CACHE: dict[tuple[Any, ...], tuple[list[Any], ProviderToolPayload]] = {}
_PROVIDER_TOOL_CACHE_MAX = 512


def _as_schema(value: Any) -> JsonSchema:
    if isinstance(value, dict):
        return cast(JsonSchema, value)
    return {}


def build_tool_parameters(tool: Any) -> JsonSchema:
    """Build the object-root JSON schema used by provider function tools."""
    tool_args = getattr(tool, "args", None)
    arg_type = getattr(tool_args, "type", "string")
    description = getattr(tool_args, "description", None)
    properties = getattr(tool_args, "properties", None)
    tool_name = getattr(tool, "name", "")
    cache_key = (tool_name, arg_type, description, id(properties))
    cached = getattr(tool, "_ika_tool_parameters_cache", None)
    if isinstance(cached, tuple):
        cached_tuple = cast(tuple[Any, ...], cached)
        if len(cached_tuple) == 2 and cached_tuple[0] == cache_key:
            return _as_schema(cached_tuple[1])

    if tool_name == "agent_end" and arg_type == "input":
        parameters = _input_parameters(
            description or "Final response content. This is REQUIRED - provide your complete final answer here."
        )
        return _cache_tool_parameters(tool, cache_key, parameters)

    if arg_type == "input":
        parameters = _input_parameters(description or f"Input for {tool_name}")
        return _cache_tool_parameters(tool, cache_key, parameters)

    if isinstance(properties, dict):
        schema_properties = cast(JsonSchema, properties)
        if not schema_properties:
            parameters = _empty_object_parameters()
            return _cache_tool_parameters(tool, cache_key, parameters)
        if schema_properties.get("type") == "object":
            parameters = {
                "type": "object",
                "properties": _as_schema(schema_properties.get("properties")),
                "required": _as_list(schema_properties.get("required")),
            }
            return _cache_tool_parameters(tool, cache_key, parameters)
        if "type" not in schema_properties:
            required_list = [str(item) for item in _as_list(schema_properties.get("__required__"))]
            props: JsonSchema = {
                key: cast(JsonSchema, value)
                for key, value in schema_properties.items()
                if key != "__required__" and isinstance(value, dict)
            }
            parameters: JsonSchema = {"type": "object", "properties": props, "required": required_list}
            return _cache_tool_parameters(tool, cache_key, parameters)
        parameters = _ensure_object_root(schema_properties)
        return _cache_tool_parameters(tool, cache_key, parameters)

    if arg_type == "object":
        parameters = _empty_object_parameters()
        return _cache_tool_parameters(tool, cache_key, parameters)

    json_type = "integer" if arg_type == "stage_index" else "string"
    required = ["input"] if arg_type == "input" else []
    parameters = {
        "type": "object",
        "properties": {
            arg_type: {
                "type": json_type,
                "description": description or f"Parameter for {tool_name}",
            }
        },
        "required": required,
    }
    return _cache_tool_parameters(tool, cache_key, parameters)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return cast(list[Any], value)
    return []


def _cache_tool_parameters(tool: Any, cache_key: tuple[Any, ...], parameters: JsonSchema) -> JsonSchema:
    try:
        tool._ika_tool_parameters_cache = (cache_key, parameters)
    except (AttributeError, TypeError):
        pass
    return parameters


def build_provider_tool_payload(provider: str, tools: list[Any]) -> ProviderToolPayload:
    """Build and cache provider-specific tool declarations for a stable tool list."""
    if not tools:
        return ProviderToolPayload([], frozenset(), ())

    fast_key = (provider, id(tools), len(tools))
    fast_cached = _PROVIDER_TOOL_FAST_CACHE.get(fast_key)
    if fast_cached is not None and fast_cached[0] is tools:
        return fast_cached[1]

    cache_key = (provider, tuple(_tool_fingerprint(tool) for tool in tools))
    cached = _PROVIDER_TOOL_CACHE.get(cache_key)
    if cached is not None:
        _PROVIDER_TOOL_FAST_CACHE[fast_key] = (tools, cached)
        return cached

    names = frozenset(getattr(tool, "name", "") for tool in tools)
    required_names = tuple(getattr(tool, "name", "") for tool in tools if getattr(tool, "required", False))
    rendered = [_build_provider_tool(provider, tool) for tool in tools]
    payload = ProviderToolPayload(rendered, names, required_names)
    if len(_PROVIDER_TOOL_CACHE) >= _PROVIDER_TOOL_CACHE_MAX:
        _PROVIDER_TOOL_CACHE.clear()
        _PROVIDER_TOOL_FAST_CACHE.clear()
    _PROVIDER_TOOL_CACHE[cache_key] = payload
    _PROVIDER_TOOL_FAST_CACHE[fast_key] = (tools, payload)
    return payload

def _tool_fingerprint(tool: Any) -> tuple[Any, ...]:
    tool_args = getattr(tool, "args", None)
    return (
        id(tool),
        getattr(tool, "name", ""),
        getattr(tool, "description", ""),
        bool(getattr(tool, "required", False)),
        bool(getattr(tool, "parallel", False)),
        getattr(tool_args, "type", "string"),
        getattr(tool_args, "description", None),
        id(getattr(tool_args, "properties", None)),
    )


def _build_provider_tool(provider: str, tool: Any) -> JsonSchema:
    parameters = build_tool_parameters(tool)
    if provider in {"openai", "deepseek", "openrouter"}:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }
    if provider == "anthropic":
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": parameters,
        }
    if provider == "gemini":
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        }

    if provider == "codex" and parameters.get("type") == "object":
        parameters = {**parameters, "additionalProperties": parameters.get("additionalProperties", False)}
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters,
    }


def _input_parameters(description: str) -> JsonSchema:
    return {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": description,
            }
        },
        "required": ["input"],
    }


def _empty_object_parameters() -> JsonSchema:
    return {"type": "object", "properties": {}, "required": []}


def _ensure_object_root(parameters: JsonSchema) -> JsonSchema:
    if parameters.get("type") == "object":
        return parameters
    return {
        "type": "object",
        "properties": _as_schema(parameters.get("properties")),
        "required": _as_list(parameters.get("required")),
    }
