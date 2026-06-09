from importlib import import_module
from typing import Any

__all__ = [
    "AgentTool",
    "BareBoneModel",
    "ToolArgs",
    "async_api_request_retry",
    "async_chat",
    "async_execute_tool",
    "async_execute_tool_calls",
    "async_summarise_message_history",
    "chat",
    "execute_tool_calls",
    "get_global_long_term_memory",
    "get_provider",
    "init_global_long_term_memory",
    "summarise_message_history",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "BareBoneModel": ("IkaModel.base", "BareBoneModel"),
    "AgentTool": ("IkaModel.base", "AgentTool"),
    "ToolArgs": ("IkaModel.base", "ToolArgs"),
    "init_global_long_term_memory": ("IkaModel.base", "init_global_long_term_memory"),
    "get_global_long_term_memory": ("IkaModel.base", "get_global_long_term_memory"),
    "chat": ("IkaModel.chat_interface.chat_interface", "chat"),
    "summarise_message_history": ("IkaModel.chat_interface.chat_interface", "summarise_message_history"),
    "execute_tool_calls": ("IkaModel.chat_interface.chat_interface", "execute_tool_calls"),
    "get_provider": ("IkaModel.chat_interface.chat_interface", "get_provider"),
    "async_chat": ("IkaModel.chat_interface.chat_interface", "async_chat"),
    "async_summarise_message_history": ("IkaModel.chat_interface.chat_interface", "async_summarise_message_history"),
    "async_execute_tool_calls": ("IkaModel.chat_interface.chat_interface", "async_execute_tool_calls"),
    "async_execute_tool": ("IkaModel.chat_interface.chat_interface", "async_execute_tool"),
    "async_api_request_retry": ("IkaModel.chat_interface.chat_interface", "async_api_request_retry"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
