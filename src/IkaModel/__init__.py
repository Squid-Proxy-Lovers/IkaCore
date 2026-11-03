import logging
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from IkaModel.base import AgentTool as AgentTool
    from IkaModel.base import BareBoneModel as BareBoneModel
    from IkaModel.base import ToolArgs as ToolArgs
    from IkaModel.chat_interface.chat_interface import async_api_request_retry as async_api_request_retry
    from IkaModel.chat_interface.chat_interface import async_chat as async_chat
    from IkaModel.chat_interface.chat_interface import async_execute_tool as async_execute_tool
    from IkaModel.chat_interface.chat_interface import async_execute_tool_calls as async_execute_tool_calls
    from IkaModel.chat_interface.chat_interface import (
        async_summarise_message_history as async_summarise_message_history,
    )
    from IkaModel.chat_interface.chat_interface import chat as chat
    from IkaModel.chat_interface.chat_interface import execute_tool_calls as execute_tool_calls
    from IkaModel.chat_interface.chat_interface import get_provider as get_provider
    from IkaModel.chat_interface.chat_interface import summarise_message_history as summarise_message_history
    from IkaModel.request_control import request_controls as request_controls
    from IkaModel.runtime_errors import IkaRequestCancelled as IkaRequestCancelled
    from IkaModel.runtime_errors import IkaRequestDeadlineExceeded as IkaRequestDeadlineExceeded
    from IkaModel.runtime_policy import RuntimeOptions as RuntimeOptions
    from IkaModel.runtime_policy import runtime_options as runtime_options

try:
    __version__ = version("ikacore")
except PackageNotFoundError:
    __version__ = "2.0.0"

__all__ = [
    "__version__",
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
    "get_provider",
    "summarise_message_history",
    "request_controls",
    "RuntimeOptions",
    "runtime_options",
    "IkaRequestCancelled",
    "IkaRequestDeadlineExceeded",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "BareBoneModel": ("IkaModel.base", "BareBoneModel"),
    "AgentTool": ("IkaModel.base", "AgentTool"),
    "ToolArgs": ("IkaModel.base", "ToolArgs"),
    "chat": ("IkaModel.chat_interface.chat_interface", "chat"),
    "summarise_message_history": ("IkaModel.chat_interface.chat_interface", "summarise_message_history"),
    "execute_tool_calls": ("IkaModel.chat_interface.chat_interface", "execute_tool_calls"),
    "get_provider": ("IkaModel.chat_interface.chat_interface", "get_provider"),
    "async_chat": ("IkaModel.chat_interface.chat_interface", "async_chat"),
    "async_summarise_message_history": ("IkaModel.chat_interface.chat_interface", "async_summarise_message_history"),
    "async_execute_tool_calls": ("IkaModel.chat_interface.chat_interface", "async_execute_tool_calls"),
    "async_execute_tool": ("IkaModel.chat_interface.chat_interface", "async_execute_tool"),
    "async_api_request_retry": ("IkaModel.chat_interface.chat_interface", "async_api_request_retry"),
    "request_controls": ("IkaModel.request_control", "request_controls"),
    "RuntimeOptions": ("IkaModel.runtime_policy", "RuntimeOptions"),
    "runtime_options": ("IkaModel.runtime_policy", "runtime_options"),
    "IkaRequestCancelled": ("IkaModel.runtime_errors", "IkaRequestCancelled"),
    "IkaRequestDeadlineExceeded": ("IkaModel.runtime_errors", "IkaRequestDeadlineExceeded"),
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


logging.getLogger("IkaModel").addHandler(logging.NullHandler())
