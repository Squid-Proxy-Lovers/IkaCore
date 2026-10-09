# pyright: strict

from .api_errors import IkaAPIError


class IkaRuntimeError(RuntimeError):
    """Base error for IkaCore runtime orchestration failures."""


class IkaRequestControlError(IkaRuntimeError, IkaAPIError):
    """An explicit cancellation or deadline stopped the current execution."""


class IkaRequestCancelled(IkaRequestControlError):
    """The caller cancelled the current execution."""


class IkaRequestDeadlineExceeded(IkaRequestControlError):
    """The caller's execution deadline expired."""


class IkaProviderPayloadError(IkaRuntimeError):
    """Provider request payload construction failed."""


class IkaProviderResponseError(IkaRuntimeError):
    """Provider response parsing failed."""


class IkaContextWindowError(IkaRuntimeError):
    """Context-window recovery failed."""


class IkaToolExecutionError(IkaRuntimeError):
    """Tool execution failed before a structured tool result could be returned."""


class IkaCheckpointError(IkaRuntimeError):
    """Checkpoint serialization, loading, or persistence failed."""


class IkaDebugDumpError(IkaRuntimeError):
    """Best-effort request/response debug dumping failed."""
