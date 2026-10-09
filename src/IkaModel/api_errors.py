"""Provider error classes with their existing public import identities."""

# pyright: strict

from typing import Optional


class IkaAPIError(RuntimeError):
    __module__ = "IkaModel.request_interface"

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class IkaRateLimitError(IkaAPIError):
    __module__ = "IkaModel.request_interface"


class IkaTimeoutError(IkaAPIError):
    __module__ = "IkaModel.request_interface"


class IkaHTTPError(IkaAPIError):
    __module__ = "IkaModel.request_interface"
