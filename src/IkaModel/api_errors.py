"""Provider request errors."""

# pyright: strict

from typing import Optional


class IkaAPIError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class IkaRateLimitError(IkaAPIError):
    pass


class IkaTimeoutError(IkaAPIError):
    pass


class IkaHTTPError(IkaAPIError):
    pass
