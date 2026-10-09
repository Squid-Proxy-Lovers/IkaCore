"""SDK-style retry decisions, used only by an explicit runtime option."""

# pyright: strict

import httpx

from .api_errors import IkaAPIError, IkaRateLimitError
from .retry_policy import is_retryable_status, response_body, retry_delay, retry_hint_seconds
from .runtime_policy import current_runtime_options


def response_delay(response: httpx.Response, attempt: int, max_retries: int) -> tuple[float, IkaAPIError]:
    error_type = IkaRateLimitError if response.status_code == 429 else IkaAPIError
    error = error_type(f"API request failed with status {response.status_code}", status_code=response.status_code)
    if not is_retryable_status(response.status_code) or attempt >= max_retries - 1:
        raise error
    hint = retry_hint_seconds(response.headers, response_body(response))
    return retry_delay(attempt, hint, current_runtime_options().retry_initial_seconds), error


def exception_delay(attempt: int) -> float:
    return retry_delay(attempt, None, current_runtime_options().retry_initial_seconds)
