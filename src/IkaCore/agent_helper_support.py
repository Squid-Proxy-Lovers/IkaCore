# pyright: strict
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Optional, cast

from .results import AgentResult

JsonDict = dict[str, Any]
MessageHistory = dict[str, Any]
FinalAnswerCheck = Callable[[AgentResult], bool]
RawFinalAnswerCheck = Callable[[AgentResult], object]


def validate_final_answer_checks_for_history(
    message_history: MessageHistory, final_answer_check: Optional[list[RawFinalAnswerCheck]]
) -> list[FinalAnswerCheck]:
    """Check the calling convention without executing user code or annotations."""
    for check in final_answer_check or []:
        if not callable(check) or inspect.iscoroutinefunction(check):
            raise ValueError("final_answer_checks must contain synchronous callable functions")
        try:
            inspect.signature(check).bind(object())
        except (TypeError, ValueError) as error:
            raise ValueError("final_answer_checks must accept one positional AgentResult") from error
    return cast(list[FinalAnswerCheck], list(final_answer_check or []))


def validate_required_fields(fields: dict[str, object]) -> None:
    for field_name, value in fields.items():
        if not value:
            raise ValueError(f"{field_name} is required for the agent")


def initial_message_history(system_prompt: Optional[str]) -> MessageHistory:
    return {
        "system": {"message": system_prompt or "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }
