# pyright: strict
from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Any, Optional, cast, get_type_hints

JsonDict = dict[str, Any]
MessageHistory = dict[str, Any]
FinalAnswerCheck = Callable[[JsonDict], bool]
RawFinalAnswerCheck = Callable[[JsonDict], object]


def _return_type_allows_bool(return_type: object) -> bool:
    get_origin = getattr(typing, "get_origin", None)
    get_args = getattr(typing, "get_args", None)
    origin = get_origin(return_type) if get_origin else None
    args = get_args(return_type) if origin and get_args else ()
    return return_type is bool or (origin is not None and bool in args)


def validate_final_answer_checks_for_history(
    message_history: MessageHistory, final_answer_check: Optional[list[RawFinalAnswerCheck]]
) -> list[FinalAnswerCheck]:
    if final_answer_check is None:
        return []
    for check in final_answer_check:
        if not callable(check):
            raise ValueError("final_answer_check must be a list of callable functions")
        func_name = getattr(check, "__name__", "unknown")
        docstring = (check.__doc__ or "").strip()
        if not docstring:
            raise ValueError(f"final_answer_check function '{func_name}' must have a docstring explaining what it does")
        try:
            hints = get_type_hints(check)
            return_type = hints.get("return")
        except (TypeError, AttributeError):
            return_type = None
        if return_type is not None:
            if not _return_type_allows_bool(return_type):
                raise ValueError(
                    f"final_answer_check function '{func_name}' must have return type annotation of bool, got {return_type}"
                )
        result: object = check(message_history)
        if not isinstance(result, bool):
            raise ValueError(
                f"final_answer_check function '{func_name}' must return a boolean, got {type(result).__name__}"
            )
    return cast(list[FinalAnswerCheck], final_answer_check)


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
