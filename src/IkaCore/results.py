"""Typed final-answer contract and explicit validation outcomes."""
from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class ValidationStatus(TypedDict):
    status: Literal["not_run", "passed", "failed"]
    attempts: int
    failed_checks: list[str]


class AgentResult(TypedDict):
    """Every final-answer check receives this mapping and must return bool."""
    final_message: str
    summary: str
    usage: dict[str, int]
    cost: dict[str, float]
    model_id: str
    validation: ValidationStatus
    status: NotRequired[str]
    runtime: NotRequired[dict[str, Any]]


class FinalAnswerValidationError(ValueError):
    """Fail-closed validation exhausted its retries; result retains the rejected answer."""
    def __init__(self, result: AgentResult):
        self.result = result
        super().__init__("Final answer failed validation: " + ", ".join(result["validation"]["failed_checks"]))
