# pyright: strict

from __future__ import annotations

import math
import uuid
from typing import Any, Callable, Optional

ToolParameters = dict[str, Any]
ToolExecutor = Callable[[ToolParameters], Any]


def validate_required_fields(fields: dict[str, object]) -> None:
    for field_name, value in fields.items():
        if not value:
            raise ValueError(f"{field_name} is required for the tool")


class IkaTools:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: ToolParameters,
        limit_calls: int = 0,
        required: bool = False,
        execute_function: Optional[ToolExecutor] = None,
        parallel: bool = True,
        id: Optional[str] = None,
        *, timeout: Optional[float] = None, replay_policy: Optional[str] = None,
        side_effect_type: Optional[str] = None,
    ) -> None:
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("tool timeout must be finite and non-negative")
        if replay_policy is not None and replay_policy not in {"allow", "deny"}:
            raise ValueError("replay_policy must be allow or deny")
        if id is None:
            id = uuid.uuid4().hex
        self.id: str = id
        self.name: str = name
        self.description: str = description
        self.parameters: ToolParameters = parameters
        self.limit_calls: int = limit_calls
        self.required: bool = required
        self.execute_function: Optional[ToolExecutor] = execute_function
        self.parallel: bool = parallel
        if timeout is not None:
            self.timeout = timeout
        if replay_policy is not None:
            self.replay_policy = replay_policy
        if side_effect_type is not None:
            self.side_effect_type = side_effect_type

        validate_required_fields(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
                "execute_function": execute_function,
            }
        )

    def execute(self, parameters: ToolParameters) -> Any:
        if self.execute_function:
            return self.execute_function(parameters)
        return '{"error": "No execute function defined for this tool"}'
