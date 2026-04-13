from typing import Callable, Dict, Optional
import uuid

VALID_SIDE_EFFECT_TYPES = {"pure", "idempotent", "side_effecting", "external_transactional"}
VALID_REPLAY_POLICIES = {"allow", "same_call_only", "deny"}


def validate_required_fields(fields: Dict[str, object]) -> None:
    for field_name, value in fields.items():
        if not value:
            raise ValueError(f"{field_name} is required for the tool")
class IkaTools:
    def __init__(
        self, 
        name:str, 
        description:str, 
        parameters:dict, 
        limit_calls:int = 0, 
        required:bool = True, 
        execute_function: Optional[Callable] = None,
        parallel: bool = True,
        id: Optional[str] = None,
        side_effect_type: str = "pure",
        replay_policy: Optional[str] = None,
    ) -> None:
    
        if id is None:
            id = uuid.uuid4().hex
        if side_effect_type not in VALID_SIDE_EFFECT_TYPES:
            raise ValueError(f"Invalid side_effect_type '{side_effect_type}'")
        if replay_policy is None:
            replay_policy = "allow" if side_effect_type in {"pure", "idempotent"} else "deny"
        if replay_policy not in VALID_REPLAY_POLICIES:
            raise ValueError(f"Invalid replay_policy '{replay_policy}'")
        self.id = id
        self.name = name
        self.description = description
        self.parameters = parameters
        self.limit_calls = limit_calls
        self.required = required
        self.execute_function = execute_function
        self.parallel = parallel
        self.side_effect_type = side_effect_type
        self.replay_policy = replay_policy

        validate_required_fields(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
                "execute_function": execute_function,
            }
        )

    def execute(self, parameters:dict) -> str:
        if self.execute_function:
            return self.execute_function(parameters)
        return '{"error": "No execute function defined for this tool"}'
