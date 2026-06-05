import uuid
from typing import Callable, Dict, Optional


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
        required:bool = False, 
        execute_function: Optional[Callable] = None,
        parallel: bool = True,
        id: Optional[str] = None
    ) -> None:
    
        if id is None:
            id = uuid.uuid4().hex
        self.id = id
        self.name = name
        self.description = description
        self.parameters = parameters
        self.limit_calls = limit_calls
        self.required = required
        self.execute_function = execute_function
        self.parallel = parallel

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
