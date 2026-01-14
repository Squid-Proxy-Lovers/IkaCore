from typing import Callable

class IkaTools:
    def __init__(self, id:str, name:str, description:str, parameters:dict, limit_calls:int = 1, required:bool = True, execute_function:Callable = None, parallel:bool = True):
        self.id = id
        self.name = name
        self.description = description
        self.parameters = parameters
        self.limit_calls = limit_calls
        self.required = required
        self.execute_function = execute_function
        self.parallel = parallel  # By default, tools can run in parallel

    def execute(self, parameters:dict) -> str:
        if self.execute_function:
            return self.execute_function(parameters)
        return '{"error": "No execute function defined for this tool"}'