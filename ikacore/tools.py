class SquidTools:
    def __init__(self, id:str, name:str, description:str, parameters:dict, limit_calls:int = 1, required:bool = True):
        self.id = id
        self.name = name
        self.description = description
        self.parameters = parameters
        self.limit_calls = limit_calls
        self.required = required

    def execute(self, input:str) -> str:
        return self.parameters[input]