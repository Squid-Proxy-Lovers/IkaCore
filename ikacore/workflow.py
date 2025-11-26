from agents import ParaBaseAgent
class ParaWorkflow:
    def __init__(self, name:str, description:str, agents:list[ParaBaseAgent], savestate:bool = False, logging:bool = False, ):
        self.name = name
        self.description = description
        self.agents = agents
        self.savestate = savestate
        self.logging = logging