from agents import ParaBaseAgent

# not sure we actually need this for now since we can just allow execution of agents directly
class ParaWorkflow:
    def __init__(self, name:str, description:str, agents:list[ParaBaseAgent], savestate:bool = False, logging:bool = False, ):
        self.name = name
        self.description = description
        self.agents = agents
        self.savestate = savestate
        self.logging = logging

    
    def run(self):
        for agent in self.agents:
            agent.run()