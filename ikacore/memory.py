"""
the goal of this class is to define 
what information is needed to be stored in memory
instead or a hard coded system created by devs 
we allow for users to define their own memory system
"""

class SquidMemorySystem:
    def __init__(self, name:str, description:str, type:str):
        self.name = name
        self.description = description
        self.type = type



