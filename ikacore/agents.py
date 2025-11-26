from typing import Callable, Optional
from tools import SquidTools
from stages import SquidStage
from squidrag import SquidRAGSource
from memory import SquidMemorySystem
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator

# primary features: 
# - managed agents 
# 	- nodes with hierarchy 
# 		- define entry and exit points
# 	- verification dual model nodes 
# - stages 
# 	- limit tool calls 
# 	- limit agents
# 	- define transition states 
# 		- funciton call 
# - check point verification 
# 	- define safe stages
# 	- debug log state mode 
# 		- track executions 
# 			- allows for retesting execution from a "interesting" point
# - enable batching mode for specific tool calls 
# - logging mode 
# 	- log to a file with 
# - self reflection / planning mode ( ? like smolagents)
# - 
class ParaBaseAgent:
    def __init__(
        self, 
        name: str, 
        description: str,
        prompt: str, 
        role: str, 
        tools: list[SquidTools], 
        checkpoint: bool = False, 
        Batch: bool = False, 
        BatchMax: int = 3, 
        Stages: list[SquidStage] = [],
        subagents: Optional[list["ParaBaseAgent"]] = None, # do we need to create a manger agent class
        next_agent: Optional["ParaBaseAgent"] = None, # agent to call after this agent completes 
        bidirectional: bool = False, # allow for feedback loops
        feedback_agent: Optional["ParaBaseAgent"] = None, # agent to receive feedback from
        maxsteps: int = 10, # maximum number of steps to take
        step_timeout: int = 900, # timeout for each step, default 900 seconds / 15 minutes
        RAGSource: Optional[list[type[SquidRAGSource]]] = None,  # list of RAG source types to use
        memory: bool = False,
        memory_finder: Optional["SquidMemorySystem"] = None,
        final_answer_check: list[Callable] | None = None,

        ):
        self.name = name
        self.description = description
        self.prompt = prompt
        self.role = role
        self.tools = tools
        self.checkpoint = checkpoint
        self.Batch = Batch
        self.BatchMax = BatchMax
        self.Stages = Stages
        # agent node types 
        self.subagents = subagents
        self.next_agent = next_agent
        self.bidirectional = bidirectional
        self.feedback_agent = feedback_agent
        
        self.maxsteps = maxsteps
        self.step_timeout = step_timeout
        self.RAGSource = RAGSource
        self.memory = memory
        self.memory_finder = memory_finder


        self.final_answer_check = final_answer_check
        
        # 1 type or use stages
        if len(self.Stages) > 0:
            if ((self.subagents is not None and len(self.subagents) > 0) or (self.next_agent is not None) or (self.bidirectional is True)):
                raise ValueError("Subagents, next agent, or bidirectional agent are not allowed when stages are defined, stages should be used to define the flow of the agent")
        else:
            if ((self.subagents is not None and len(self.subagents) > 0) ^ (self.next_agent is not None) ^ (self.bidirectional is True)):
                raise ValueError("Subagents, next agent, or bidirectional agent must be defined, only one of the three can be defined without stages")