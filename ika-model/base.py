import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union

from .tools import Tool

_LOG = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = "You are a memeber in a complex Agentic system, every behavior you take is to help the system achieve its goals."



# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_weather",
#             "description": "Get weather of a location, the user should supply a location first.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     }
#                 },
#                 "required": ["location"
#             },
#         }
#     },
# ]

@dataclass
class ToolArgs:
    type: str
    description: str
    # we are going to assume that all args are required




@dataclass
class AgentTool:
    def __init__(self, name: str, description: str, args: ToolArgs, required: bool = True): 
        self.validate(name)
        self.name = name
        self.description = description
        self.args = args
        self.required = required
    
    def validate(name: str):
        if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', name):
            # we need for deepseek
            raise ValueError("Name must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.")




@dataclass
class Model(ABC)
    def __init__(self, model_id: str, api_key: str, api_url: str, system_prompt=DEFAULT_SYSTEM_PROMPT, content_prompt: str, max_tokens: int = 20000, temperature: float = 0):
        self.model_id = model_id
        self.api_key = api_key
        self.api_url = find_api_url(model_id)  
        self.system_prompt = system_prompt
        self.content_prompt = content_prompt
        self.max_tokens = max_tokens
        if temperature < 0 or temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        self.temperature = temperature # define this as a percentage between 0 and 1, for any provider that uses a different scale, we will need to convert it to the correct scale
        self.agent_tools: list[AgentTool] = [] 
        self.deepthinking: bool = False



    def find_api_url(self, model_id: str) -> str:
        return {
            "deepseek": "https://api.deepseek.com/chat/completions",
            "openai": "https://api.openai.com/v1/chat/completions",
            "anthropic": "https://api.anthropic.com/v1/chat/completions",
            "gemini": "https://api.gemini.com/v1/chat/completions",
        }.get(model_id)