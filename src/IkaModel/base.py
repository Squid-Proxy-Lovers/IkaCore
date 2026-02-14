import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List
import sys

# add parent to path since imports assume it
sys.path.insert(0, str(Path(__file__).parent.parent / "IkaMem"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from IkaCore.cli_output import get_cli_output, OutputType


_LOG = logging.getLogger(__name__)

class AgentEndException(Exception):
    """Raised when agent_end is called to immediately stop execution."""
    def __init__(self, final_text: str):
        self.final_text = final_text
        super().__init__(final_text)

_GLOBAL_LONG_TERM_MEMORY: Optional["LTMemory"] = None 


def load_gemini_payload():
    """
    Dynamically load gemini payload function.
    This is to avoid circular import issues.
    """
    try:
        from IkaModel.google import gemini_fill_payload
        return gemini_fill_payload
    except ImportError as e:
        _LOG.error(f"Failed to import gemini_fill_payload: {e}")
        raise


DEFAULT_SYSTEM_PROMPT = """
You are a member in a complex Agentic system, every behavior you take is to help the system achieve its goals.

IMPORTANT: When you have completed your assigned task, you MUST call the agent_end tool with your final answer. 
Do not continue making tool calls after the task is complete. The agent_end tool signals that you have finished and provides your final output.
CRITICAL: MAKE SURE YOUR FINAL ANSWER IS ANSWER THE ORGINAL TASK, MAKE SURE YOUR RESPONSE IS VERY DETAILED AND COMPLETE!
"""

TOKENMAX_MAPPING = {
    "gpt-4o": 128000,                    # 128K tokens context window :contentReference[oaicite:2]{index=2}
    "gpt-4.1": 1000000,                  # 1M tokens context window :contentReference[oaicite:3]{index=3}
    "gpt-4.1-mini": 1000000,             # same 1M tokens context :contentReference[oaicite:4]{index=4}
    "gpt-4.1-nano": 1000000,             # same 1M tokens context :contentReference[oaicite:5]{index=5}
    "claude-2": 100000,                  # ~100K context (historical) :contentReference[oaicite:6]{index=6}
    "claude-2.1": 200000,                # ~200K context (expanded) :contentReference[oaicite:7]{index=7}
    "claude-3-haiku": 200000,            # typical 200K context :contentReference[oaicite:8]{index=8}
    "claude-3-sonnet": 200000,           # typical 200K context :contentReference[oaicite:9]{index=9}
    "claude-3-opus": 200000,             # typical 200K context :contentReference[oaicite:10]{index=10}
    "claude-sonnet-4": 200000,           # base context (200K) :contentReference[oaicite:11]{index=11}
    "claude-opus-4": 200000,              # base context window :contentReference[oaicite:12]{index=12}
    "claude-sonnet-4 (1M beta)": 1000000,
    "gemini-1.5-pro": 1000000,            # 1M token window on many configs :contentReference[oaicite:14]{index=14}
    "gemini-2.5-pro": 1000000,            # ~1M token window :contentReference[oaicite:15]{index=15}
    "gemini-3-pro": 1000000,              # ~1M token window reported :contentReference[oaicite:16]{index=16}
    "deepseek-v3.2": 131072,              # ~131K tokens context window :contentReference[oaicite:18]{index=18}
    "deepseek-v3.2-speciale": 131072,     # similar ~131K context :contentReference[oaicite:19]{index=19}
    "deepseek-r1": 131072,                # ~131K context (preview/hosted) :contentReference[oaicite:20]{index=20}
    # OpenRouter models (common defaults)
    "llama-3.1-70b": 131072,
    "llama-3.1-8b": 131072,
    "qwen-2.5-72b": 131072,
}



@dataclass
class ToolArgs:
    type: str
    description: str
    agent: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[dict] = None
    properties: Optional[dict] = None
    
    # we are going to assume that all args are required


@dataclass
class AgentTool:
    def __init__(self, id:str, name: str, description: str, args: ToolArgs, required: bool = True, parallel: bool = False, limit_calls: int = 0): 
        self.validate(name)
        self.id = id
        self.name = name
        self.description = description
        self.args = args   
        self.required = required
        self.parallel = parallel # assume that all tools are not parallel by default
        self.limit_calls = limit_calls


    @staticmethod
    def validate(name: str):
        if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', name):
            # we need for deepseek
            raise ValueError("Name must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.")


def init_global_long_term_memory(
    embedder_config: dict, 
    search_limit: int = 10,
    filter_func: Optional[Any] = None
) -> "LTMemory":
    """
    initialize global long-term memory (called once).
    
    Args:
        embedder_config: mem0 config
        search_limit: default number of results to return when searching
        filter_func: optional custom filter function(results: list) -> list
                     receives: list of search results sorted by relevance
                     must return: filtered list of results
                     default (no user filter): returns top 5 most relevant results
    """
    global _GLOBAL_LONG_TERM_MEMORY
    if _GLOBAL_LONG_TERM_MEMORY is None:
        from IkaMem import LTMemory
        _GLOBAL_LONG_TERM_MEMORY = LTMemory(embedder_config=embedder_config)
        _GLOBAL_LONG_TERM_MEMORY._search_limit = search_limit
        _GLOBAL_LONG_TERM_MEMORY._filter_func = filter_func or (lambda results: results[:5])
        _LOG.info("global long-term memory initialized")
    return _GLOBAL_LONG_TERM_MEMORY


def get_global_long_term_memory() -> Optional["LTMemory"]:
    return _GLOBAL_LONG_TERM_MEMORY


@dataclass
class BareBoneModel:
    def __init__(
        self, 
        model_id: str, 
        api_key: str,
        api_url: str, 
        system_prompt=DEFAULT_SYSTEM_PROMPT, 
        content_prompt: str = "", 
        max_tokens: int = 20000, 
        temperature: float = 0,
        parallel_tool_calls: bool = False,
        agent_name: Optional[str] = None,
        agent_hierarchy: Optional[List[str]] = None,
        suppress_init_output: bool = False,
        reasoning_effort: Optional[str] = None,
        ):

        # User MUST provide the following:
        if not api_key:
            raise ValueError("api_key is required for BareBoneModel")
        if not model_id:
            raise ValueError("model_id is required for BareBoneModel")
        if not api_url:
            raise ValueError("api_url is required for BareBoneModel")

        self.model_id = model_id
        self.api_key = api_key
        self.api_url = api_url
        self.system_prompt = system_prompt
        self.content_prompt = content_prompt
        self.max_tokens = max_tokens
        if temperature < 0 or temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        self.temperature = temperature
        self.agent_tools: list[AgentTool] = [] 
        self.deepthinking: bool = False
        self.parallel_tool_calls = parallel_tool_calls
        self.agent_name = agent_name
        self.agent_hierarchy = agent_hierarchy or []
        self.reasoning_effort = reasoning_effort
        
        # Display prompts using CLI output (unless suppressed)
        if not suppress_init_output:
            cli = get_cli_output()
            hierarchy = self.agent_hierarchy or [self.agent_name or "Agent"]
            agent_name = hierarchy[0] if hierarchy else "Agent"
            step = cli.get_step(agent_name) if agent_name else 0

            if self.system_prompt:
                cli.emit(
                    OutputType.AGENT_INIT,
                    f"System Prompt:\n{self.system_prompt}",
                    hierarchy,
                    step=step
                )

            if self.content_prompt:
                cli.emit(
                    OutputType.AGENT_INIT,
                    f"Content Prompt:\n{self.content_prompt}",
                    hierarchy,
                    step=step
                )


_SUMMARY_PROMPT_PATH = Path(__file__).parent / "summary_prompt"
try:
    with open(_SUMMARY_PROMPT_PATH, "r", encoding="utf-8") as f:
        SUMMARY_PROMPT = f.read()
except FileNotFoundError:
    SUMMARY_PROMPT = "Please summarize the following conversation history concisely, preserving key information and context."
