"""Data structures for IkaGeneral execution environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ContextPack:
    """Named, mutable container for context data shared across the execution."""

    name: str
    content: str  # free-form text, code, data
    book_name: str = "context_packs"
    tags: list[str] = field(default_factory=list)
    source: str = "agent"  # where this pack came from
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    size_tokens: int = 0  # estimated token count


@dataclass
class CcpContextConfig:
    """Configuration for CCP-backed context pack persistence."""

    session: str
    shelf_name: str = "ika_general_context"
    book_name: str = "context_packs"
    shelf_description: str = "IkaGeneral context pack storage"
    book_description: str = "Persisted IkaGeneral context packs"
    client_binary: Optional[str] = None
    client_home: Optional[str] = None


@dataclass
class ExecutionCheckpoint:
    """Snapshot of execution memory at a point in time."""

    name: Optional[str]  # None for auto-checkpoints
    step_number: int
    execution_memory: dict  # deep copy of message_history
    loaded_packs: list[str]  # which packs were loaded at this point
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SubAgentDefinition:
    """Definition for a spawnable sub-agent."""

    name: str
    description: str  # system prompt for the sub-agent
    prompt: str  # initial task prompt
    tools: list[str] = field(default_factory=list)  # tool names available
    model: Optional[str] = None  # None = inherit parent model
    model_params: dict = field(default_factory=dict)  # temperature, max_tokens, etc.
    context_packs: list[str] = field(default_factory=list)  # pack names to pre-load
    status: str = "created"  # created, running, completed, failed
    last_result: Any = None  # result from most recent run


@dataclass
class DynamicToolDefinition:
    """Definition for an agent-created dynamic tool."""

    name: str
    description: str
    parameters: dict = field(default_factory=dict)  # JSON schema for tool parameters
    code: str = ""  # Python code string defining execute(args: dict) -> str
    created_at: datetime = field(default_factory=datetime.now)
