"""Conversation turns, deliberately separate from task replacement."""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from .agent_helper_support import initial_message_history

if TYPE_CHECKING:
    from .agents import IkaBaseAgent


def replace_prompt(agent: IkaBaseAgent, value: str) -> None:
    """Assigning prompt starts a fresh task; send() retains the conversation."""
    agent.state.prompt = value
    agent.state.turn_message = None
    agent.state.message_history = initial_message_history(getattr(agent, "system_prompt", None))
    agent.state._resume_state = None
    agent._snapshot_resume_messages = None
    for key in ("_workflow_context", "_workflow_context_label", "_workflow_base_input", "_workflow_prompt"):
        agent.__dict__.pop(key, None)
    agent.state._last_interrupt = None
    agent.state.maxsteps = agent.config.limits.maxsteps
    agent.state.validation_attempts = 0
    agent.state.validation_maxsteps = None
    session = getattr(agent, "_snapshot_runtime", None)
    if session is not None:
        from .snapshot_session import SnapshotSession
        agent._snapshot_runtime = SnapshotSession(agent, session.store, session.control)


def send(agent: IkaBaseAgent, message: str) -> dict[str, Any]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("send requires a nonempty user message")
    if agent.state._last_interrupt or agent.state._resume_state:
        raise ValueError("Resume the pending turn before sending a new message")
    session = agent._snapshot_runtime
    if session is not None and session.root_frame_id is not None:
        frame = session.store.get_frame(session.root_frame_id)
        if frame is not None and frame["status"] == "paused":
            raise ValueError("Resume the pending turn before sending a new message")
    agent.state.prompt = message
    agent.state.turn_message = message
    agent.maxsteps = agent.config.limits.maxsteps
    return agent.execution()


def transcript(agent: IkaBaseAgent) -> dict[str, Any]:
    """A defensive copy of the replayable transcript, including its current summary."""
    return deepcopy(agent.message_history)
