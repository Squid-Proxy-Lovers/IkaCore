"""Small compatibility seams for snapshot transcript and pending-turn replay."""

# pyright: strict

import uuid
from typing import Any, Optional, cast

from IkaCore.agent_runtime_payloads import JsonDict, history_section
from IkaModel.base import AgentEndException
from IkaModel.execution_hooks import emit_boundary


def resume_pending_response(agent: Any) -> Optional[dict[str, Any]]:
    session = getattr(agent, "_snapshot_runtime", None)
    if session is None or session.pending_response is None:
        return None
    saved = session.pending_response
    session.pending_response, session.skip_boundary = None, None
    if saved.get("agent_end_exception"):
        raise AgentEndException(saved["response"])
    return saved["response"]


def restore_messages(agent: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    saved = getattr(agent, "_snapshot_resume_messages", None)
    if isinstance(saved, list) and saved:
        agent._snapshot_resume_messages = None
        return cast(list[dict[str, Any]], saved)
    return messages


def stage_boundary(stage_index: int, remaining_steps: int, name: str, messages: list[dict[str, Any]]) -> None:
    emit_boundary("stage_entry", {"stage_index": stage_index, "remaining_steps": remaining_steps,
                                  "name": name, "messages": messages})


def stage_initial_messages(
    self: Any,
    stage: Any,
    resume_input: Optional[str],
    message_history: JsonDict,
    content_prompt: str,
) -> list[JsonDict]:
    if resume_input is None:
        from IkaModel.runtime_policy import current_runtime_options
        if current_runtime_options().optimize_provider_payloads:
            history_section(message_history, "messages")[str(uuid.uuid4())] = {
                "message": content_prompt, "tokens": 0, "type": "stage_input",
            }
        from .snapshot_agent_state import restore_messages
        return restore_messages(self, [{"role": "user", "content": content_prompt}])
    if self.logger:
        self.logger.log_hitl_input(stage.name, resume_input)
    if resume_input.strip():
        messages_bucket = history_section(message_history, "messages")
        messages_bucket[str(uuid.uuid4())] = {
            "message": resume_input,
            "tokens": 0,
            "type": "hitl_input",
        }
    return [{"role": "user", "content": resume_input}]

