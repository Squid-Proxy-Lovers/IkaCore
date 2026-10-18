"""Snapshot transcript and pending-turn replay."""

# pyright: strict

import json
import uuid
from copy import deepcopy
from typing import Any, Optional, cast

from IkaCore.agent_execution_helpers import stage_step_limit
from IkaCore.agent_runtime_payloads import JsonDict, history_section
from IkaCore.execution_types import ExecutionLoopState
from IkaModel.base import AgentEndException, HumanInputRequired
from IkaModel.execution_hooks import current_tool_frame, emit_boundary


def clear_resume_state(agent: Any) -> None:
    agent._resume_state = None
    agent._snapshot_resume_messages = None


def restore_loop_state(agent: Any, step_limit: int, remaining_steps: int) -> ExecutionLoopState:
    raw_resume = getattr(agent, "_resume_state", None)
    resume = cast(JsonDict, raw_resume) if isinstance(raw_resume, dict) else None
    saved = resume.get("loop_state") if resume is not None else None
    session = getattr(agent, "_snapshot_runtime", None)
    if saved is None and resume is not None and resume.get("scope") in {"stage", "hitl"} and session is not None:
        # Older v2 snapshots have remaining budgets but no counters. The stage's
        # entry boundary lets us subtract its consumed allowance without migration.
        entries: list[JsonDict] = session.store.list_snapshots(session.run_id, session.stage_frame_id)
        entry: JsonDict = next((s["state_json"] for s in entries if s["snapshot_kind"] == "frame_entry"), dict[str, Any]())
        initial = entry.get("remaining_steps", remaining_steps)
        consumed = max(0, initial - remaining_steps)
        stage = agent.stages[resume.get("stage_index", 0)]
        step_limit = max(0, stage_step_limit(stage, initial) - consumed)
    state = ExecutionLoopState(**saved) if saved is not None else ExecutionLoopState(step_limit, remaining_steps=remaining_steps)
    agent._resume_state = None
    if session is not None:
        session.loop_state = state
    return state


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
        history_section(message_history, "messages")[str(uuid.uuid4())] = {
            "message": content_prompt, "tokens": 0, "type": "stage_input",
        }
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
    messages = restore_messages(self, [])
    messages.append({"role": "user", "content": resume_input})
    return messages


def delegation_interrupt(child: Any, task: str, result: dict[str, Any],
                         tool_frame: str | None = None) -> dict[str, Any]:
    data = deepcopy(result.get("interrupt_data", {}))
    data["delegated"] = {"agent_name": child.name, "task_input": task,
                         "snapshot_id": result.get("snapshot_id"),
                         "status": result.get("status"), "tool_frame_id": tool_frame}
    data["resumable"] = bool(result.get("snapshot_id"))
    if not data["resumable"]:
        data["error"] = "Call parent.enable_snapshots() before execution to resume delegated pauses."
    return data


def raise_delegated_interrupt(child: Any, task: str, result: dict[str, Any]) -> None:
    if result.get("status") in {"awaiting_user_input", "paused"}:
        raise HumanInputRequired(delegation_interrupt(child, task, result, current_tool_frame()))

def _interrupt_matches(value: JsonDict, snapshot_id: str) -> bool:
    delegated = value.get("delegated")
    return (value.get("__ika_interrupt__") is True and isinstance(delegated, dict)
            and cast(JsonDict, delegated).get("snapshot_id") == snapshot_id)


def _replace_interrupt(value: Any, snapshot_id: str, output: str) -> Any:
    if isinstance(value, str):
        if "__ika_interrupt__" not in value:
            return value
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, dict) and _interrupt_matches(cast(JsonDict, parsed), snapshot_id):
            return output
        updated = _replace_interrupt(parsed, snapshot_id, output)
        return json.dumps(updated) if updated != parsed else value
    if isinstance(value, dict):
        mapped = cast(JsonDict, value)
        if _interrupt_matches(mapped, snapshot_id):
            return {"result": output}
        return {key: _replace_interrupt(item, snapshot_id, output) for key, item in mapped.items()}
    if isinstance(value, list):
        return [_replace_interrupt(item, snapshot_id, output) for item in cast(list[Any], value)]
    return value


def _update_parent_result(agent: Any, state: dict[str, Any], snapshot_id: str, output: str) -> None:
    agent.message_history = _replace_interrupt(agent.message_history, snapshot_id, output)
    agent._snapshot_resume_messages = _replace_interrupt(agent._snapshot_resume_messages, snapshot_id, output)
    agent._snapshot_runtime.live_messages = _replace_interrupt(agent._snapshot_runtime.live_messages, snapshot_id, output)
    state["message_history"] = deepcopy(agent.message_history)

def _validate_delegated_snapshot(session: Any, pending: JsonDict) -> None:
    snapshot = session.store.load_snapshot(pending["snapshot_id"])
    if snapshot is None or snapshot["run_id"] != session.run_id:
        raise ValueError("Delegated snapshot must belong to the parent run")
    root = session.store.get_frame(snapshot["state_json"].get("root_frame_id", snapshot["frame_id"]))
    lineage = session.store.get_frame_lineage(snapshot["frame_id"])
    if (root is None or root["frame_name"] != pending["agent_name"]
            or session.root_frame_id not in {frame["frame_id"] for frame in lineage}):
        raise ValueError("Delegated snapshot must belong to the parent and the configured child")


def resume_delegation(agent: Any, state: dict[str, Any], answer: str | None) -> dict[str, Any] | None:
    data = state.get("interrupt_data", {})
    pending = data["delegated"]
    if pending["status"] == "awaiting_user_input" and (answer is None or not answer.strip()):
        return {"status": "awaiting_user_input", "interrupt_data": data,
                "last_content": state.get("last_content", "")}
    session = agent._snapshot_runtime
    sources = agent.stages[state["stage_index"]].subagents if agent.stages else agent.subagents
    template = next((child for child in sources if child.name == pending["agent_name"]), None)
    if template is None:
        raise ValueError("The interrupted delegated agent is no longer configured")
    _validate_delegated_snapshot(session, pending)
    child = agent.tool_runtime._prepare_subagent_run(template, pending["task_input"], None)
    try:
        child.enable_snapshots(session.store.db_path, deepcopy(session.control))
        result = child.resume_from_snapshot(pending["snapshot_id"], resume_input=answer,
                                            allow_unsafe_replay=session.allow_unsafe_replay)
    finally:
        child.shutdown()
    if result.get("status") in {"awaiting_user_input", "paused"}:
        data = delegation_interrupt(child, pending["task_input"], result, pending.get("tool_frame_id"))
        _update_parent_result(agent, state, pending["snapshot_id"], json.dumps({"__ika_interrupt__": True, **data}))
        return {"status": result["status"], "last_content": result.get("final_message", ""),
                "interrupt_data": data}
    output = result.get("summary") or result.get("final_message", "")
    _update_parent_result(agent, state, pending["snapshot_id"], output)
    if pending.get("tool_frame_id"):
        session.store.update_frame(pending["tool_frame_id"], status="completed", outputs={"result": output})
    state["scope"] = "stage" if agent.stages else "agent"
    state.pop("interrupt_data", None)
    return None
