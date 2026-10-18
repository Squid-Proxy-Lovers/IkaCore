"""Persisted agent-as-tool continuations; human answers belong to the child."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from IkaModel.base import HumanInputRequired
from IkaModel.execution_hooks import current_tool_frame


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


def _replace_interrupt(value: Any, snapshot_id: str, output: str) -> Any:
    if isinstance(value, str):
        if "__ika_interrupt__" not in value:
            return value
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, dict) and parsed.get("delegated", {}).get("snapshot_id") == snapshot_id:
            return output
        updated = _replace_interrupt(parsed, snapshot_id, output)
        return json.dumps(updated) if updated != parsed else value
    if isinstance(value, dict):
        if value.get("__ika_interrupt__") and value.get("delegated", {}).get("snapshot_id") == snapshot_id:
            return {"result": output}
        return {key: _replace_interrupt(item, snapshot_id, output) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_interrupt(item, snapshot_id, output) for item in value]
    return value



def _update_parent_result(agent: Any, state: dict[str, Any], snapshot_id: str, output: str) -> None:
    agent.message_history = _replace_interrupt(agent.message_history, snapshot_id, output)
    agent._snapshot_resume_messages = _replace_interrupt(agent._snapshot_resume_messages, snapshot_id, output)
    agent._snapshot_runtime.live_messages = _replace_interrupt(agent._snapshot_runtime.live_messages, snapshot_id, output)
    state["message_history"] = deepcopy(agent.message_history)

def resume_delegation(agent: Any, state: dict[str, Any], answer: str | None) -> dict[str, Any] | None:
    data = state.get("interrupt_data", {})
    pending = data["delegated"]
    if pending["status"] == "awaiting_user_input" and (answer is None or not answer.strip()):
        return {"status": "awaiting_user_input", "interrupt_data": data,
                "last_content": state.get("last_content", "")}
    session = agent._snapshot_runtime
    sources = agent.Stages[state["stage_index"]].subagents if agent.Stages else agent.subagents
    template = next((child for child in sources if child.name == pending["agent_name"]), None)
    if template is None:
        raise ValueError("The interrupted delegated agent is no longer configured")
    snapshot = session.store.load_snapshot(pending["snapshot_id"])
    if snapshot is None or snapshot["run_id"] != session.run_id:
        raise ValueError("Delegated snapshot must belong to the parent run")
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
    state["message_history"] = deepcopy(agent.message_history)
    state["scope"] = "stage" if agent.Stages else "agent"
    state.pop("interrupt_data", None)
    return None
