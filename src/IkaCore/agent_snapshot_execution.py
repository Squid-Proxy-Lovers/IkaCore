"""Run opt-in snapshot sessions around the unchanged agent dispatcher."""

# pyright: strict

from collections.abc import Callable
from typing import Any, Optional

from IkaModel.execution_hooks import execution_hooks
from IkaModel.request_control import check_request_controls
from IkaModel.runtime_errors import IkaRequestControlError

from .runtime_control import RuntimePauseRequested
from .snapshot_agent_hooks import ensure_session, hooks_for_session
from .snapshot_session import SnapshotSession, record_boundary


def runtime_info(session: SnapshotSession) -> dict[str, Any]:
    return {"run_id": session.run_id, "frame_id": session.root_frame_id, "last_snapshot_id": session.last_snapshot_id}


def execute_with_snapshots(agent: Any, execute: Callable[..., dict[str, Any]], checkpoint_uid: Optional[str],
                           resume_input: Optional[str]) -> dict[str, Any]:
    session = ensure_session(agent)
    if session is None:
        return execute(checkpoint_uid=checkpoint_uid, resume_input=resume_input)
    session.replaying = False
    session.store.update_run_status(str(session.run_id), "running")
    finished = False
    with execution_hooks(hooks_for_session(session)):
        try:
            if session.last_snapshot_id is None:
                record_boundary(session, "frame_entry", {})
            check_request_controls()
            result = execute(checkpoint_uid=checkpoint_uid, resume_input=resume_input)
            if result.get("status") == "awaiting_user_input":
                snapshot = record_boundary(session, "human_input", result)
                session.store.update_frame(str(session.root_frame_id), status="paused")
                session.store.update_run_status(str(session.run_id), "paused")
                finished = True
                return {**result, "checkpoint_uid": snapshot, "runtime": runtime_info(session)}
            record_boundary(session, "agent_completed", {"last_content": result.get("final_message", ""), "output": result})
            if session.stage_frame_id is not None:
                session.store.update_frame(session.stage_frame_id, status="completed")
            session.store.update_frame(str(session.root_frame_id), status="completed", outputs=result)
            run = session.store.get_run(str(session.run_id))
            if run is not None and run["root_frame_id"] == session.root_frame_id:
                session.store.update_run_status(str(session.run_id), "completed", ended=True)
            finished = True
            return {**result, "runtime": runtime_info(session)}
        except RuntimePauseRequested as paused:
            finished = True
            return {"status": "paused", "final_message": "", "summary": "", "checkpoint_uid": paused.snapshot_id,
                    "runtime": runtime_info(session), "pause_point": paused.checkpoint_kind}
        except IkaRequestControlError:
            finished = True
            session.store.update_frame(str(session.root_frame_id), status="cancelled")
            session.store.update_run_status(str(session.run_id), "cancelled", ended=True)
            raise
        finally:
            if not finished:
                session.store.update_frame(str(session.root_frame_id), status="failed")
                session.store.update_run_status(str(session.run_id), "failed", ended=True)
