# pyright: strict, reportPrivateUsage=false
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

from .execution_types import ExecutionLoopState
from .runtime_control import RuntimeControl
from .snapshot_session import SnapshotSession
from .snapshot_store import SnapshotStore

"AgentSnapshots component; owns one execution responsibility."
if TYPE_CHECKING:
    from .agents import IkaBaseAgent


class AgentSnapshots:
    def __init__(self, agent: IkaBaseAgent):
        self.agent = agent

    def enable_snapshots(
        self, db_path: str = "runtime-data/snapshots.db", control: Optional[RuntimeControl] = None
    ) -> SnapshotStore:
        agent = self.agent
        store = SnapshotStore(db_path)
        agent._snapshot_runtime = SnapshotSession(agent, store, control or RuntimeControl())
        return store

    def configure_runtime_control(
        self,
        *,
        pause_points: Optional[set[str]] = None,
        parallel_replay: Optional[dict[str, list[int]]] = None,
        reuse_historical_parallel_results: bool = True,
    ) -> RuntimeControl:
        agent = self.agent
        if agent._snapshot_runtime is None:
            agent.enable_snapshots()
        session = agent._snapshot_runtime
        if session is None:
            raise ValueError("snapshot runtime could not be enabled")
        control = session.control
        if pause_points is not None:
            control.pause_points = set(pause_points)
        if parallel_replay is not None:
            control.parallel_replay = deepcopy(parallel_replay)
        control.reuse_historical_parallel_results = reuse_historical_parallel_results
        return control

    def request_pause_at(self, *checkpoint_kinds: str) -> None:
        agent = self.agent
        control = (
            agent._snapshot_runtime.control
            if agent._snapshot_runtime is not None
            else agent.configure_runtime_control()
        )
        control.request_pause(*checkpoint_kinds)

    def clear_pause_requests(self, *checkpoint_kinds: str) -> None:
        agent = self.agent
        if agent._snapshot_runtime is not None:
            agent._snapshot_runtime.control.clear_pause(*checkpoint_kinds)

    def resume_from_snapshot(
        self, snapshot_id: str, *, allow_unsafe_replay: bool = False, resume_input: Optional[str] = None
    ) -> dict[str, Any]:
        agent = self.agent
        session: Optional[SnapshotSession] = agent._snapshot_runtime
        if session is None:
            raise ValueError("enable_snapshots must be called before snapshot replay")
        info = session.store.resume_from_snapshot(snapshot_id)
        state = info["state"]
        if "completed_output" in state and resume_input is not None:
            raise ValueError("A completed snapshot has no pending input; restore it, then send(message)")
        if state.get("tool_checkpoint") and (not allow_unsafe_replay):
            raise ValueError("mid-tool snapshots require an explicit unsafe-replay decision")
        root_frame = state.get("root_frame_id", info["frame_id"])
        if "completed_output" not in state and not allow_unsafe_replay and (
            not session.store.assess_frame_replay(root_frame, after_snapshot_id=snapshot_id)["safe"]
        ):
            raise ValueError(
                "historical snapshot contains non-replayable tool calls; explicit unsafe-replay approval required"
            )
        agent._resume_state = deepcopy(state)
        agent.message_history = deepcopy(state.get("message_history", agent.message_history))
        agent._total_usage = deepcopy(state.get("total_usage", agent._total_usage))
        agent._total_cost = deepcopy(state.get("total_cost", agent._total_cost))
        session.run_id, session.root_frame_id = (info["run_id"], state.get("root_frame_id", info["frame_id"]))
        session.stage_index = state.get("stage_index")
        stages = [
            frame for frame in session.store.get_frame_lineage(info["frame_id"]) if frame["frame_type"] == "stage"
        ]
        session.stage_frame_id = stages[-1]["frame_id"] if stages else None
        session.last_snapshot_id, session.skip_boundary = (snapshot_id, state.get("runtime_boundary"))
        if session.skip_boundary == "frame_entry":
            session.skip_boundary = None
        session.pending_response = state.get("pending_response")
        session.loop_state = ExecutionLoopState(**state["loop_state"]) if state.get("loop_state") else None
        session.live_messages = state.get("chat_messages")
        agent._snapshot_resume_messages = deepcopy(session.live_messages)
        agent.maxsteps = state.get("maxsteps", agent.maxsteps)
        agent.state.remaining_steps = state.get("validation_remaining_steps")
        agent.state.prompt = state.get("prompt", agent.prompt)
        agent.state.turn_message = state.get("turn_message")
        agent.state.turn_message_consumed = state.get("turn_message_consumed", True)
        agent.state.compaction_count = state.get("compaction_count", 0)
        agent.state.validation_attempts = state.get("validation_attempts", 0)
        agent.state.validation_maxsteps = state.get("validation_maxsteps")
        if "completed_output" in state:
            # A process can die after committing the completion boundary but
            # before finalizing frame/run metadata. Restore saved accounting and
            # finish that metadata without repeating the model or tool effects.
            root = state.get("root_frame_id", info["frame_id"])
            session.run_id, session.root_frame_id = info["run_id"], root
            session.last_snapshot_id = snapshot_id
            with session.store.transaction():
                session.store.update_frame(info["frame_id"], status="completed")
                session.store.update_frame(root, status="completed", outputs=state["completed_output"])
                run = session.store.get_run(info["run_id"])
                if run is not None and run["root_frame_id"] == root:
                    session.store.update_run_status(info["run_id"], "completed", ended=True)
            agent._resume_state = None
            agent._snapshot_resume_messages = None
            agent._last_interrupt = None
            session.replaying = False
            return {**state["completed_output"], "runtime": {
                "run_id": info["run_id"], "frame_id": info["frame_id"], "last_snapshot_id": snapshot_id}}
        session.allow_unsafe_replay = allow_unsafe_replay
        session.replaying = True
        return agent.execution(resume_input=resume_input)

    def fork_from_snapshot(
        self, snapshot_id: str, *, allow_unsafe_replay: bool = False, resume_input: str | None = None
    ) -> dict[str, Any]:
        agent = self.agent
        if agent._snapshot_runtime is None:
            raise ValueError("enable_snapshots must be called before snapshot replay")
        fork = agent._snapshot_runtime.store.fork_run_from_snapshot(
            snapshot_id, allow_unsafe_replay=allow_unsafe_replay
        )
        return agent.resume_from_snapshot(
            fork["snapshot_id"], allow_unsafe_replay=allow_unsafe_replay, resume_input=resume_input)

    def restart_from_frame(self, frame_id: str, *, allow_unsafe_replay: bool = False) -> dict[str, Any]:
        agent = self.agent
        if agent._snapshot_runtime is None:
            raise ValueError("enable_snapshots must be called before snapshot replay")
        replay = agent._snapshot_runtime.store.restart_frame(frame_id, allow_unsafe_replay=allow_unsafe_replay)
        return agent.resume_from_snapshot(replay["snapshot_id"], allow_unsafe_replay=allow_unsafe_replay)
