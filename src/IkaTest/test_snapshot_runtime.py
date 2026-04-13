import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

mock_ika_mem = MagicMock()
with patch.dict("sys.modules", {"IkaMem": mock_ika_mem}):
    from IkaCore.agents import IkaBaseAgent
from IkaCore.checkpoint import CheckpointStore
from IkaCore.runtime_control import RuntimePauseRequested
from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools
from IkaCore.workflow import IkaWorkflow, WorkflowEdge, WorkflowNode


def _minimal_agent(tmp_path: Path, **kwargs) -> IkaBaseAgent:
    defaults = {
        "name": "RootAgent",
        "description": "Test agent",
        "prompt": "Do the task",
        "model_id": "gpt-4o",
        "api_key": "test-key",
        "checkpoint": True,
        "checkpoint_db_path": str(tmp_path / "snapshots.db"),
    }
    defaults.update(kwargs)
    return IkaBaseAgent(**defaults)


class ReplayStubAgent(IkaBaseAgent):
    def __init__(self, *args, call_log=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_log = call_log if call_log is not None else []

    def execution(self, checkpoint_uid=None):
        instance_id = getattr(self, "_runtime_workflow_instance_id", 0)
        self.call_log.append(instance_id)
        frame_id = self._ensure_runtime_entry_frame(
            invocation_type=self._runtime_invocation_type or "workflow_node",
            position={"mode": "workflow", "instance_id": instance_id},
            inputs={"prompt": self.prompt},
        )
        answer = f"{self.name}:{instance_id}"
        self._complete_runtime_frame(
            frame_id,
            status="completed",
            outputs={"final_message": answer, "summary": answer},
        )
        return {
            "final_message": answer,
            "summary": answer,
            "runtime": {
                "run_id": self._runtime_run_id,
                "frame_id": frame_id,
                "last_snapshot_id": getattr(self, "_last_snapshot_id", None),
            },
        }


class SharedLog(list):
    def __deepcopy__(self, memo):
        return self


class TestRuntimeStore:
    def test_runtime_store_persists_runs_frames_snapshots_and_events(self, tmp_path: Path):
        store = CheckpointStore(str(tmp_path / "runtime.db"))
        run_id = store.create_run("agent", "root")
        root_frame = store.create_frame(run_id, "agent", "root")
        store.set_run_root_frame(run_id, root_frame)
        child_frame = store.create_frame(
            run_id,
            "subagent_call",
            "child",
            parent_frame_id=root_frame,
            caller_frame_id=root_frame,
            return_to_frame_id=root_frame,
        )
        snapshot_id = store.create_snapshot(
            run_id,
            child_frame,
            "boundary",
            {"scope": "agent", "hello": "world"},
            label="child-boundary",
        )
        store.record_event(run_id, child_frame, "child_completed", {"ok": True})

        snapshot = store.load_snapshot(snapshot_id)
        assert snapshot is not None
        assert snapshot["state_json"]["hello"] == "world"

        tree = store.get_call_tree(run_id)
        assert len(tree) == 1
        assert tree[0]["frame_id"] == root_frame
        assert len(tree[0]["children"]) == 1
        assert tree[0]["children"][0]["frame_id"] == child_frame
        assert tree[0]["children"][0]["return_to_frame_id"] == root_frame

        events = store.list_events(run_id)
        assert len(events) == 1
        assert events[0]["event_type"] == "child_completed"
        assert events[0]["payload_json"]["ok"] is True

    def test_fork_run_from_snapshot_clones_lineage_into_new_run(self, tmp_path: Path):
        store = CheckpointStore(str(tmp_path / "runtime.db"))
        run_id = store.create_run("agent", "root")
        root_frame = store.create_frame(run_id, "agent", "root", position={"level": 0})
        store.set_run_root_frame(run_id, root_frame)
        child_frame = store.create_frame(
            run_id,
            "stage",
            "child",
            parent_frame_id=root_frame,
            caller_frame_id=root_frame,
            return_to_frame_id=root_frame,
            position={"level": 1},
        )
        snapshot_id = store.create_snapshot(
            run_id,
            child_frame,
            "frame_entry",
            {
                "scope": "stage",
                "run_id": run_id,
                "frame_id": child_frame,
                "root_frame_id": root_frame,
                "parent_frame_id": root_frame,
                "return_to_frame_id": root_frame,
            },
        )

        forked = store.fork_run_from_snapshot(snapshot_id)
        assert forked["run_id"] != run_id
        assert forked["frame_id"] != child_frame

        new_run = store.get_run(forked["run_id"])
        assert new_run["branch_from_snapshot_id"] == snapshot_id
        frames = store.list_frames(forked["run_id"])
        assert len(frames) == 2
        root_clone = next(frame for frame in frames if frame["parent_frame_id"] is None)
        child_clone = next(frame for frame in frames if frame["parent_frame_id"] == root_clone["frame_id"])
        assert child_clone["frame_name"] == "child"
        assert child_clone["return_to_frame_id"] == root_clone["frame_id"]

    def test_snapshot_chain_uses_linked_history(self, tmp_path: Path):
        store = CheckpointStore(str(tmp_path / "runtime.db"))
        run_id = store.create_run("agent", "root")
        frame_id = store.create_frame(run_id, "agent", "root")
        first = store.create_snapshot(run_id, frame_id, "frame_entry", {"step": 0})
        second = store.create_snapshot(run_id, frame_id, "pre_model", {"step": 1})
        third = store.create_snapshot(run_id, frame_id, "post_model", {"step": 2})

        chain = store.get_snapshot_chain(third)
        assert [item["snapshot_id"] for item in chain] == [first, second, third]
        assert chain[1]["prev_snapshot_id"] == first
        assert chain[2]["prev_snapshot_id"] == second

    def test_assess_frame_replay_blocks_non_replayable_tool_events(self, tmp_path: Path):
        store = CheckpointStore(str(tmp_path / "runtime.db"))
        run_id = store.create_run("agent", "root")
        frame_id = store.create_frame(run_id, "agent", "root")
        store.record_event(
            run_id,
            frame_id,
            "tool_completed",
            {
                "tool_name": "write_to_db",
                "side_effect_type": "side_effecting",
                "replay_policy": "deny",
            },
        )

        assessment = store.assess_frame_replay(frame_id)
        assert assessment["safe"] is False
        assert assessment["blockers"][0]["tool_name"] == "write_to_db"

    def test_restart_frame_requires_allow_unsafe_for_denied_replay(self, tmp_path: Path):
        store = CheckpointStore(str(tmp_path / "runtime.db"))
        run_id = store.create_run("agent", "root")
        frame_id = store.create_frame(run_id, "agent", "root")
        store.set_run_root_frame(run_id, frame_id)
        store.create_snapshot(
            run_id,
            frame_id,
            "frame_entry",
            {"scope": "agent", "run_id": run_id, "frame_id": frame_id, "root_frame_id": frame_id},
        )
        store.record_event(
            run_id,
            frame_id,
            "tool_completed",
            {
                "tool_name": "write_to_db",
                "side_effect_type": "side_effecting",
                "replay_policy": "deny",
            },
        )

        with pytest.raises(ValueError) as exc:
            store.restart_frame(frame_id)
        assert "non-replayable" in str(exc.value)

        replay = store.restart_frame(frame_id, allow_unsafe_replay=True)
        assert replay["run_id"] != run_id
        assert replay["snapshot_id"] is not None


class TestAgentSnapshots:
    def test_execution_creates_runtime_entry_and_boundary_snapshots(self, tmp_path: Path):
        agent = _minimal_agent(tmp_path)

        def fake_chat_wrapper(*args, **kwargs):
            return {
                "message_history": agent.message_history,
                "content": "",
                "content_before_tools": "",
                "tool_calls": [{"function": {"name": "agent_end", "arguments": '{"input":"done"}'}}],
                "executed_tool_calls": [{"function": {"name": "agent_end", "arguments": '{"input":"done"}'}}],
                "usage": {},
                "cost": {},
            }

        agent.chat_wrapper = fake_chat_wrapper  # type: ignore[method-assign]
        result = agent.execution()

        assert result["final_message"] == "done"
        runtime = result["runtime"]
        assert runtime["run_id"] is not None
        assert runtime["frame_id"] is not None

        snapshots = agent.checkpoint_store.list_snapshots(runtime["run_id"])
        assert len(snapshots) >= 1
        assert snapshots[0]["snapshot_kind"] == "frame_entry"
        assert snapshots[0]["state_json"]["agent_name"] == "RootAgent"

    def test_subagent_execution_persists_parent_child_call_tree(self, tmp_path: Path):
        child = _minimal_agent(tmp_path, name="ChildAgent", checkpoint=False)
        child.chat_wrapper = lambda *args, **kwargs: {  # type: ignore[method-assign]
            "message_history": child.message_history,
            "content": "",
            "content_before_tools": "",
            "tool_calls": [{"function": {"name": "agent_end", "arguments": '{"input":"child result"}'}}],
            "executed_tool_calls": [{"function": {"name": "agent_end", "arguments": '{"input":"child result"}'}}],
            "usage": {},
            "cost": {},
        }
        parent = _minimal_agent(tmp_path, subagents=[child])
        parent._runtime_run_id = parent.checkpoint_store.create_run("agent", parent.name)
        parent._runtime_frame_id = parent.checkpoint_store.create_frame(
            parent._runtime_run_id,
            "agent",
            parent.name,
        )
        parent.checkpoint_store.set_run_root_frame(parent._runtime_run_id, parent._runtime_frame_id)

        executors = parent.build_tool_executors(parent.build_simple_tools(), parent_hierarchy=[parent.name])
        output = executors["ChildAgent"]({"input": "delegate"})

        assert "child result" in output
        frames = parent.checkpoint_store.list_frames(parent._runtime_run_id)
        assert len(frames) == 2
        child_frame = next(frame for frame in frames if frame["frame_name"] == "ChildAgent")
        assert child_frame["frame_type"] == "subagent_call"
        assert child_frame["parent_frame_id"] == parent._runtime_frame_id
        assert child_frame["return_to_frame_id"] == parent._runtime_frame_id

    def test_stage_checkpoint_load_binds_to_existing_runtime_frames(self, tmp_path: Path):
        stage = IkaStage("Analysis", "Inspect", [], checkpoint=True)
        agent = _minimal_agent(tmp_path, Stages=[stage], maxsteps=5)

        def fake_chat_wrapper(*args, **kwargs):
            return {
                "message_history": agent.message_history,
                "content": "stage content",
                "content_before_tools": "",
                "tool_calls": [],
                "executed_tool_calls": [],
                "usage": {},
                "cost": {},
            }

        agent.chat_wrapper = fake_chat_wrapper  # type: ignore[method-assign]
        agent._runtime_run_id = agent.checkpoint_store.create_run("agent", agent.name)
        agent._runtime_frame_id = agent.checkpoint_store.create_frame(
            agent._runtime_run_id,
            "agent",
            agent.name,
        )
        agent.checkpoint_store.set_run_root_frame(agent._runtime_run_id, agent._runtime_frame_id)

        agent.execute_stage(0, 2)
        snapshots = agent.checkpoint_store.list_snapshots(agent._runtime_run_id)
        boundary = next(item for item in snapshots if item["snapshot_kind"] == "boundary")

        resumed = _minimal_agent(tmp_path, Stages=[IkaStage("Analysis", "Inspect", [], checkpoint=True)], maxsteps=5)
        payload = resumed.load_checkpoint(boundary["snapshot_id"])

        assert payload is not None
        assert resumed._runtime_run_id == agent._runtime_run_id
        assert resumed._runtime_frame_id == agent._runtime_frame_id
        assert resumed._runtime_stage_frame_ids[0] == boundary["frame_id"]

    def test_resume_restart_and_fork_agent_helpers(self, tmp_path: Path):
        def make_chat_wrapper(bound_agent: IkaBaseAgent):
            def fake_chat_wrapper(*args, **kwargs):
                return {
                    "message_history": bound_agent.message_history,
                    "content": "",
                    "content_before_tools": "",
                    "tool_calls": [{"function": {"name": "agent_end", "arguments": '{"input":"done"}'}}],
                    "executed_tool_calls": [{"function": {"name": "agent_end", "arguments": '{"input":"done"}'}}],
                    "usage": {},
                    "cost": {},
                }
            return fake_chat_wrapper

        agent = _minimal_agent(tmp_path)

        agent.chat_wrapper = make_chat_wrapper(agent)  # type: ignore[method-assign]
        first = agent.execution()
        run_id = first["runtime"]["run_id"]
        frame_id = first["runtime"]["frame_id"]
        snapshots = agent.checkpoint_store.list_snapshots(run_id, frame_id)
        entry_snapshot = next(item for item in snapshots if item["snapshot_kind"] == "frame_entry")

        resumed = _minimal_agent(tmp_path)
        resumed.chat_wrapper = make_chat_wrapper(resumed)  # type: ignore[method-assign]
        resumed_result = resumed.resume_from_snapshot(entry_snapshot["snapshot_id"])
        assert resumed_result["final_message"] == "done"
        assert resumed_result["runtime"]["run_id"] == run_id

        restarted = _minimal_agent(tmp_path)
        restarted.chat_wrapper = make_chat_wrapper(restarted)  # type: ignore[method-assign]
        restarted_result = restarted.restart_from_frame(frame_id)
        assert restarted_result["final_message"] == "done"
        assert restarted_result["runtime"]["run_id"] != run_id

        forked = _minimal_agent(tmp_path)
        forked.chat_wrapper = make_chat_wrapper(forked)  # type: ignore[method-assign]
        forked_result = forked.fork_from_snapshot(entry_snapshot["snapshot_id"])
        assert forked_result["final_message"] == "done"
        assert forked_result["runtime"]["run_id"] != run_id

    def test_tool_execution_records_replay_metadata(self, tmp_path: Path):
        calls = []

        def writer(args):
            calls.append(args["value"])
            return "written"

        tool = IkaTools(
            "writer",
            "write data",
            {"value": {"type": "string", "description": "value", "required": True}},
            execute_function=writer,
            side_effect_type="side_effecting",
            replay_policy="deny",
        )
        agent = _minimal_agent(tmp_path, tools=[tool])
        agent._runtime_run_id = agent.checkpoint_store.create_run("agent", agent.name)
        agent._runtime_frame_id = agent.checkpoint_store.create_frame(agent._runtime_run_id, "agent", agent.name)
        agent.checkpoint_store.set_run_root_frame(agent._runtime_run_id, agent._runtime_frame_id)

        executors = agent.build_tool_executors(agent.tools)
        assert executors["writer"]({"value": "x"}) == "written"
        events = agent.checkpoint_store.list_frame_events(agent._runtime_frame_id)
        completed = [event for event in events if event["event_type"] == "tool_completed"]
        assert completed
        payload = completed[0]["payload_json"]
        assert payload["tool_name"] == "writer"
        assert payload["side_effect_type"] == "side_effecting"
        assert payload["replay_policy"] == "deny"

    def test_pause_request_at_pre_model_returns_paused_result(self, tmp_path: Path):
        agent = _minimal_agent(tmp_path)
        agent.request_pause_at("pre_model")
        agent.chat_wrapper = lambda *args, **kwargs: agent._runtime_checkpoint("pre_model")  # type: ignore[method-assign]

        result = agent.execution()
        assert result["runtime"]["paused"] is True
        assert result["runtime"]["checkpoint_kind"] == "pre_model"
        assert result["runtime"]["snapshot_id"] is not None

    def test_generator_tool_emits_mid_tool_snapshots_and_can_pause(self, tmp_path: Path):
        def streaming_tool(args):
            yield {"chunk": 1}
            yield {"chunk": 2}
            return "stream-complete"

        tool = IkaTools(
            "streamer",
            "stream data",
            {"value": {"type": "string", "description": "value", "required": True}},
            execute_function=streaming_tool,
        )
        agent = _minimal_agent(tmp_path, tools=[tool])
        agent._runtime_run_id = agent.checkpoint_store.create_run("agent", agent.name)
        agent._runtime_frame_id = agent.checkpoint_store.create_frame(agent._runtime_run_id, "agent", agent.name)
        agent.checkpoint_store.set_run_root_frame(agent._runtime_run_id, agent._runtime_frame_id)
        agent.request_pause_at("tool_progress:*")

        executors = agent.build_tool_executors(agent.tools)
        with pytest.raises(Exception) as exc:
            executors["streamer"]({"value": "x"})
        assert exc.type.__name__ == "RuntimePauseRequested"
        assert exc.value.checkpoint_kind.startswith("tool_progress:streamer:")
        snapshots = agent.checkpoint_store.list_snapshots(agent._runtime_run_id, agent._runtime_frame_id)
        assert any(item["snapshot_kind"].startswith("tool_progress:streamer:") for item in snapshots)


class TestWorkflowRuntime:
    def test_workflow_registers_node_frames_under_workflow_run(self, tmp_path: Path):
        a = _minimal_agent(tmp_path, name="NodeA")
        b = _minimal_agent(tmp_path, name="NodeB", checkpoint=False)

        def stub_execution(agent: IkaBaseAgent, answer: str):
            def _runner(checkpoint_uid=None):
                frame_id = agent._ensure_runtime_entry_frame(
                    invocation_type=agent._runtime_invocation_type or "workflow_node",
                    position={"mode": "workflow"},
                    inputs={"prompt": agent.prompt},
                )
                agent._complete_runtime_frame(
                    frame_id,
                    status="completed",
                    outputs={"final_message": answer, "summary": answer},
                )
                return {
                    "final_message": answer,
                    "summary": answer,
                    "runtime": {
                        "run_id": agent._runtime_run_id,
                        "frame_id": frame_id,
                        "last_snapshot_id": None,
                    },
                }

            return _runner

        a.execution = stub_execution(a, "A done")  # type: ignore[method-assign]
        b.execution = stub_execution(b, "B done")  # type: ignore[method-assign]

        workflow = IkaWorkflow(
            name="TestWorkflow",
            description="workflow runtime",
            nodes=[
                WorkflowNode(name="a", agent=a),
                WorkflowNode(name="b", agent=b),
            ],
            edges=[WorkflowEdge(source="a", target="b", edge_type="next")],
            compress_hook=lambda contexts, agent: "\n".join(contexts),
        )

        results = workflow.run()
        assert set(results.keys()) == {"a", "b"}

        run_id = workflow._runtime_run_id
        assert run_id is not None
        frames = a.checkpoint_store.list_frames(run_id)
        workflow_frame = next(frame for frame in frames if frame["frame_type"] == "workflow")
        node_frames = [frame for frame in frames if frame["frame_type"] == "workflow_node"]

        assert workflow_frame["frame_name"] == "TestWorkflow"
        assert len(node_frames) >= 2
        assert all(frame["parent_frame_id"] == workflow_frame["frame_id"] for frame in node_frames)

    def test_async_parallel_replay_reruns_selected_instances_and_reuses_history(self, tmp_path: Path):
        call_log = SharedLog()
        agent = ReplayStubAgent(
            name="ParallelNode",
            description="parallel",
            prompt="Do work",
            model_id="gpt-4o",
            api_key="test-key",
            checkpoint=True,
            checkpoint_db_path=str(tmp_path / "parallel.db"),
            call_log=call_log,
        )

        workflow = IkaWorkflow(
            name="ParallelWorkflow",
            description="parallel runtime",
            nodes=[WorkflowNode(name="parallel", agent=agent, instances=3)],
            edges=[],
            compress_hook=lambda contexts, _: "\n".join(contexts),
        )

        first_results = workflow.run(use_async=True)
        first_run_id = workflow._runtime_run_id
        assert first_run_id is not None
        assert sorted(call_log) == [0, 1, 2]

        call_log.clear()
        replay_workflow = IkaWorkflow(
            name="ParallelWorkflow",
            description="parallel runtime",
            nodes=[WorkflowNode(name="parallel", agent=agent, instances=3)],
            edges=[],
            compress_hook=lambda contexts, _: "\n".join(contexts),
        )
        replay_results = replay_workflow.run(
            use_async=True,
            replay_from_run_id=first_run_id,
            parallel_replay={"parallel": [1]},
        )

        assert call_log == [1]
        summary = replay_results["parallel"].summary
        assert "[Instance 0]:" in summary
        assert "[Instance 1]:" in summary
        assert "[Instance 2]:" in summary
