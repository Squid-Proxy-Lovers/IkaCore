"""Persist a release approval, exit, resume in a new process, and fork a rejection."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from examples.apps.common import agent, call, emit, end, parser, session
from IkaCore import IkaStage, IkaWorkflow, OutputConfig, WorkflowEdge, WorkflowNode


def worker(args) -> dict:
    state = args.state
    state.mkdir(parents=True, exist_ok=True)
    plan = agent(args, "plan", "Propose a harmless documentation release. Call agent_end with exactly DOCS_ONLY.")
    approve = agent(args, "approve", "Obtain a human release decision.", Stages=[IkaStage(
        "approval", "If no human decision is present, call ask_user with question 'Approve docs release?'. "
        "After YES, agent_end with exactly APPROVED. After NO, agent_end with exactly REJECTED.", [],
        hitl=True, stage_max_step=3)])
    ship = agent(args, "ship", "Read upstream approval. Do not publish anything. Call agent_end with exactly "
                 "READY if APPROVED, or HOLD if REJECTED.")
    workflow = IkaWorkflow("release", "Plan, approve, prepare", [WorkflowNode("plan", plan),
                           WorkflowNode("approve", approve), WorkflowNode("ship", ship)],
                          [WorkflowEdge("plan", "approve"), WorkflowEdge("approve", "ship")],
                          output=OutputConfig(cli=False))
    store = workflow.enable_snapshots(str(state / "workflow.db"))
    decision = "REJECTED" if args.phase == "fork" else "APPROVED"
    scripts = ({"plan": [end("DOCS_ONLY")], "approve": [call("ask_user", question="Approve docs release?")]}
               if args.phase == "start" else {"approve": [end(decision)],
                                             "ship": [end("HOLD" if args.phase == "fork" else "READY")]})
    with session(args, scripts):
        try:
            if args.phase == "start":
                result = workflow.run()
                info = workflow.snapshot_runtime_info()
                assert "ship" not in result and list(info["interrupted_instances"]) == ["approve:0"]
                (state / "pending.json").write_text(json.dumps({"id": info["last_snapshot_id"]}))
                return {"paused": True, "dependent_blocked": True}
            snapshot_id = json.loads((state / "pending.json").read_text())["id"]
            if args.phase == "fork":
                fork = store.fork_run_from_snapshot(snapshot_id)
                assert fork["snapshot_id"] != snapshot_id
                snapshot_id = fork["snapshot_id"]
            results = workflow.resume_from_snapshot(snapshot_id, resume_input="NO" if args.phase == "fork" else "YES")
            assert results["approve"].final == decision
            assert results["ship"].final == ("HOLD" if args.phase == "fork" else "READY")
            assert not workflow.snapshot_runtime_info()["interrupted_instances"]
            return {"resumed": True, "alternative": args.phase == "fork", "dependency_preserved": True}
        finally:
            workflow.async_executor.shutdown()
            for item in (plan, approve, ship):
                item.shutdown()


def main() -> None:
    cli = parser(__doc__)
    cli.add_argument("--phase", choices=("demo", "start", "resume", "fork"), default="demo")
    cli.add_argument("--state", type=Path, default=Path("runtime-data/approval-demo"))
    args = cli.parse_args()
    if args.phase != "demo":
        emit("approval_workflow", **worker(args))
        return
    with tempfile.TemporaryDirectory(prefix="ika-approval-") as directory:
        outcomes = []
        for phase in ("start", "resume", "fork"):
            command = [sys.executable, "-m", "examples.apps.approval_workflow.main", "--phase", phase,
                       "--state", directory, "--live" if args.live else "--offline", "--model", args.model]
            if args.use_local_login:
                command.append("--use-local-login")
            completed = subprocess.run(command, capture_output=True, text=True, timeout=100)
            if completed.returncode:
                # No raw live response/exception text in the demo's public output.
                if not args.live:
                    print(completed.stderr, file=sys.stderr)
                raise RuntimeError(f"approval {phase} subprocess failed (exit {completed.returncode})")
            outcomes.append(json.loads(completed.stdout))
        assert all(row["status"] == "passed" for row in outcomes)
        emit("approval_workflow", fresh_process_resume=True, rejection_fork=True, dependent_blocked=True)


if __name__ == "__main__":
    main()
