"""Multi-turn preference chat with explicit memory, compaction and session resume."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from examples.apps.common import agent, call, emit, end, parser, session
from IkaCore import ExecutionLimits, PromptConfig
from IkaMem import STMemory
from IkaMem.agent_tools import MemoryTools
from IkaModel import request_controls, runtime_options


class JsonStorage:
    """Small persistent exact-substring backend; no external embedding service."""
    def __init__(self, path: Path):
        self.path = path
        self.rows = json.loads(path.read_text()) if path.exists() else []

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        self.rows.append({"data": value, "metadata": dict(metadata)})
        self.path.write_text(json.dumps(self.rows))

    def search(self, query: str, limit: int, score_threshold: float) -> list[Any]:
        return [row for row in self.rows if query.lower() in str(row["data"]).lower()
                and score_threshold <= 1.0][:limit]

    def reset(self) -> None:
        self.rows.clear()
        self.path.write_text("[]")



def worker(args) -> dict:
    args.state.mkdir(parents=True, exist_ok=True)
    storage = JsonStorage(args.state / "memory.json")
    memory = STMemory(storage=storage)
    memory.agent = "chat"
    calls = []
    tools = MemoryTools("chat", short_term=memory, before_call=calls.append).tools(
        access={"short_term_save": True, "short_term_search": True,
                "long_term_save": False, "long_term_search": False})
    helper = agent(args, "chat", "You are a concise preference assistant. Call agent_end after answering each turn.", tools=tools,
                   limits=ExecutionLimits(maxsteps=2, max_tool_rounds=2, max_tool_calls=4, step_timeout=30, context_budget=700),
                   prompts=PromptConfig(system_prompt="Be concise. Call agent_end with input containing your answer."))
    # A deliberately small budget makes compaction reproducible with short synthetic turns.
    helper.enable_snapshots(str(args.state / "chat.db"))
    scripts = {"chat": ([call("short_term_save", input="Preferred color: blue"), end("Remembered blue.")] +
                        [end("Acknowledged blue.")] * (args.turns - 1)) if args.phase == "first" else
                       [call("short_term_search", query="blue"), end("Your preferred color is blue.")]}
    args.session_timeout = 180
    with session(args, scripts), runtime_options(compact_at_fraction=0.55):
        try:
            if args.phase == "second":
                saved = json.loads((args.state / "session.json").read_text())
                # Consume the saved post-model turn in this new process without another request.
                helper.clear_pause_requests()
                old = helper.resume_from_snapshot(saved["snapshot_id"])
                assert old["final_message"]
                with request_controls(timeout=30):
                    result = helper.send("TASK:chat\nUse short_term_search to recall my preferred color, then agent_end with it.")
                assert "blue" in result["final_message"].lower() and "short_term_search" in calls
                return {"memory_recalled": True, "fresh_process_resume": True}
            compactions = 0
            result: dict[str, Any] = {}
            for turn in range(args.turns):
                prompt = ("TASK:chat\nRemember my preferred color is blue using short_term_save, then agent_end."
                                 if turn == 0 else f"TASK:chat\nTurn {turn}: acknowledge my blue preference. "
                                 "Fixture meeting notes: " + "discuss lighting pilot and retain preference; " * 3 +
                                 "Reply briefly using agent_end.")
                if turn == args.turns - 1:
                    helper.request_pause_at("post_model")
                with request_controls(timeout=30):
                    result = helper.send(prompt)
                assert result.get("status") == "paused" if turn == args.turns - 1 else result["final_message"]
                compactions = helper.compaction_count
            assert "short_term_save" in calls and compactions > 0
            (args.state / "session.json").write_text(json.dumps({"snapshot_id": result["snapshot_id"]}))
            return {"turns": args.turns, "compactions": compactions, "memory_saved": True}
        finally:
            helper.shutdown()


def main() -> None:
    cli = parser(__doc__)
    cli.add_argument("--phase", choices=("demo", "first", "second"), default="demo")
    cli.add_argument("--turns", type=int, choices=range(4, 25), default=12)
    cli.add_argument("--state", type=Path, default=Path("runtime-data/chat-demo"))
    args = cli.parse_args()
    if args.phase != "demo":
        emit("long_chat", **worker(args))
        return
    with tempfile.TemporaryDirectory(prefix="ika-chat-") as directory:
        outcomes = []
        for phase in ("first", "second"):
            command = [sys.executable, "-m", "examples.apps.long_chat.main", "--phase", phase, "--state", directory,
                       "--turns", str(args.turns), "--live" if args.live else "--offline", "--model", args.model]
            if args.use_local_login:
                command.append("--use-local-login")
            completed = subprocess.run(command, capture_output=True, text=True, timeout=195)
            if completed.returncode:
                if not args.live:
                    print(completed.stderr, file=sys.stderr)
                raise RuntimeError(f"chat {phase} subprocess failed (exit {completed.returncode})")
            outcomes.append(json.loads(completed.stdout))
        emit("long_chat", turns=args.turns, compactions=outcomes[0]["compactions"], fresh_process_resume=True,
             memory_recalled=True)


if __name__ == "__main__":
    main()
