"""Mixin providing execution-control meta-tool executors."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution_env import IkaExecutionEnvironment

LOG = logging.getLogger(__name__)


class ExecutionControlMixin:
    """Executor methods for execution-control meta-tools.

    Mixed into :class:`IkaExecutionEnvironment`.
    """

    # ── branch ─────────────────────────────────────────────────────────────

    async def _exec_branch_execution_path(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        # Deferred import to avoid circular reference
        from .execution_env import IkaExecutionEnvironment

        current_depth = len(self._agent_hierarchy)
        max_depth = 3
        if current_depth >= max_depth:
            return json.dumps({
                "status": "blocked_depth_limit",
                "error": (
                    "Branch execution blocked. Maximum chain depth is 3 "
                    "(root -> level1 -> level2)."
                ),
                "current_chain": " -> ".join(self._agent_hierarchy),
                "max_depth": max_depth,
                "next_required_action": (
                    "Do the comparison or synthesis in the current agent instead of branching again."
                ),
            })

        if self._running_subagents:
            return json.dumps({
                "error": "Cannot branch while sub-agents are running"
            })

        prompts_str = args.get("prompts", "[]")
        try:
            prompts = (
                json.loads(prompts_str)
                if isinstance(prompts_str, str)
                else prompts_str
            )
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in prompts"})

        if not isinstance(prompts, list) or len(prompts) < 2:
            return json.dumps({
                "error": "prompts must be a JSON array with at least 2 items"
            })

        n = len(prompts)

        async def _run_branch(prompt: str, idx: int) -> dict:
            branch_budget = None
            if self._cost_budget is not None:
                remaining = max(0.0, self._cost_budget - self._total_cost)
                branch_budget = remaining / n

            branch_name = f"branch_{idx}"

            sub_env = IkaExecutionEnvironment(
                model_id=self._model_id,
                api_key=self._api_key,
                api_url=self._api_url,
                system_prompt=self._system_prompt,
                initial_prompt=prompt,
                tools=list(self._user_tools),
                ccp_context=self._ccp_context_config,
                cost_budget=branch_budget,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                _execution_name=branch_name,
                _agent_hierarchy=self._agent_hierarchy + [branch_name],
                _trace_writer=self._trace_writer,
                _parent_env=self,
                _shared_context_registry=self._context_pack_registry,
                _shared_lock=self._context_lock,
            )
            sub_env._loaded_packs = list(self._loaded_packs)

            try:
                result = await sub_env.run()
                return {
                    "branch": idx,
                    "prompt": prompt[:200],
                    "status": "completed",
                    "result": str(result),
                    "cost": sub_env._total_cost,
                }
            except Exception as e:
                return {
                    "branch": idx,
                    "prompt": prompt[:200],
                    "status": "failed",
                    "error": str(e),
                    "cost": sub_env._total_cost,
                }

        tasks = [_run_branch(p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        branch_results: list = []
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                branch_results.append({
                    "branch": i, "status": "error", "error": str(r),
                })
            else:
                branch_results.append(r)
                if isinstance(r, dict):
                    self._total_cost += r.get("cost", 0)

        return json.dumps({"branches": branch_results, "total": n})

    # ── reset_to_checkpoint ────────────────────────────────────────────────

    def _exec_reset_to_checkpoint(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        target = args.get("target", "")

        checkpoint = None
        try:
            step_num = int(target)
            for cp in self._checkpoints:
                if cp.step_number == step_num:
                    checkpoint = cp
                    break
        except ValueError:
            for cp in self._checkpoints:
                if cp.name == target:
                    checkpoint = cp
                    break

        if checkpoint is None:
            available = [
                f"'{cp.name or f'auto@{cp.step_number}'}' (step {cp.step_number})"
                for cp in self._checkpoints
            ]
            return json.dumps({
                "error": f"Checkpoint '{target}' not found",
                "available": available,
            })

        self._pending_reset = checkpoint
        return json.dumps({
            "status": "reset_pending",
            "target": target,
            "step": checkpoint.step_number,
            "note": "Execution memory will be restored after this tool round.",
        })

    # ── create_checkpoint ──────────────────────────────────────────────────

    def _exec_create_checkpoint(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        name = args.get("name", "")
        cp = self._create_checkpoint(name)
        return json.dumps({
            "status": "created", "name": name, "step": cp.step_number,
        })

    # ── view_checkpoints ───────────────────────────────────────────────────

    def _exec_view_checkpoints(
        self: "IkaExecutionEnvironment", _args: dict
    ) -> str:
        checkpoints = []
        for cp in self._checkpoints:
            checkpoints.append({
                "name": cp.name or "(auto)",
                "step": cp.step_number,
                "loaded_packs": cp.loaded_packs,
                "created_at": cp.created_at.isoformat(),
            })
        return json.dumps({"checkpoints": checkpoints, "total": len(checkpoints)})

    # ── view_execution_history ─────────────────────────────────────────────

    def _exec_view_execution_history(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        limit = args.get("limit", 10) or 10
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                limit = 10

        history: list = []
        if self._message_history:
            messages = self._message_history.get("messages", {})
            items = list(messages.items())[-limit:]
            for msg_id, msg in items:
                history.append({
                    "id": msg_id[:8],
                    "type": msg.get("type", "message"),
                    "tokens": msg.get("tokens", 0),
                    "preview": str(msg.get("message", ""))[:300],
                })

        mh = self._message_history
        return json.dumps({
            "history": history,
            "total_messages": len(mh.get("messages", {})) if mh else 0,
            "current_step": self._step,
            "compaction_count": mh.get("compaction_count", 0) if mh else 0,
        })

    # ── check_budget ───────────────────────────────────────────────────────

    def _exec_check_budget(
        self: "IkaExecutionEnvironment", _args: dict
    ) -> str:
        return json.dumps({
            "cost_budget": self._cost_budget,
            "total_cost": round(self._total_cost, 6),
            "remaining": (
                round(self._cost_budget - self._total_cost, 6)
                if self._cost_budget
                else None
            ),
            "total_usage": self._total_usage,
            "current_step": self._step,
        })

    # ── end_execution ──────────────────────────────────────────────────────

    def _exec_end_execution(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        result = args.get("result", "")
        self._end_result = result
        return "Execution terminated successfully."
