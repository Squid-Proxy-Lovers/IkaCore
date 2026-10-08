"""Mixin providing sub-agent meta-tool executors."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from .helpers import resolve_api_url
from .models import SubAgentDefinition

if TYPE_CHECKING:
    from .execution_env import IkaExecutionEnvironment

LOG = logging.getLogger(__name__)


class SubAgentMixin:
    """Executor methods for sub-agent meta-tools.

    Mixed into :class:`IkaExecutionEnvironment`.
    """

    # ── make ───────────────────────────────────────────────────────────────

    def _exec_make_subagent(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        name = args.get("name", "")
        if name in self._subagent_defs:
            return json.dumps({
                "error": f"Sub-agent '{name}' already exists. "
                "Use edit_subagent to modify."
            })

        tools_str = args.get("tools", "")
        tool_names = (
            [t.strip() for t in tools_str.split(",") if t.strip()]
            if tools_str
            else []
        )

        model_params: dict = {}
        if args.get("model_params"):
            try:
                model_params = json.loads(args["model_params"])
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON in model_params"})

        cp_str = args.get("context_packs", "")
        cp_names = (
            [c.strip() for c in cp_str.split(",") if c.strip()] if cp_str else []
        )

        defn = SubAgentDefinition(
            name=name,
            description=args.get("description", ""),
            prompt=args.get("prompt", ""),
            tools=tool_names,
            model=args.get("model") or None,
            model_params=model_params,
            context_packs=cp_names,
            status="created",
            last_result=None,
        )
        self._subagent_defs[name] = defn
        return json.dumps({"status": "created", "name": name})

    # ── run (async) ────────────────────────────────────────────────────────

    async def _exec_run_subagent(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        # Deferred import to avoid circular reference at module level
        from .execution_env import IkaExecutionEnvironment

        name = args.get("name", "")
        current_depth = len(self._agent_hierarchy)
        max_depth = 3
        if current_depth >= max_depth:
            return json.dumps({
                "status": "blocked_depth_limit",
                "error": (
                    "Sub-agent execution blocked. Maximum chain depth is 3 "
                    "(root -> level1 -> level2)."
                ),
                "current_chain": " -> ".join(self._agent_hierarchy),
                "attempted_subagent": name,
                "max_depth": max_depth,
                "next_required_action": (
                    "Do the work in the current agent, or return findings to the parent instead "
                    "of spawning another subagent."
                ),
            })
        if name not in self._subagent_defs:
            return json.dumps({
                "error": f"Sub-agent '{name}' not found. "
                "Create it first with make_subagent."
            })

        defn = self._subagent_defs[name]

        if name in self._running_subagents:
            return json.dumps({"error": f"Sub-agent '{name}' is already running"})

        defn.status = "running"
        self._running_subagents.add(name)

        try:
            # Resolve user tools available to the sub-agent
            sub_tools = [t for t in self._user_tools if t.name in defn.tools]

            # Pre-load context packs (shared references)
            pre_loaded = []
            for cp_ref in defn.context_packs:
                pack_key = self._resolve_context_pack_key(cp_ref)
                if pack_key and pack_key in self._context_pack_registry:
                    pre_loaded.append(self._context_pack_registry[pack_key])

            # Compute sub-agent's budget share
            sub_budget = None
            if self._cost_budget is not None:
                sub_budget = max(0.0, self._cost_budget - self._total_cost)

            sub_env = IkaExecutionEnvironment(
                model_id=defn.model or self._model_id,
                api_key=self._api_key,
                api_url=(
                    resolve_api_url(defn.model) if defn.model else self._api_url
                ),
                system_prompt=defn.description,
                initial_prompt=defn.prompt,
                tools=sub_tools,
                context_packs=pre_loaded,
                ccp_context=self._ccp_context_config,
                cost_budget=sub_budget,
                max_tokens=defn.model_params.get("max_tokens", self._max_tokens),
                temperature=defn.model_params.get(
                    "temperature", self._temperature
                ),
                _execution_name=defn.name,
                _agent_hierarchy=self._agent_hierarchy + [defn.name],
                _trace_writer=self._trace_writer,
                _parent_env=self,
                _shared_context_registry=self._context_pack_registry,
                _shared_lock=self._context_lock,
            )

            result = await sub_env.run()

            # Propagate cost to parent
            self._total_cost += sub_env._total_cost
            for k in self._total_usage:
                self._total_usage[k] += sub_env._total_usage.get(k, 0)

            defn.status = "completed"
            defn.last_result = result
            return json.dumps({
                "status": "completed",
                "name": name,
                "result": str(result),
            })

        except Exception as e:
            defn.status = "failed"
            defn.last_result = str(e)
            LOG.exception("Sub-agent '%s' failed", name)
            return json.dumps({"status": "failed", "name": name, "error": str(e)})
        finally:
            self._running_subagents.discard(name)

    # ── edit ───────────────────────────────────────────────────────────────

    def _exec_edit_subagent(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        name = args.get("name", "")
        if name not in self._subagent_defs:
            return json.dumps({"error": f"Sub-agent '{name}' not found"})

        defn = self._subagent_defs[name]
        if args.get("description"):
            defn.description = args["description"]
        if args.get("prompt"):
            defn.prompt = args["prompt"]
        if args.get("tools"):
            defn.tools = [
                t.strip() for t in args["tools"].split(",") if t.strip()
            ]
        if args.get("model"):
            defn.model = args["model"]
        if args.get("model_params"):
            try:
                defn.model_params = json.loads(args["model_params"])
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON in model_params"})
        if args.get("context_packs"):
            defn.context_packs = [
                c.strip() for c in args["context_packs"].split(",") if c.strip()
            ]

        return json.dumps({"status": "updated", "name": name})

    # ── list ───────────────────────────────────────────────────────────────

    def _exec_list_subagents(
        self: "IkaExecutionEnvironment", _args: dict
    ) -> str:
        agents = []
        for name, defn in self._subagent_defs.items():
            agents.append({
                "name": name,
                "description": defn.description[:100],
                "model": defn.model or "(inherited)",
                "tools": defn.tools,
                "context_packs": defn.context_packs,
                "status": defn.status,
                "has_result": defn.last_result is not None,
            })
        return json.dumps({"subagents": agents, "total": len(agents)})
