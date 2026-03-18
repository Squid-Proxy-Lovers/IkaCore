"""IkaGeneral: Continuous General Agent Execution Environment.

Builds on IkaCore infrastructure to provide a higher-level execution system
with context packs, sub-agents, dynamic tools, branching, and checkpoints.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from IkaCore.tools import IkaTools
from IkaModel.base import BareBoneModel, AgentEndException
from IkaModel.chat_interface.chat_interface import async_chat

from .ccp_context_store import CcpContextStore
from .execution_trace import ExecutionTraceWriter
from .models import (
    CcpContextConfig,
    ContextPack,
    DynamicToolDefinition,
    ExecutionCheckpoint,
    SubAgentDefinition,
)
from .helpers import ENV_PREAMBLE, convert_tools, resolve_api_url

# Mixins — each file contributes executor methods to the class
from .context_pack_ops import ContextPackMixin
from .subagent_ops import SubAgentMixin
from .dynamic_tool_ops import DynamicToolMixin
from .execution_control_ops import ExecutionControlMixin
from .meta_tool_registry import MetaToolRegistryMixin

LOG = logging.getLogger(__name__)


def _make_execution_scoped_shelf_name(base_shelf_name: str) -> str:
    base = (base_shelf_name or "ika_general_context").strip() or "ika_general_context"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}-{stamp}"


class IkaExecutionEnvironment(
    ContextPackMixin,
    SubAgentMixin,
    DynamicToolMixin,
    ExecutionControlMixin,
    MetaToolRegistryMixin,
):
    """Continuous, autonomous execution environment with meta-tool capabilities.

    Use the builder pattern to construct::

        env = (IkaExecutionEnvironment.builder()
            .model("claude-sonnet-4-6")
            .api_key("sk-...")
            .system_prompt("You are a research agent...")
            .initial_prompt("Research X and produce a report...")
            .tools([web_search, file_read])
            .cost_budget(5.00)
            .auto_checkpoint_interval(10)
            .build())

        result = await env.run()
    """

    # ── Construction ───────────────────────────────────────────────────────

    def __init__(
        self,
        model_id: str,
        api_key: str,
        api_url: Optional[str] = None,
        system_prompt: str = "",
        initial_prompt: str = "",
        tools: Optional[List[IkaTools]] = None,
        context_packs: Optional[List[ContextPack]] = None,
        ccp_context: Optional[CcpContextConfig] = None,
        cost_budget: Optional[float] = None,
        auto_checkpoint_interval: int = 0,
        max_tokens: int = 50000,
        temperature: float = 0.0,
        # Internal — used when creating sub-envs / branches
        _execution_name: Optional[str] = None,
        _agent_hierarchy: Optional[List[str]] = None,
        _trace_writer: Optional[ExecutionTraceWriter] = None,
        _parent_env: Optional["IkaExecutionEnvironment"] = None,
        _shared_context_registry: Optional[Dict[str, ContextPack]] = None,
        _shared_lock: Optional[asyncio.Lock] = None,
        _request_timeout: Optional[float] = None,
    ):
        # LLM config
        self._model_id = model_id
        self._api_key = api_key
        self._api_url = api_url or resolve_api_url(model_id)
        self._system_prompt = system_prompt
        self._initial_prompt = initial_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._request_timeout = _request_timeout

        # User-provided tools
        self._user_tools: List[IkaTools] = list(tools) if tools else []

        # Budget / checkpointing
        self._cost_budget = cost_budget
        self._auto_checkpoint_interval = auto_checkpoint_interval

        # Context packs — shared registry between parent / sub-agents
        self._context_pack_registry: Dict[str, ContextPack] = (
            _shared_context_registry if _shared_context_registry is not None else {}
        )
        self._context_lock: asyncio.Lock = _shared_lock or asyncio.Lock()
        self._loaded_packs: List[str] = []
        self._seed_context_pack_keys: set[str] = set()
        self._plan_pack_write_fingerprints: Dict[str, str] = {}

        # Pre-load packs supplied at construction
        if context_packs:
            for pack in context_packs:
                key = self._register_context_pack(pack)
                self._loaded_packs.append(key)
                self._seed_context_pack_keys.add(key)

        # Global registries (persist across checkpoint resets)
        self._subagent_defs: Dict[str, SubAgentDefinition] = {}
        self._dynamic_tools: Dict[str, DynamicToolDefinition] = {}

        # Checkpoints
        self._checkpoints: List[ExecutionCheckpoint] = []

        # Execution state
        self._step: int = 0
        self._end_result: Any = None
        self._pending_reset: Optional[ExecutionCheckpoint] = None
        self._total_cost: float = 0.0
        self._total_usage: Dict[str, int] = {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        }

        # User interruption
        self._interrupt_event = asyncio.Event()
        self._interrupt_message: str = ""

        # Running sub-agents (enforces branch constraint)
        self._running_subagents: set = set()

        # Parent reference
        self._parent_env = _parent_env
        self._execution_name = _execution_name or ("root" if _parent_env is None else "agent")
        self._agent_hierarchy = list(_agent_hierarchy or [self._execution_name])
        self._ccp_context_template = copy.deepcopy(ccp_context) if ccp_context else None
        self._ccp_context_config: Optional[CcpContextConfig] = None
        self._ccp_context_store: Optional[CcpContextStore] = None
        if self._parent_env is not None:
            self._prepare_ccp_context_for_run()
        self._trace_writer = _trace_writer or (_parent_env._trace_writer if _parent_env else None)

        # Runtime state (populated in _setup)
        self._message_history: Optional[dict] = None
        self._messages: Optional[list] = None
        self._barebone: Optional[BareBoneModel] = None
        self._tool_executors: Dict[str, Callable] = {}

    def _prepare_ccp_context_for_run(self) -> None:
        if self._parent_env is not None and self._parent_env._ccp_context_config is not None:
            self._ccp_context_config = copy.deepcopy(self._parent_env._ccp_context_config)
        elif self._ccp_context_template is not None:
            self._ccp_context_config = copy.deepcopy(self._ccp_context_template)
            self._ccp_context_config.shelf_name = _make_execution_scoped_shelf_name(
                self._ccp_context_config.shelf_name
            )
        else:
            self._ccp_context_config = None

        self._ccp_context_store = (
            CcpContextStore(self._ccp_context_config)
            if self._ccp_context_config else None
        )

    # ── Builder ────────────────────────────────────────────────────────────

    @classmethod
    def builder(cls) -> "IkaExecutionEnvironment.Builder":
        """Return a new Builder instance."""
        return cls.Builder()

    class Builder:
        """Fluent builder for :class:`IkaExecutionEnvironment`."""

        def __init__(self) -> None:
            self._cfg: Dict[str, Any] = {}

        def model(self, model_id: str, api_key: Optional[str] = None, api_url: Optional[str] = None):
            self._cfg["model_id"] = model_id
            if api_key is not None:
                self._cfg["api_key"] = api_key
            if api_url is not None:
                self._cfg["api_url"] = api_url
            return self

        def api_key(self, key: str):
            self._cfg["api_key"] = key
            return self

        def system_prompt(self, prompt: str):
            self._cfg["system_prompt"] = prompt
            return self

        def initial_prompt(self, prompt: str):
            self._cfg["initial_prompt"] = prompt
            return self

        def tools(self, tools: List[IkaTools]):
            self._cfg["tools"] = tools
            return self

        def context_packs(self, packs: List[ContextPack]):
            self._cfg["context_packs"] = packs
            return self

        def ccp_context(
            self,
            session: str,
            *,
            shelf_name: str = "ika_general_context",
            book_name: str = "context_packs",
            shelf_description: str = "IkaGeneral context pack storage",
            book_description: str = "Persisted IkaGeneral context packs",
            client_binary: Optional[str] = None,
            client_home: Optional[str] = None,
        ):
            self._cfg["ccp_context"] = CcpContextConfig(
                session=session,
                shelf_name=shelf_name,
                book_name=book_name,
                shelf_description=shelf_description,
                book_description=book_description,
                client_binary=client_binary,
                client_home=client_home,
            )
            return self

        def cost_budget(self, budget: float):
            self._cfg["cost_budget"] = budget
            return self

        def auto_checkpoint_interval(self, interval: int):
            self._cfg["auto_checkpoint_interval"] = interval
            return self

        def max_tokens(self, tokens: int):
            self._cfg["max_tokens"] = tokens
            return self

        def temperature(self, temp: float):
            self._cfg["temperature"] = temp
            return self

        def timeout(self, seconds: Optional[float]):
            self._cfg["_request_timeout"] = seconds
            return self

        def build(self) -> "IkaExecutionEnvironment":
            if "model_id" not in self._cfg:
                raise ValueError("model_id is required — call .model() first")
            if "api_key" not in self._cfg:
                raise ValueError(
                    "api_key is required — call .api_key() or .model(id, api_key=...) first"
                )
            return IkaExecutionEnvironment(**self._cfg)

    @staticmethod
    def _normalize_context_pack_content(content: str) -> str:
        lines = [line.rstrip() for line in str(content).strip().splitlines()]
        return "\n".join(lines)

    def _root_plan_pack_fingerprint_key(self, name: str, book: str) -> Optional[str]:
        if self._parent_env is not None:
            return None
        if self._execution_name != "root":
            return None
        if book.strip() != "plans":
            return None
        clean_name = name.strip()
        if not clean_name:
            return None
        return f"{book.strip()}::{clean_name}"

    def _compute_plan_pack_fingerprint(self, content: str) -> str:
        normalized = self._normalize_context_pack_content(content)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(self) -> Any:
        """Execute the agent loop until ``end_execution`` or budget exhaustion."""
        if self._parent_env is None:
            self._prepare_ccp_context_for_run()

        owns_trace_writer = False
        if self._parent_env is None and self._trace_writer is None:
            self._trace_writer = ExecutionTraceWriter.create_default(Path.cwd() / "logs", root_name=self._execution_name)
            self._trace_writer.install()
            owns_trace_writer = True
        if self._parent_env is None and self._trace_writer is not None:
            self._trace_writer.append_root_text(f"Request: {self._initial_prompt}\n\n")

        self._setup()

        assert self._barebone is not None
        assert self._messages is not None

        try:
            while True:
                # ── Budget check ───────────────────────────────────────────────
                if self._cost_budget is not None and self._total_cost >= self._cost_budget:
                    LOG.warning(
                        "Cost budget exhausted (%.4f / %.4f)",
                        self._total_cost, self._cost_budget,
                    )
                    return {
                        "status": "budget_exhausted",
                        "total_cost": self._total_cost,
                        "total_usage": self._total_usage,
                    }

                # ── User interruption ──────────────────────────────────────────
                if self._interrupt_event.is_set():
                    self._messages.append({
                        "role": "user",
                        "content": f"[User Interruption]: {self._interrupt_message}",
                    })
                    self._interrupt_event.clear()

                # ── LLM + tool execution round ─────────────────────────────────
                try:
                    response = await async_chat(
                        self._barebone,
                        self._messages,
                        self._message_history,
                        self._tool_executors,
                        max_tool_rounds=20,
                        timeout=self._request_timeout,
                    )
                except AgentEndException as e:
                    if self._end_result is not None:
                        return self._end_result
                    return e.response

                # ── Cost tracking ──────────────────────────────────────────────
                cost = response.get("cost") or {}
                if cost:
                    self._total_cost += cost.get("total_cost", 0) or 0
                usage = response.get("usage") or {}
                for k in ("input_tokens", "output_tokens", "total_tokens"):
                    self._total_usage[k] += usage.get(k, 0) or 0

                # ── Check end_execution ────────────────────────────────────────
                if self._end_result is not None:
                    return self._end_result

                # ── Check pending checkpoint reset ─────────────────────────────
                if self._pending_reset is not None:
                    cp = self._pending_reset
                    self._pending_reset = None
                    self._message_history = copy.deepcopy(cp.execution_memory)
                    self._loaded_packs = list(cp.loaded_packs)
                    self._step = cp.step_number
                    self._messages = [
                        {"role": "user",
                         "content": "Execution reset to checkpoint. Continue from here."}
                    ]
                    LOG.info("Reset to checkpoint '%s' at step %d", cp.name, cp.step_number)
                    continue

                # ── Step tracking & auto-checkpoint ────────────────────────────
                executed = response.get("executed_tool_calls", [])
                self._step += len(executed)

                if (
                    self._auto_checkpoint_interval > 0
                    and self._step > 0
                    and self._step % self._auto_checkpoint_interval == 0
                ):
                    self._create_checkpoint(name=None)

                # ── Continuation prompt if no progress ─────────────────────────
                if not executed and not response.get("content"):
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "Continue with your task. Use available tools to make "
                            "progress, or call end_execution when done."
                        ),
                    })
        finally:
            if owns_trace_writer and self._trace_writer is not None:
                self._trace_writer.uninstall()

    def interrupt(self, message: str) -> None:
        """Interrupt the running agent with a user message (thread-safe)."""
        self._interrupt_message = message
        self._interrupt_event.set()

    # ── Setup ──────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Initialize all runtime state for a fresh execution."""
        full_system = ENV_PREAMBLE + (self._system_prompt or "")
        self._sync_context_packs_from_ccp()
        self._plan_pack_write_fingerprints = {}

        # Message history (used by IkaCore for tracking & summarization)
        self._message_history = {
            "system": {"message": full_system, "tokens": 0},
            "first_input": {"message": self._initial_prompt, "tokens": 0},
            "summary": {"message": "", "tokens": 0},
            "messages": {},
            "compaction_count": 0,
            "_context_warning_issued": None,
        }

        # Messages list (actual conversation sent to LLM API)
        self._messages = [{"role": "user", "content": self._initial_prompt}]

        # Inject pre-loaded context packs
        for pack_name in self._loaded_packs:
            pack = self._context_pack_registry.get(pack_name)
            if pack:
                self._messages.append({
                    "role": "user",
                    "content": f"[Context Pack: {pack.book_name}/{pack.name}]\n{pack.content}",
                })

        # Build meta-tools and their executors
        meta_tools, meta_executors = self._build_meta_tools()
        all_ika_tools = self._user_tools + meta_tools

        # Convert IkaTools → AgentTool for the LLM
        agent_tools = convert_tools(all_ika_tools)

        # Build executor map
        self._tool_executors = {}
        for tool in self._user_tools:
            if tool.execute_function:
                self._tool_executors[tool.name] = tool.execute_function
        self._tool_executors.update(meta_executors)

        # Create BareBoneModel
        self._barebone = BareBoneModel(
            model_id=self._model_id,
            api_key=self._api_key,
            api_url=self._api_url,
            system_prompt=full_system,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            agent_name=self._execution_name,
            agent_hierarchy=self._agent_hierarchy,
            suppress_init_output=True,
        )
        self._barebone.agent_tools = agent_tools  # type: ignore[attr-defined]
        self._barebone._current_step = 0  # type: ignore[attr-defined]
        self._barebone._tool_call_counts = {}  # type: ignore[attr-defined]

        # Reset per-run execution state
        self._step = 0
        self._end_result = None
        self._pending_reset = None

    def _sync_context_packs_from_ccp(self) -> None:
        if self._ccp_context_store is None:
            return

        summaries = self._ccp_context_store.list_pack_summaries()
        for summary in summaries:
            name = summary.get("name", "")
            book_name = summary.get("book_name", self._ccp_context_store.book_name)
            key = self._context_pack_key(name, book_name)
            if not name or key in self._context_pack_registry:
                continue
            self._register_context_pack(self._ccp_context_store.summary_to_pack(summary))

        for key in self._seed_context_pack_keys:
            pack = self._context_pack_registry.get(key)
            if pack is None:
                continue
            self._register_context_pack(self._ccp_context_store.upsert_pack(pack))

    # ── Checkpoint helper (used by mixin + auto-checkpoint) ────────────────

    def _create_checkpoint(self, name: Optional[str]) -> ExecutionCheckpoint:
        """Create a checkpoint at the current execution step."""
        cp = ExecutionCheckpoint(
            name=name,
            step_number=self._step,
            execution_memory=copy.deepcopy(self._message_history),
            loaded_packs=list(self._loaded_packs),
            created_at=datetime.now(),
        )
        self._checkpoints.append(cp)
        LOG.info("Checkpoint created: %s", name or f"auto@step{self._step}")
        return cp
