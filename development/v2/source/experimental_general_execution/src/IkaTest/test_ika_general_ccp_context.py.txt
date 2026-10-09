import asyncio
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from IkaGeneral.context_pack_ops import ContextPackMixin
from IkaGeneral.execution_env import IkaExecutionEnvironment
from IkaGeneral.models import CcpContextConfig, ContextPack
from IkaGeneral.models import SubAgentDefinition


class FakeCcpStore:
    def __init__(self):
        now = datetime(2026, 1, 1, 12, 0, 0)
        self.book_name = "context_packs"
        self.remote = {
            "remote-pack": ContextPack(
                name="remote-pack",
                content="remote context",
                book_name="research",
                tags=["remote"],
                source="ccp",
                created_at=now,
                updated_at=now,
                size_tokens=3,
            )
        }
        self.upserted: list[str] = []

    def list_pack_summaries(self, book_name=None):
        return [
            self._summary(pack)
            for pack in self.remote.values()
            if book_name is None or pack.book_name == book_name
        ]

    def summary_to_pack(self, summary, existing_content=""):
        pack = self.remote[summary["name"]]
        return ContextPack(
            name=pack.name,
            content=existing_content,
            book_name=pack.book_name,
            tags=list(pack.tags),
            source=pack.source,
            created_at=pack.created_at,
            updated_at=pack.updated_at,
            size_tokens=pack.size_tokens,
        )

    def upsert_pack(self, pack: ContextPack):
        stored = deepcopy(pack)
        stored.source = "ccp"
        self.remote[pack.name] = stored
        self.upserted.append(pack.name)
        return deepcopy(stored)

    def get_pack(self, name: str, book_name=None):
        if name not in self.remote:
            raise KeyError(name)
        if book_name is not None and self.remote[name].book_name != book_name:
            raise KeyError(name)
        return deepcopy(self.remote[name])

    def list_pack_names(self):
        return sorted(self.remote)

    def search_packs(self, query: str, book_name=None):
        query_lower = query.lower()
        return [
            {
                "name": pack.name,
                "book": pack.book_name,
                "tags": pack.tags,
                "source": pack.source,
            }
            for pack in self.remote.values()
            if (book_name is None or pack.book_name == book_name)
            and (query_lower in pack.name.lower() or query_lower in pack.content.lower())
        ]

    def _summary(self, pack: ContextPack):
        return {
            "name": pack.name,
            "labels": pack.tags,
            "book_name": pack.book_name,
            "description": json.dumps(
                {
                    "source": pack.source,
                    "created_at": pack.created_at.isoformat(),
                    "updated_at": pack.updated_at.isoformat(),
                    "size_tokens": pack.size_tokens,
                }
            ),
        }


class RecordingCcpStore:
    def __init__(self, config):
        self.config = config
        self.book_name = config.book_name

    def list_pack_summaries(self, book_name=None):
        return []

    def upsert_pack(self, pack: ContextPack):
        return deepcopy(pack)

    def summary_to_pack(self, summary, existing_content=""):
        raise AssertionError("summary_to_pack should not be called in this test")


class FakeEnv(ContextPackMixin):
    def __init__(self, store=None):
        self._context_lock = asyncio.Lock()
        self._context_pack_registry = {}
        self._loaded_packs = []
        self._messages = []
        self._ccp_context_store = store
        self._plan_pack_write_fingerprints = {}

    def _root_plan_pack_fingerprint_key(self, _name: str, _book: str):
        return None

    def _compute_plan_pack_fingerprint(self, content: str) -> str:
        return content


def test_sync_context_packs_from_ccp_merges_remote_and_seeds_local():
    env = IkaExecutionEnvironment(
        model_id="deepseek-chat",
        api_key="k",
        initial_prompt="task",
        context_packs=[
            ContextPack(
                name="seed-pack",
                content="seed content",
                tags=["seed"],
                source="agent",
                size_tokens=3,
            )
        ],
    )
    env._ccp_context_store = FakeCcpStore()
    env._sync_context_packs_from_ccp()

    assert "research::remote-pack" in env._context_pack_registry
    assert "seed-pack" in env._ccp_context_store.remote
    assert env._ccp_context_store.upserted == ["seed-pack"]


def test_create_context_pack_persists_to_ccp():
    store = FakeCcpStore()
    env = FakeEnv(store)

    result = json.loads(
        asyncio.run(
            env._exec_create_context_pack(
                {"name": "notes", "book": "plans", "content": "important context", "tags": "alpha,beta"}
            )
        )
    )

    assert result["status"] == "created"
    assert "notes" in store.remote
    assert store.remote["notes"].book_name == "plans"
    assert env._context_pack_registry["plans::notes"].source == "ccp"


def test_load_context_pack_fetches_from_ccp():
    store = FakeCcpStore()
    env = FakeEnv(store)

    result = json.loads(env._exec_load_context_pack({"name": "remote-pack", "book": "research"}))

    assert result["status"] == "loaded"
    assert env._context_pack_registry["research::remote-pack"].content == "remote context"
    assert env._messages[-1]["content"].endswith("remote context")


def test_list_context_packs_uses_ccp_backend():
    store = FakeCcpStore()
    env = FakeEnv(store)

    result = json.loads(env._exec_list_context_packs({}))

    assert result["backend"] == "ccp"
    assert result["total"] == 1
    assert result["packs"][0]["name"] == "remote-pack"
    assert result["packs"][0]["book"] == "research"


def test_search_context_packs_uses_ccp_backend():
    store = FakeCcpStore()
    env = FakeEnv(store)

    result = json.loads(env._exec_search_context_packs({"query": "remote", "book": "research"}))

    assert result["backend"] == "ccp"
    assert result["total"] == 1
    assert result["matches"][0]["name"] == "remote-pack"


def test_duplicate_create_context_pack_replaces_existing_entry():
    env = FakeEnv()
    first = json.loads(
        asyncio.run(
            env._exec_create_context_pack(
                {"name": "plan", "book": "plans", "content": "initial", "tags": "alpha"}
            )
        )
    )
    result = json.loads(
        asyncio.run(
            env._exec_create_context_pack(
                {"name": "plan", "book": "plans", "content": "updated", "tags": "alpha,beta"}
            )
        )
    )

    assert first["status"] == "created"
    assert result["status"] == "recreated"
    assert result["replaced_existing"] is True
    assert env._context_pack_registry["plans::plan"].content == "updated"
    assert env._context_pack_registry["plans::plan"].tags == ["alpha", "beta"]


def test_delete_context_pack_removes_entry():
    env = FakeEnv()
    asyncio.run(
        env._exec_create_context_pack(
            {"name": "plan", "book": "plans", "content": "initial", "tags": "alpha"}
        )
    )

    result = json.loads(
        asyncio.run(
            env._exec_delete_context_pack(
                {"name": "plan", "book": "plans"}
            )
        )
    )

    assert result["status"] == "deleted"
    assert "plans::plan" not in env._context_pack_registry


def test_root_plan_pack_duplicate_write_is_blocked_when_content_is_materially_identical():
    env = IkaExecutionEnvironment(
        model_id="deepseek-chat",
        api_key="k",
        initial_prompt="task",
    )

    first = json.loads(
        asyncio.run(
            env._exec_create_context_pack(
                {"name": "master_plan", "book": "plans", "content": "phase 1\nphase 2", "tags": "plan"}
            )
        )
    )
    second = json.loads(
        asyncio.run(
            env._exec_create_context_pack(
                {"name": "master_plan", "book": "plans", "content": "phase 1\nphase 2", "tags": "plan"}
            )
        )
    )

    assert first["status"] == "created"
    assert second["status"] == "blocked_duplicate_create"
    assert "stop_repeating_tool_call" in second


def test_root_plan_pack_rewrite_is_allowed_when_content_changes():
    env = IkaExecutionEnvironment(
        model_id="deepseek-chat",
        api_key="k",
        initial_prompt="task",
    )

    asyncio.run(
        env._exec_create_context_pack(
            {"name": "master_plan", "book": "plans", "content": "phase 1", "tags": "plan"}
        )
    )
    second = json.loads(
        asyncio.run(
            env._exec_create_context_pack(
                {"name": "master_plan", "book": "plans", "content": "phase 1\nphase 2", "tags": "plan"}
            )
        )
    )

    assert second["status"] == "recreated"
    assert second["replaced_existing"] is True


def test_subagent_execution_is_blocked_beyond_depth_two():
    env = IkaExecutionEnvironment(
        model_id="deepseek-chat",
        api_key="k",
        initial_prompt="task",
        _execution_name="lvl2",
        _agent_hierarchy=["root", "lvl1", "lvl2"],
    )
    env._subagent_defs["child"] = SubAgentDefinition(
        name="child",
        description="worker",
        prompt="do work",
    )

    result = json.loads(asyncio.run(env._exec_run_subagent({"name": "child"})))

    assert result["status"] == "blocked_depth_limit"
    assert result["max_depth"] == 3


def test_branch_execution_is_blocked_beyond_depth_two():
    env = IkaExecutionEnvironment(
        model_id="deepseek-chat",
        api_key="k",
        initial_prompt="task",
        _execution_name="lvl2",
        _agent_hierarchy=["root", "lvl1", "lvl2"],
    )

    result = json.loads(
        asyncio.run(
            env._exec_branch_execution_path({"prompts": json.dumps(["a", "b"])})
        )
    )

    assert result["status"] == "blocked_depth_limit"
    assert result["max_depth"] == 3


def test_execution_env_defaults_to_no_timeout():
    env = IkaExecutionEnvironment(
        model_id="deepseek-chat",
        api_key="k",
        initial_prompt="task",
    )

    assert env._request_timeout is None


def test_root_env_uses_execution_scoped_ccp_shelf():
    with patch("IkaGeneral.execution_env.CcpContextStore", RecordingCcpStore):
        env = IkaExecutionEnvironment(
            model_id="deepseek-chat",
            api_key="k",
            initial_prompt="task",
            ccp_context=CcpContextConfig(
                session="spl-dev-lib",
                shelf_name="ika_general_context",
                book_name="plans",
            ),
        )
        env._prepare_ccp_context_for_run()

    assert env._ccp_context_config is not None
    assert env._ccp_context_config.shelf_name.startswith("ika_general_context-")
    assert env._ccp_context_config.shelf_name != "ika_general_context"
    assert env._ccp_context_store is not None
    assert env._ccp_context_store.config.shelf_name == env._ccp_context_config.shelf_name


def test_child_env_inherits_parent_execution_scoped_ccp_shelf():
    with patch("IkaGeneral.execution_env.CcpContextStore", RecordingCcpStore):
        parent = IkaExecutionEnvironment(
            model_id="deepseek-chat",
            api_key="k",
            initial_prompt="parent task",
            ccp_context=CcpContextConfig(
                session="spl-dev-lib",
                shelf_name="ika_general_context",
                book_name="plans",
            ),
        )
        parent._prepare_ccp_context_for_run()
        child = IkaExecutionEnvironment(
            model_id="deepseek-chat",
            api_key="k",
            initial_prompt="child task",
            ccp_context=parent._ccp_context_config,
            _parent_env=parent,
        )

    assert parent._ccp_context_config is not None
    assert child._ccp_context_config is not None
    assert child._ccp_context_config.shelf_name == parent._ccp_context_config.shelf_name
    assert child._ccp_context_store is not None
    assert child._ccp_context_store.config.shelf_name == parent._ccp_context_config.shelf_name


def test_root_env_generates_new_shelf_for_each_execution():
    with patch("IkaGeneral.execution_env.CcpContextStore", RecordingCcpStore):
        env = IkaExecutionEnvironment(
            model_id="deepseek-chat",
            api_key="k",
            initial_prompt="task",
            ccp_context=CcpContextConfig(
                session="spl-dev-lib",
                shelf_name="ika_general_context",
                book_name="plans",
            ),
        )
        with patch("IkaGeneral.execution_env._make_execution_scoped_shelf_name", side_effect=["ika_general_context-run1", "ika_general_context-run2"]):
            env._prepare_ccp_context_for_run()
            first_shelf = env._ccp_context_config.shelf_name
            env._prepare_ccp_context_for_run()
            second_shelf = env._ccp_context_config.shelf_name

    assert first_shelf == "ika_general_context-run1"
    assert second_shelf == "ika_general_context-run2"
