"""Offline contracts for optional memory backends, coercion and access controls."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from IkaMem import LTMemory, STMemory
from IkaMem.agent_memory import MemoryOperations
from IkaMem.agent_tools import MemoryTools, _first_string_arg, _float_arg, _int_arg, memory_access_defaults
from IkaMem.memory import Memory
from IkaMem.storage import mem0_storage as mem0
from IkaTest.test_mem0_storage import DummyMemory, DummyMemoryClient


def test_base_memory_delegates_and_exposes_agent_task():
    storage = Mock()
    storage.search.return_value = ["found"]
    memory = Memory(storage)
    memory.task, memory.agent = "task", "agent"
    assert (memory.task, memory.agent) == ("task", "agent")
    memory.save("value")
    storage.save.assert_called_once_with("value", {})
    assert memory.search("query", 3, .4) == ["found"]
    memory.reset()
    storage.reset.assert_called_once()


@pytest.mark.parametrize("kind", [LTMemory, STMemory])
def test_mem0_configuration_is_forwarded_without_a_live_backend(monkeypatch, kind):
    backend = Mock()
    target = "IkaMem.long_term_memory.Mem0Store" if kind is LTMemory else "IkaMem.short_term_memory.Mem0Store"
    factory = Mock(return_value=backend)
    monkeypatch.setattr(target, factory)
    config = {"provider": "mem0", "config": {"user_id": "fixture"}}
    memory = kind(embedder_config=config)
    factory.assert_called_once_with(memory_type="long_term" if kind is LTMemory else "short_term", config=config["config"])
    assert memory.storage is backend


def test_memory_tool_coercion_permissions_and_mixed_results():
    assert _first_string_arg({"input": 7}, "input") == "7"
    assert _first_string_arg({}, "input") == ""
    assert _int_arg({"limit": True}, "limit", 5) == 5
    assert _float_arg({"score": "bad"}, "score", .6) == .6
    assert _float_arg({"score": False}, "score", .6) == .6
    short, long = Mock(), Mock()
    short.search.return_value = [{"data": 7, "metadata": "bad"}, "plain"]
    long.search.return_value = [SimpleNamespace(task="task", expected_output="done", metadata="bad"), {"task": 7, "output": 8, "metadata": []}, "other"]
    tools = MemoryTools("memory", short_term=short, long_term=long)
    assert tools.tools(access=dict.fromkeys(memory_access_defaults(True), False)) == []
    result = tools.operations.search_long_term(" q ")
    assert result["results"] == [{"task": "task", "output": "done", "metadata": {}}, {"task": "7", "output": "8", "metadata": {}}, {"task": "other", "output": "", "metadata": {}}]
    assert tools.operations.search_short_term("q")["results"][0] == {"data": "7", "metadata": {}}
    permissions = {"short_term_save": True, "short_term_search": False, "long_term_save": False, "long_term_search": False}
    only = tools.tools(access=permissions)
    assert [t.name for t in only] == ["short_term_save"]
    assert tools._short_term_memory_executors({}) == {}
    assert tools._long_term_memory_executors({}, None) == {}


def test_memory_initialization_and_no_filter_paths(monkeypatch):
    short = Mock()
    monkeypatch.setattr("IkaMem.agent_memory.STMemory", Mock(return_value=short))
    operations = MemoryOperations("fixture")
    operations.init_short_term_memory({"provider": "fake"})
    assert operations.short_term_memory is short and short.agent == "fixture"
    long = Mock()
    long.search.return_value = [{"content": "task", "expected_output": "result"}]
    operations.long_term_memory = long
    result = operations.search_long_term("q")
    assert result["count"] == 1 and result["results"][0]["output"] == "result"
    assert "expected list" in operations.search_long_term("q", filter_func=lambda x: ())["error"]


@pytest.mark.parametrize("config,expected", [({}, {}), ({"user_id": "u"}, {"AND": [{"user_id": "u"}]}), ({"agent_id": "a"}, {"AND": [{"agent_id": "a"}]}), ({"user_id": "u", "agent_id": "a"}, {"OR": [{"user_id": "u"}, {"agent_id": "a"}]})])
def test_mem0_identity_filters_and_default_local_initialization(monkeypatch, config, expected):
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.setattr(mem0, "_MEM0_AVAILABLE", True)
    monkeypatch.setattr(mem0, "Memory", DummyMemory)
    monkeypatch.setattr(mem0, "MemoryClient", DummyMemoryClient)
    store = mem0.Mem0Store("long_term", config)
    assert isinstance(store.memory, DummyMemory)
    assert store._create_filter_for_search() == expected
    store.save("value", {})
    assert store.search("query")[0]["content"] == "local hit"
    store.memory = None
    store.reset()


def test_mem0_optional_import_shapes_and_cloud_without_project(monkeypatch):
    monkeypatch.setattr(mem0.importlib, "import_module", lambda name: SimpleNamespace(Memory=DummyMemory, MemoryClient=DummyMemoryClient))
    assert mem0._load_mem0_classes() == (DummyMemory, DummyMemoryClient)
    monkeypatch.setattr(mem0.importlib, "import_module", lambda name: SimpleNamespace(Memory=object()))
    assert mem0._load_mem0_classes() == (None, None)
    monkeypatch.setattr(mem0, "_MEM0_AVAILABLE", True)
    monkeypatch.setattr(mem0, "Memory", DummyMemory)
    monkeypatch.setattr(mem0, "MemoryClient", DummyMemoryClient)
    store = mem0.Mem0Store("long_term", {"api_key": "test"})
    assert store.memory.kwargs == {"api_key": "test"}
    assert store.memory.project_updates == []
    store.save("value", {})
    assert "run_id" not in store.memory.add_calls[-1][1]
    store.search("query")
    assert "user_id" not in store.memory.search_calls[-1]


def test_storage_package_optional_import_failure_is_recoverable(monkeypatch):
    import builtins

    import IkaMem.storage as storage
    original = builtins.__import__
    def importing(name, *a, **kw):
        if name == "IkaMem.storage.mem0_storage":
            raise ImportError("optional dependency")
        return original(name, *a, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", importing)
        importlib.reload(storage)
        assert storage.Mem0Store is None
    importlib.reload(storage)
    assert storage.Mem0Store is mem0.Mem0Store
