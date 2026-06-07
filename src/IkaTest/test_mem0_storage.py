import pytest

from IkaMem.storage import mem0_storage


class DummyMemoryClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.project_updates = []
        self.add_calls = []
        self.search_calls = []
        self.reset_called = False
        DummyMemoryClient.instances.append(self)

    def update_project(self, **kwargs):
        self.project_updates.append(kwargs)

    def add(self, conversations, **params):
        self.add_calls.append((conversations, params))

    def search(self, **params):
        self.search_calls.append(params)
        return {"results": [{"memory": "cloud hit", "score": 0.9}]}

    def reset(self):
        self.reset_called = True


class DummyMemory:
    instances = []

    def __init__(self):
        self.add_calls = []
        self.search_calls = []
        self.reset_called = False
        DummyMemory.instances.append(self)

    @classmethod
    def from_config(cls, config):
        instance = cls()
        instance.config = config
        return instance

    def add(self, conversations, **params):
        self.add_calls.append((conversations, params))

    def search(self, **params):
        self.search_calls.append(params)
        return {"results": [{"memory": "local hit"}]}

    def reset(self):
        self.reset_called = True


@pytest.fixture(autouse=True)
def fake_mem0_backend(monkeypatch):
    DummyMemoryClient.instances.clear()
    DummyMemory.instances.clear()
    monkeypatch.setattr(mem0_storage, "_MEM0_AVAILABLE", True)
    monkeypatch.setattr(mem0_storage, "MemoryClient", DummyMemoryClient)
    monkeypatch.setattr(mem0_storage, "Memory", DummyMemory)


def test_mem0_store_validates_memory_type():
    with pytest.raises(ValueError, match="memory_type"):
        mem0_storage.Mem0Store("invalid")


def test_mem0_store_rejects_missing_or_incomplete_mem0_backend(monkeypatch):
    monkeypatch.setattr(mem0_storage, "_MEM0_AVAILABLE", False)
    with pytest.raises(ImportError, match="optional 'mem0' package"):
        mem0_storage.Mem0Store("short_term")

    monkeypatch.setattr(mem0_storage, "_MEM0_AVAILABLE", True)
    monkeypatch.setattr(mem0_storage, "Memory", None)
    with pytest.raises(ImportError, match="complete mem0 installation"):
        mem0_storage.Mem0Store("short_term")


def test_mem0_store_cloud_client_save_search_and_reset():
    store = mem0_storage.Mem0Store(
        "short_term",
        config={
            "api_key": "key",
            "org_id": "org",
            "project_id": "project",
            "user_id": "user",
            "agent_id": "agent",
            "run_id": "run",
            "infer": False,
            "includes": ["fact"],
            "excludes": ["noise"],
            "custom_categories": [{"name": "fact"}],
        },
    )

    client = DummyMemoryClient.instances[-1]
    assert client.kwargs == {"api_key": "key", "org_id": "org", "project_id": "project"}
    assert client.project_updates == [{"custom_categories": [{"name": "fact"}]}]

    store.save("remember this", {"source": "test"})
    conversations, params = client.add_calls[-1]
    assert conversations == [{"role": "assistant", "content": "remember this"}]
    assert params["metadata"] == {"type": "short_term", "source": "test"}
    assert params["infer"] is False
    assert params["includes"] == ["fact"]
    assert params["excludes"] == ["noise"]
    assert params["run_id"] == "run"
    assert params["user_id"] == "user"
    assert params["agent_id"] == "agent"
    assert params["version"] == "v2"
    assert params["output_format"] == "v1.1"

    results = store.search("query", limit=2, score_threshold=0.7)
    search_params = client.search_calls[-1]
    assert results == [{"memory": "cloud hit", "score": 0.9, "content": "cloud hit"}]
    assert search_params["query"] == "query"
    assert search_params["limit"] == 2
    assert search_params["metadata"] == {"type": "short_term"}
    assert search_params["run_id"] == "run"
    assert search_params["threshold"] == 0.7
    assert search_params["filters"] == {"AND": [{"run_id": "run"}]}

    store.reset()
    assert client.reset_called is True


def test_mem0_store_local_memory_strips_cloud_only_search_params():
    store = mem0_storage.Mem0Store(
        "long_term",
        config={
            "local_mem0_config": {"vector_store": {"provider": "inmemory"}},
            "user_id": "user",
            "agent_id": "agent",
        },
    )

    local_memory = DummyMemory.instances[-1]
    assert local_memory.config == {"vector_store": {"provider": "inmemory"}}

    store.save("local memory", {"scope": "unit"})
    _, save_params = local_memory.add_calls[-1]
    assert save_params["metadata"] == {"type": "long_term", "scope": "unit"}
    assert save_params["user_id"] == "user"
    assert save_params["agent_id"] == "agent"
    assert "includes" not in save_params
    assert "output_format" not in save_params

    results = store.search("query")
    search_params = local_memory.search_calls[-1]
    assert results == [{"memory": "local hit", "content": "local hit"}]
    assert search_params["query"] == "query"
    assert search_params["limit"] == 5
    assert search_params["filters"] == {"OR": [{"user_id": "user"}, {"agent_id": "agent"}]}
    assert search_params["threshold"] == 0.6
    assert "metadata" not in search_params
    assert "version" not in search_params
    assert "output_format" not in search_params
    assert "run_id" not in search_params
