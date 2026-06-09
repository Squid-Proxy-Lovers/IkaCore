import pytest

from IkaMem.long_term_memory import LTMemory
from IkaMem.memory_items import LTMemItem
from IkaMem.short_term_memory import STMemory


class RecordingStorage:
    def __init__(self, results=None):
        self.saved = []
        self.search_calls = []
        self.results = results if results is not None else []
        self.reset_called = False

    def save(self, value, metadata):
        self.saved.append((value, metadata))

    def search(self, query, limit, score_threshold):
        self.search_calls.append((query, limit, score_threshold))
        return self.results

    def reset(self):
        self.reset_called = True


class FailingStorage(RecordingStorage):
    def save(self, value, metadata):
        raise RuntimeError("save failed")

    def search(self, query, limit, score_threshold):
        raise RuntimeError("search failed")


def test_short_term_memory_saves_plain_and_mem0_prefixed_values():
    storage = RecordingStorage()
    memory = STMemory(storage=storage)
    memory.agent = "agent-a"

    memory.save("note", {"source": "unit"})

    assert storage.saved == [("note", {"source": "unit"})]

    mem0_storage = RecordingStorage()
    mem0_memory = STMemory(embedder_config={"provider": "mem0"}, storage=mem0_storage)
    mem0_memory.agent = "agent-a"

    mem0_memory.save("insight")

    value, metadata = mem0_storage.saved[-1]
    assert value == "Remember the following insights from Agent run: insight"
    assert metadata == {}


def test_short_term_memory_search_reset_and_error_paths():
    storage = RecordingStorage(results=[{"content": "hit"}])
    memory = STMemory(storage=storage)

    assert memory.search("query", limit=3, score_threshold=0.7) == [{"content": "hit"}]
    assert storage.search_calls == [("query", 3, 0.7)]
    memory.reset()
    assert storage.reset_called is True

    failing = STMemory(storage=FailingStorage())
    with pytest.raises(RuntimeError, match="save failed"):
        failing.save("note")
    with pytest.raises(RuntimeError, match="search failed"):
        failing.search("query")


def test_long_term_memory_saves_structured_item_and_raw_values():
    storage = RecordingStorage()
    memory = LTMemory(storage=storage)
    item = LTMemItem(
        agent="agent-a",
        task="classify issue",
        expected_output="valid",
        datetime="2026-01-01T00:00:00Z",
        quality=0.8,
        metadata={"kind": "triage"},
    )

    memory.save(item, metadata={"source": "unit"})
    memory.save("raw", metadata={"plain": True})

    assert storage.saved[0] == (
        "classify issue",
        {
            "kind": "triage",
            "source": "unit",
            "agent": "agent-a",
            "expected_output": "valid",
            "quality": 0.8,
            "datetime": "2026-01-01T00:00:00Z",
        },
    )
    assert storage.saved[1] == ("raw", {"plain": True})


def test_long_term_memory_search_filter_limit_and_error_paths():
    storage = RecordingStorage(results=["a", "b", "c"])
    memory = LTMemory(storage=storage)
    memory._search_limit = 2
    memory._filter_func = lambda results: results[:1]

    assert memory.search("query") == ["a"]
    assert storage.search_calls == [("query", 2, 0.6)]

    memory._filter_func = lambda _results: "not-list"
    with pytest.raises(TypeError, match="filter function must return list"):
        memory.search("query", limit=4)

    failing = LTMemory(storage=FailingStorage())
    with pytest.raises(RuntimeError, match="save failed"):
        failing.save("raw")
    with pytest.raises(RuntimeError, match="search failed"):
        failing.search("query")


def test_memory_requires_storage_backend_when_provider_not_configured():
    with pytest.raises(ValueError, match="no storage backend"):
        STMemory()
    with pytest.raises(ValueError, match="no storage backend"):
        LTMemory()
