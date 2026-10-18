from types import SimpleNamespace
from unittest.mock import MagicMock

from IkaMem.agent_memory import MemoryOperations


def _minimal_agent(**kwargs):
    return MemoryOperations('A')


def test_short_term_memory_save_and_error_paths():
    agent = _minimal_agent()

    assert agent.save_short_term("data") == "error: short-term memory not initialized"

    agent.short_term_memory = MagicMock()
    assert agent.save_short_term("important data", {"scope": "unit"}) == "saved to short-term memory: important data..."
    agent.short_term_memory.save.assert_called_once_with("important data", {"scope": "unit"})

    agent.short_term_memory.save.side_effect = RuntimeError("disk full")
    assert agent.save_short_term("again") == "error saving to short-term memory: disk full"


def test_long_term_memory_save_validates_payload_and_normalizes_metadata():
    agent = _minimal_agent()

    assert agent.save_long_term({"task": "t", "output": "o"}) == "error: long-term memory not initialized"

    agent.long_term_memory = MagicMock()
    assert agent.save_long_term({"task": "", "output": "o"}) == "error: long_term_save requires 'task' and 'output'"
    assert agent.save_long_term({"task": "t", "output": ""}) == "error: long_term_save requires 'task' and 'output'"

    result = agent.save_long_term({"task": "task name", "output": "expected", "metadata": "raw"})

    assert result == "saved to long-term memory - task: task name..."
    saved_item = agent.long_term_memory.save.call_args.args[0]
    assert saved_item.agent == "A"
    assert saved_item.task == "task name"
    assert saved_item.expected_output == "expected"
    assert saved_item.quality == 1.0
    assert saved_item.metadata == {"metadata": "raw"}

    agent.long_term_memory.save.side_effect = RuntimeError("write failed")
    assert "error saving to long-term memory: write failed" == agent.save_long_term(
        {"task": "task", "output": "expected"}
    )


def test_short_term_search_clamps_inputs_and_normalizes_results():
    agent = _minimal_agent()

    assert agent.search_short_term("query") == {"error": "short-term memory not initialized"}

    agent.short_term_memory = MagicMock()
    assert agent.search_short_term("  ") == {"error": "query is required"}

    agent.short_term_memory.search.return_value = [
        {"data": "stored", "metadata": {"scope": "short"}},
        {"content": "content-only"},
        "plain",
    ]

    result = agent.search_short_term(" query ", limit=100, score_threshold=2.0)

    agent.short_term_memory.search.assert_called_once_with(query="query", limit=50, score_threshold=1.0)
    assert result == {
        "query": "query",
        "limit": 50,
        "score_threshold": 1.0,
        "count": 3,
        "results": [
            {"data": "stored", "metadata": {"scope": "short"}},
            {"data": "content-only", "metadata": {}},
            {"data": "plain", "metadata": {}},
        ],
    }

    agent.short_term_memory.search.side_effect = RuntimeError("search failed")
    assert agent.search_short_term("query") == {"error": "error searching short-term memory: search failed"}


def test_long_term_search_filters_and_normalizes_result_shapes():
    agent = _minimal_agent()

    assert agent.search_long_term("query") == {"error": "long-term memory not initialized"}

    agent.long_term_memory = MagicMock()
    assert agent.search_long_term("") == {"error": "query is required"}

    object_result = SimpleNamespace(task="task", expected_output="output", metadata={"kind": "object"})
    agent.long_term_memory.search.return_value = [
        object_result,
        {"task": "dict task", "expected_output": "dict output", "metadata": {"kind": "dict"}},
        {"content": "content task", "output": "content output"},
        "plain",
    ]

    result = agent.search_long_term(
        " query ",
        limit=0,
        score_threshold=-1.0,
        filter_func=lambda results: results[:3],
    )

    agent.long_term_memory.search.assert_called_once_with(query="query", limit=5, score_threshold=0.0)
    assert result == {
        "query": "query",
        "limit": 5,
        "score_threshold": 0.0,
        "count": 3,
        "results": [
            {"task": "task", "output": "output", "metadata": {"kind": "object"}},
            {"task": "dict task", "output": "dict output", "metadata": {"kind": "dict"}},
            {"task": "content task", "output": "content output", "metadata": {}},
        ],
    }

    agent.long_term_memory.search.side_effect = RuntimeError("search failed")
    assert agent.search_long_term("query") == {"error": "error searching long-term memory: search failed"}


def test_memory_integration_uses_ordinary_tools_and_stage_permissions():
    import json

    from IkaCore import IkaBaseAgent, IkaStage
    from IkaMem.agent_tools import MemoryTools

    backend = MagicMock()
    backend.search.return_value = [{'data': 'note', 'metadata': {}}]
    integration = MemoryTools('A', short_term=backend)
    tools = integration.tools(access={'short_term_save': False})
    assert [tool.name for tool in tools] == ['short_term_search']
    stage = IkaStage('review', 'remember notes', tools)
    agent = IkaBaseAgent('A', 'description', 'task', model_id='gpt-4o', api_key='key', Stages=[stage])
    exposed = {tool.name for tool in agent.build_stage(stage)}
    assert 'short_term_search' in exposed and 'short_term_save' not in exposed
    executors = agent.build_tool_executors(stage.tools, stage=stage)
    result = json.loads(executors['short_term_search']({'query': 'note', 'limit': 'bad', 'score_threshold': True}))
    assert result['count'] == 1
    assert tools[0].replay_policy == 'allow'
    save = integration.tools()[0]
    assert save.replay_policy == 'deny' and save.side_effect_type == 'write'
    assert save.execute({'data': 'remember'}) == 'saved to short-term memory: remember...'
    agent.shutdown()
