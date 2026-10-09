import asyncio
from unittest.mock import AsyncMock, Mock

from IkaModel import runtime_options
from IkaModel.base import BareBoneModel
from IkaModel.chat_interface import chat_request
from IkaModel.execution_hooks import ExecutionHooks, execution_hooks


def test_default_compaction_boundary_stays_at_eighty_percent(monkeypatch):
    model = BareBoneModel("gpt-4o", "test", "system", [])
    model.context_budget = 100
    summarize = Mock()
    monkeypatch.setattr(chat_request, "summarise_message_history", summarize)
    monkeypatch.setattr(chat_request, "get_total_tokens", lambda history: 80)
    chat_request._summarize_sync_if_near_budget(model, {}, None)
    summarize.assert_not_called()
    monkeypatch.setattr(chat_request, "get_total_tokens", lambda history: 81)
    chat_request._summarize_sync_if_near_budget(model, {}, None)
    summarize.assert_called_once()


def test_opt_in_compaction_threshold_and_snapshot_boundaries_sync_and_async(monkeypatch):
    model = BareBoneModel("gpt-4o", "test", "system", [])
    model.context_budget = 100
    sync = Mock()
    asynchronous = AsyncMock()
    events = []
    hooks = ExecutionHooks(lambda label, payload: events.append(label), lambda: {}, lambda *args: "frame", lambda *args: None)
    monkeypatch.setattr(chat_request, "get_total_tokens", lambda history: 65)
    monkeypatch.setattr(chat_request, "summarise_message_history", sync)
    monkeypatch.setattr(chat_request, "async_summarise_message_history", asynchronous)
    with runtime_options(compact_at_fraction=0.6), execution_hooks(hooks):
        chat_request._summarize_sync_if_near_budget(model, {}, None)
        asyncio.run(chat_request._summarize_async_if_near_budget(model, {}, None))
    assert events == ["pre_compaction", "post_compaction"] * 2
    sync.assert_called_once()
    asynchronous.assert_awaited_once()
    events.clear()
    with execution_hooks(hooks):
        chat_request._summarize_sync_if_near_budget(model, {}, None)
    assert not events


def test_context_usage_is_read_only_and_matches_history_token_accounting():
    import copy

    from IkaModel.summarization import get_context_usage
    history = {'system': {'tokens': 10}, 'first_input': {'tokens': 20}, 'summary': {'tokens': 30},
               'messages': {'one': {'tokens': 5}}}
    before = copy.deepcopy(history)
    usage = get_context_usage(history, 'gpt-4o', context_budget=100)
    assert usage == {'token_count': 65, 'max_tokens': 100, 'usage_ratio': 0.65, 'warning_level': 'warning'}
    assert history == before
