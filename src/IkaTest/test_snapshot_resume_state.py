"""Resumed transcripts, execution positions, and stage budgets are consumed once."""
import json
from copy import deepcopy

import pytest

from IkaCore import IkaBaseAgent, IkaStage, OutputConfig
from IkaTest.agent_config_fixtures import agent_kwargs


def _turn(content='partial', control=None):
    calls = [{'function': {'name': control, 'arguments': json.dumps({'input': content})}}] if control else []
    return {'content': content, 'content_before_tools': content, 'tool_calls': calls,
            'executed_tool_calls': calls, 'usage': {'input_tokens': 1}, 'cost': {}}


@pytest.fixture
def agent_factory(tmp_path, monkeypatch):
    agents = []
    def create(**options):
        defaults = {'maxsteps': 6, 'max_stage_extensions': 0, 'max_step_extensions': 0}
        defaults.update(options)
        agent = IkaBaseAgent('test', 'test', 'task', model_id='gpt-4o', api_key='test',
                            output=OutputConfig(show_usage_level0=False), **agent_kwargs(defaults))
        agent.enable_snapshots(str(tmp_path / f'runtime-{len(agents)}.db'))
        agents.append(agent)
        return agent
    monkeypatch.setattr('IkaCore.agent_policy.run_summarization', lambda *args, **kwargs: 'forced final')
    yield create
    for agent in agents:
        agent.shutdown()


def test_hitl_resume_consumes_saved_transcript_before_entering_next_stage(agent_factory, monkeypatch):
    agent = agent_factory(Stages=[IkaStage('first', 'ask for approval', [], hitl=True),
                                  IkaStage('second', 'write final report', [])])
    seen = []
    def chat(model, messages, history, **kwargs):
        seen.append(deepcopy(messages))
        if len(seen) == 1:
            messages.append({'role': 'assistant', 'content': 'Approve?'})
            return {**_turn('Approve?'), 'interrupted': True, 'status': 'awaiting_user_input',
                    'interrupt_data': {'question': 'Approve?'}}
        return _turn('approved', 'stage_end') if len(seen) == 2 else _turn('done', 'agent_end')
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    interrupted = agent.execution()
    saved = agent._snapshot_runtime.store.load_snapshot(interrupted['snapshot_id'])['state_json']['chat_messages']
    assert agent.resume_from_snapshot(interrupted['snapshot_id'], resume_input='Yes, approved')['final_message'] == 'done'
    assert seen[1] == saved + [{'role': 'user', 'content': 'Yes, approved'}]
    assert seen[2][0]['content'].endswith(agent.policy_runtime._simple_end_instruction())
    assert 'CURRENT STAGE IS:\n\nwrite final report' in seen[2][0]['content']
    assert all(item['content'] != 'Approve?' for item in seen[2])
    assert agent._snapshot_resume_messages is None


@pytest.mark.parametrize('failed_replay', [False, True])
def test_fresh_execution_starts_at_first_stage_after_replay(agent_factory, monkeypatch, failed_replay):
    agent = agent_factory(Stages=[IkaStage('first', 'first phase', []), IkaStage('second', 'second phase', [])])
    calls = []
    def chat(model, messages, history, **kwargs):
        index = kwargs['current_stage_index']
        calls.append(index)
        if len(calls) == 1:
            agent.request_pause_at('pre_model')
            return _turn('advance', 'stage_end')
        if failed_replay and len(calls) == 2:
            raise ValueError('replay failed')
        return _turn('done', 'agent_end') if index == 1 else _turn('advance', 'stage_end')
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    paused = agent.execution()
    agent.clear_pause_requests()
    if failed_replay:
        with pytest.raises(ValueError, match='replay failed'):
            agent.resume_from_snapshot(paused['snapshot_id'])
    else:
        assert agent.resume_from_snapshot(paused['snapshot_id'])['final_message'] == 'done'
    resumed_calls = list(calls)
    fresh = agent.execution()
    assert calls == resumed_calls + [0, 1]
    assert fresh['runtime']['run_id'] != paused['runtime']['run_id']
    assert agent._resume_state is None
    assert agent._snapshot_resume_messages is None
    assert agent._snapshot_runtime.pending_response is None


@pytest.mark.parametrize('boundary', ['pre_model', 'post_model', 'step_entry'])
@pytest.mark.parametrize('extensions', [0, 1])
def test_mid_stage_resume_preserves_consumed_steps_limit_and_extensions(agent_factory, monkeypatch, boundary, extensions):
    agent = agent_factory(Stages=[IkaStage('phase', 'work', [], stage_max_step=2)],
                          max_stage_extensions=extensions, extend_stage_steps_by=2)
    calls = []
    trigger = 2 + extensions if boundary == 'post_model' else 1 + 2 * extensions
    def chat(model, messages, history, **kwargs):
        calls.append(deepcopy(messages))
        if len(calls) == trigger:
            agent.request_pause_at(boundary)
        return _turn()
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    paused = agent.execution()
    assert paused['status'] == 'paused'
    agent.clear_pause_requests()
    assert agent.resume_from_snapshot(paused['snapshot_id'])['final_message'] == 'forced final'
    assert len(calls) == 2 + 2 * extensions
    saved = agent._snapshot_runtime.store.load_snapshot(paused['snapshot_id'])['state_json']['loop_state']
    assert saved['used_steps'] == (trigger - 1 if boundary == 'post_model' else trigger)
    assert saved['step_limit'] == 2 + 2 * extensions
    assert saved['extension_count'] == extensions


def test_mid_stage_resume_does_not_double_charge_steps_to_following_stage(agent_factory, monkeypatch):
    agent = agent_factory(maxsteps=4, Stages=[IkaStage('first', 'work', [], stage_max_step=2),
                                             IkaStage('second', 'finish', [], stage_max_step=0)])
    calls = []
    def chat(model, messages, history, **kwargs):
        calls.append(kwargs['current_stage_index'])
        if len(calls) == 1:
            agent.request_pause_at('pre_model')
        return _turn()
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    paused = agent.execution()
    agent.clear_pause_requests()
    agent.resume_from_snapshot(paused['snapshot_id'])
    assert calls == [0, 0, 1, 1]


@pytest.mark.parametrize('staged', [False, True])
def test_loop_budget_survives_resume_in_a_new_agent(agent_factory, monkeypatch, staged):
    options = {'maxsteps': 2, 'max_stage_extensions': 1, 'extend_stage_steps_by': 2,
               'max_step_extensions': 1, 'extend_steps_by': 2,
               'Stages': [IkaStage('phase', 'work', [], stage_max_step=2)] if staged else []}
    agent = agent_factory(**options)
    calls = []
    def chat(model, messages, history, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            agent.request_pause_at('pre_model')
        return _turn()
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    paused = agent.execution()
    assert len(calls) == 3
    other = agent_factory(**options)
    other.enable_snapshots(agent._snapshot_runtime.store.db_path)
    assert other.resume_from_snapshot(paused['snapshot_id'])['final_message'] == 'forced final'
    assert len(calls) == 4


def test_hitl_resume_preserves_the_stage_budget_already_consumed(agent_factory, monkeypatch):
    agent = agent_factory(Stages=[IkaStage('phase', 'ask then finish', [], hitl=True, stage_max_step=2)])
    calls = []
    def chat(model, messages, history, **kwargs):
        calls.append(deepcopy(messages))
        return {**_turn('Approve?'), 'interrupted': True, 'status': 'awaiting_user_input'} if len(calls) == 1 else _turn()
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    first = agent.execution()
    agent.resume_from_snapshot(first['snapshot_id'], resume_input='Yes')
    assert len(calls) == 2


@pytest.mark.parametrize('use_async', [False, True])
@pytest.mark.parametrize('extensions', [0, 1])
def test_exhausted_hitl_resume_extends_or_finalizes_before_stage_completion(
    agent_factory, monkeypatch, use_async, extensions,
):
    agent = agent_factory(Stages=[IkaStage('approval', 'ask approval', [], hitl=True, stage_max_step=1)],
                          max_stage_extensions=extensions, extend_stage_steps_by=2, use_async=use_async)
    calls, summaries = [], []
    def chat(model, messages, history, **kwargs):
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            return {**_turn('Approve?'), 'interrupted': True, 'status': 'awaiting_user_input'}
        return _turn('Answer processed', 'agent_end')
    async def async_chat(*args, **kwargs):
        return chat(*args, **kwargs)
    def summarize(model, history, **kwargs):
        summaries.append((deepcopy(history), kwargs['prompt_kind']))
        return 'forced final'
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    monkeypatch.setattr('IkaCore.agent_chat_support.async_chat', async_chat)
    monkeypatch.setattr('IkaCore.agent_policy.run_summarization', summarize)
    paused = agent.execution()
    result = agent.resume_from_snapshot(paused['snapshot_id'], resume_input='YES')
    assert result['final_message'] == ('Answer processed' if extensions else 'forced final')
    assert len(calls) == 1 + extensions
    assert summaries[0][1] == ('what_remains' if extensions else 'force_answer')
    assert any(item.get('message') == 'YES' for item in summaries[0][0]['messages'].values())
    if extensions:
        assert {'role': 'user', 'content': 'YES'} in calls[1]


@pytest.mark.parametrize('staged', [False, True])
@pytest.mark.parametrize('use_async', [False, True])
def test_snapshot_without_loop_state_uses_only_remaining_budget(agent_factory, monkeypatch, staged, use_async):
    options = {'maxsteps': 6 if staged else 3, 'use_async': use_async,
               'Stages': [IkaStage('phase', 'work', [], stage_max_step=3)] if staged else []}
    agent = agent_factory(**options)
    calls = []
    def chat(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            agent.request_pause_at('pre_model')
        return _turn()
    async def async_chat(*args, **kwargs):
        return chat(*args, **kwargs)
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    monkeypatch.setattr('IkaCore.agent_chat_support.async_chat', async_chat)
    paused = agent.execution()
    store = agent._snapshot_runtime.store
    snapshot = store.load_snapshot(paused['snapshot_id'])
    old_state = snapshot['state_json']
    del old_state['loop_state']
    historical = store.create_snapshot(snapshot['run_id'], snapshot['frame_id'], 'pre_model', old_state)
    other = agent_factory(**options)
    other.enable_snapshots(store.db_path)
    other.resume_from_snapshot(historical)
    assert len(calls) == 3


def test_snapshot_without_loop_state_subtracts_only_current_stage_steps(agent_factory, monkeypatch):
    agent = agent_factory(maxsteps=6, Stages=[IkaStage('first', 'start', [], stage_max_step=1),
                                             IkaStage('second', 'finish', [], stage_max_step=3)])
    calls = []
    def chat(*args, **kwargs):
        calls.append(kwargs['current_stage_index'])
        if len(calls) == 2:
            agent.request_pause_at('pre_model')
        return _turn()
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    paused = agent.execution()
    store = agent._snapshot_runtime.store
    snapshot = store.load_snapshot(paused['snapshot_id'])
    state = snapshot['state_json']
    del state['loop_state']
    historical = store.create_snapshot(snapshot['run_id'], snapshot['frame_id'], 'pre_model', state)
    agent.clear_pause_requests()
    agent.resume_from_snapshot(historical)
    assert calls == [0, 1, 1, 1]


@pytest.mark.parametrize('staged', [False, True])
@pytest.mark.parametrize('use_async', [False, True])
def test_exhausted_legacy_budget_does_not_regain_a_turn(agent_factory, monkeypatch, staged, use_async):
    options = {'maxsteps': 1, 'use_async': use_async,
               'Stages': [IkaStage('ask', 'approval', [], hitl=True, stage_max_step=1)] if staged else []}
    calls = []
    def chat(*args, **kwargs):
        calls.append(True)
        return {**_turn('Approve?'), 'interrupted': True, 'status': 'awaiting_user_input'} if staged else _turn()
    async def async_chat(*args, **kwargs):
        return chat(*args, **kwargs)
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    monkeypatch.setattr('IkaCore.agent_chat_support.async_chat', async_chat)
    agent = agent_factory(**options)
    if not staged:
        agent.request_pause_at('post_model')
    paused = agent.execution()
    store = agent._snapshot_runtime.store
    saved = store.load_snapshot(paused['snapshot_id'])
    state = saved['state_json']
    if staged:
        assert state['remaining_steps'] == 0
    del state['loop_state']
    legacy = store.create_snapshot(saved['run_id'], saved['frame_id'], saved['snapshot_kind'], state)
    other = agent_factory(**options)
    other.enable_snapshots(store.db_path)
    result = other.resume_from_snapshot(legacy, resume_input='ANSWER-YES' if staged else None)
    assert len(calls) == 1
    assert result['final_message'] == 'forced final'
    assert result.get('status') != 'awaiting_user_input'
