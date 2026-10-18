"""Resumed transcripts, execution positions, and stage budgets are consumed once."""

import json
from copy import deepcopy

import pytest

from IkaCore import IkaBaseAgent, IkaStage


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
                            show_usage_level0=False, **defaults)
        agent.enable_snapshots(str(tmp_path / f'runtime-{len(agents)}.db'))
        agents.append(agent)
        return agent
    monkeypatch.setattr('IkaCore.agent_runtime_foundation.run_summarization', lambda *args, **kwargs: 'forced final')
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
    assert seen[2][0]['content'].endswith(agent._simple_end_instruction())
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
