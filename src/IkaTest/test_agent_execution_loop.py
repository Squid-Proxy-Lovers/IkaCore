"""Behavioral contracts shared by simple agents and staged agents."""

import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from IkaCore import IkaBaseAgent, IkaStage
from IkaCore.execution_types import AgentChatTurn, ExecutionLoopState
from IkaModel.base import AgentEndException
from IkaModel.execution_hooks import ExecutionHooks, execution_hooks
from IkaModel.request_control import check_request_controls, request_controls
from IkaModel.runtime_errors import IkaRequestCancelled


def _call(name, text='done'):
    return {'function': {'name': name, 'arguments': json.dumps({'input': text})}}


def _response(content='', calls=()):
    return {'content': content, 'content_before_tools': content, 'tool_calls': list(calls),
            'executed_tool_calls': list(calls), 'usage': {'input_tokens': 2, 'output_tokens': 1, 'total_tokens': 3},
            'cost': {'total_cost': 0.01}}


@pytest.fixture
def scenario(monkeypatch):
    agents = []
    def make(staged, responses, **options):
        defaults = {'maxsteps': 3, 'max_step_extensions': 0, 'max_stage_extensions': 0}
        defaults.update(options)
        agent = IkaBaseAgent('fixture', 'fixture', 'test task', model_id='gpt-4o', api_key='fixture',
                            Stages=[IkaStage('phase', 'stage task', [])] if staged else [],
                            show_usage_level0=False, **defaults)
        agents.append(agent)
        calls = []
        iterator = iter(responses)
        def chat(model, messages, *args):
            check_request_controls()
            calls.append(deepcopy(messages))
            value = next(iterator)
            if isinstance(value, Exception):
                raise value
            return value
        agent._run_chat_round = chat
        agent._enforce_rate_limit_model = MagicMock()
        summaries = []
        def summarize(*args, **kwargs):
            summaries.append(kwargs['prompt_kind'])
            return 'forced final' if kwargs['prompt_kind'] == 'force_answer' else 'remaining work'
        monkeypatch.setattr('IkaCore.agent_runtime_foundation.run_summarization', summarize)
        return agent, calls, summaries
    yield make
    for agent in agents:
        agent.shutdown()


@pytest.mark.parametrize('staged', [False, True])
def test_loop_counts_turns_and_uses_executed_control_calls(scenario, staged):
    first = _response('intermediate', [_call('agent_end', 'premature')])
    first['executed_tool_calls'] = [_call('fixture')]
    agent, calls, summaries = scenario(staged, [first, _response('final', [_call('agent_end')])])
    result = agent.execution()
    assert result['final_message'] == 'done'
    assert result['usage']['total_tokens'] == 6 and result['cost']['total_cost'] == 0.02
    assert len(calls) == agent._enforce_rate_limit_model.call_count == 2
    assert summaries == []
    assert any(item.get('content') == 'intermediate' for item in calls[1]) == staged


@pytest.mark.parametrize('staged', [False, True])
def test_agent_end_exception_uses_content_and_counts_usage_once(scenario, staged):
    agent, calls, _ = scenario(staged, [AgentEndException(_response('finished via exception'))])
    result = agent.execution()
    assert result['final_message'] == 'finished via exception'
    assert result['usage']['total_tokens'] == 3 and len(calls) == 1


@pytest.mark.parametrize('staged', [False, True])
@pytest.mark.parametrize('extensions', [0, 1])
def test_loop_preserves_extension_limits_and_forced_finalization(scenario, staged, extensions):
    options = {'maxsteps': 1, 'max_step_extensions': extensions, 'max_stage_extensions': extensions,
               'extend_steps_by': 1, 'extend_stage_steps_by': 1}
    agent, calls, summaries = scenario(staged, [_response('partial')] * (1 + extensions), **options)
    result = agent.execution()
    assert result['final_message'] == 'forced final'
    assert len(calls) == 1 + extensions
    assert summaries == ['what_remains'] * extensions + ['force_answer']
    assert agent.maxsteps == (1 if staged else 1 + extensions)
    if extensions:
        assert 'remaining work' in calls[1][-1]['content']


def test_stages_advance_and_jump_without_replaying_extra_turns(scenario):
    agent, calls, summaries = scenario(True, [_response('first', [_call('stage_end')]),
        _response('back', [{'function': {'name': 'change_stage', 'arguments': '{"stage_index":0}'}}]),
        _response('finished', [_call('agent_end')])])
    agent.Stages.append(IkaStage('second', 'second stage task', [], allowed_back_to=[0]))
    result = agent.execution()
    assert result['final_message'] == 'done' and len(calls) == 3
    assert 'second stage task' in calls[1][0]['content']
    assert 'stage task' in calls[2][0]['content'] and summaries == []


@pytest.mark.parametrize('staged', [False, True])
def test_empty_turns_keep_existing_mode_policy(scenario, staged):
    agent, calls, summaries = scenario(staged, [_response()] * 3)
    assert agent.execution()['final_message'] == 'forced final'
    assert len(calls) == (3 if staged else 2)
    assert summaries == ['force_answer']


@pytest.mark.parametrize('staged', [False, True])
def test_invalid_completion_is_reported_and_propagates(scenario, staged):
    agent, calls, summaries = scenario(staged, [_response('', [_call('agent_end', '')])])
    agent.logger = MagicMock()
    with pytest.raises(ValueError, match='empty or invalid'):
        agent.execution()
    assert len(calls) == 1 and summaries == []
    if staged:
        assert 'ERROR in agent_end' in agent.logger.log_action.call_args.args[0]


@pytest.mark.parametrize('staged', [False, True])
def test_empty_exception_completion_reports_error(scenario, staged):
    agent, calls, _ = scenario(staged, [AgentEndException(_response())])
    agent.logger = MagicMock()
    with pytest.raises(ValueError, match='no output'):
        agent.execution()
    assert len(calls) == 1
    if staged:
        assert agent.logger.log_action.call_args.args[0].startswith('ERROR:')


@pytest.mark.parametrize('staged', [False, True])
def test_cancel_at_second_step_prevents_another_model_call(scenario, staged):
    agent, calls, _ = scenario(staged, [_response('partial'), _response('final', [_call('agent_end')])])
    cancelled = False
    remaining = []
    def boundary(label, payload):
        nonlocal cancelled
        if label == 'step_entry':
            remaining.append(payload['remaining_steps'])
            cancelled = len(remaining) == 2
    with request_controls(cancel_checker=lambda: cancelled), execution_hooks(ExecutionHooks(boundary, lambda: {}, lambda *args: "tool", lambda *args: None)):
        with pytest.raises(IkaRequestCancelled):
            agent.execution()
    assert len(calls) == 1 and remaining == [3, 2]


@pytest.mark.parametrize('staged', [False, True])
def test_progress_logs_preserve_step_numbers_and_cli_updates(scenario, staged, monkeypatch):
    agent, calls, _ = scenario(staged, [_response('partial'), _response('final', [_call('agent_end')])])
    agent.logger = MagicMock()
    cli = MagicMock()
    monkeypatch.setattr('IkaCore.agent_runtime_foundation.get_cli_output', lambda: cli)
    assert agent.execution()['final_message'] == 'done'
    indices = [call.kwargs['step_idx'] for call in agent.logger.log_step.call_args_list]
    assert indices == ([1] if staged else [0, 1])
    if not staged:
        cli.agent_init.assert_called_once_with('fixture', ['fixture'], step=1, description='Step 1/3')
        assert cli.set_step.call_args_list[-1].args == ('fixture', 2)
    assert len(calls) == 2


@pytest.mark.parametrize('staged', [False, True])
def test_final_fallback_uses_pre_tool_text(scenario, staged):
    agent, _, _ = scenario(staged, [])
    runtime = agent._prepare_stage_runtime(0, 3, None) if staged else agent._prepare_simple_runtime()
    turn = AgentChatTurn(_response(), True, 'after tools', 'before tools', [], [])
    result = agent._finish_execution_turn(runtime, ExecutionLoopState(3, used_steps=2), turn, None, 0)
    assert tuple(result) == (0, 'before tools', True, 'before tools', 2)
