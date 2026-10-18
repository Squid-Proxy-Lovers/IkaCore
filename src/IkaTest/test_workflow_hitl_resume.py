"""Workflow replay continues the saved child, including independently answered children."""
from collections import Counter
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from IkaCore import ExecutionLimits, ExecutionPolicy, IkaBaseAgent, IkaStage, IkaTools, OutputConfig
from IkaCore.workflow import IkaWorkflow, WorkflowEdge, WorkflowNode
from IkaModel.chat_interface.tool_execution_sync import execute_tool
from IkaTest.test_snapshot_resume_state import _turn


@pytest.fixture
def approval_workflow(tmp_path, monkeypatch):
    workflows, agents, calls = [], [], []
    def create(*, parallel=False, use_async=False, tool=None, version_root=False):
        names = ('root', 'left', 'right', 'join') if parallel else ('approval', 'final')
        approvals = {'left', 'right'} if parallel else {'approval'}
        nodes = []
        for name in names:
            stages = [IkaStage('approve', 'ask approval', [tool] if tool else [], hitl=True)] if name in approvals else []
            agent = IkaBaseAgent(name, 'test', 'task', model_id='gpt-4o', api_key='test', Stages=stages,
                                limits=ExecutionLimits(maxsteps=6), policy=ExecutionPolicy(max_stage_extensions=0, use_async=use_async), output=OutputConfig(show_usage_level0=False))
            agent._test_version_root = version_root
            agents.append(agent)
            nodes.append(WorkflowNode(name, agent))
        edges = ([WorkflowEdge('root', n) for n in approvals] + [WorkflowEdge(n, 'join') for n in approvals]
                 if parallel else [WorkflowEdge('approval', 'final')])
        workflow = IkaWorkflow('approval-flow', 'test', nodes, edges,
                               compress_hook=lambda contexts, agent: '\n'.join(contexts))
        store = workflow.enable_snapshots(str(tmp_path / 'workflow.db'))
        workflows.append(workflow)
        return workflow, store, calls
    def chat(model, messages, history, **kwargs):
        calls.append((model.agent_name, deepcopy(messages)))
        if model.agent_name == 'root' and agents[-1]._test_version_root:
            return _turn(f"ROOT-VERSION-{sum(name == 'root' for name, _ in calls)}", 'agent_end')
        answers = [m['content'] for m in messages if m['role'] == 'user' and m['content'].startswith('ANSWER-')]
        if kwargs.get('current_stage_index') is not None and not answers and not any(m['content'] == 'YES' for m in messages):
            if 'effect' in kwargs['tool_executors']:
                execute_tool('effect', {}, kwargs['tool_executors'])
            messages.append({'role': 'assistant', 'content': 'Approve?'})
            return {**_turn('Approve?'), 'interrupted': True, 'status': 'awaiting_user_input'}
        return _turn(answers[-1] if answers else model.agent_name + ' done', 'agent_end')
    async def async_chat(*args, **kwargs):
        return chat(*args, **kwargs)
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', chat)
    monkeypatch.setattr('IkaCore.agent_chat_support.async_chat', async_chat)
    monkeypatch.setattr('IkaCore.workflow_async.get_cli_output', lambda: MagicMock())
    yield create
    for workflow in workflows:
        workflow.async_executor.shutdown()
    for agent in agents:
        agent.shutdown()


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('answer_child', [False, True])
def test_workflow_resume_answers_saved_instance_and_continues_dependents(approval_workflow, asynchronous, answer_child):
    workflow, store, calls = approval_workflow(use_async=asynchronous)
    assert workflow.run(use_async=asynchronous) == {}
    info = workflow.snapshot_runtime_info()
    interrupted = info['interrupted_instances']
    assert list(interrupted) == ['approval:0']
    saved = store.load_snapshot(info['last_snapshot_id'])['state_json']
    assert saved['interrupted_instances'] == interrupted
    # Opening the database in another workflow must retain the interrupted mapping.
    other, _, _ = approval_workflow(use_async=asynchronous)
    if answer_child:
        child = workflow.nodes[0].agent.clone_for_run()
        child.enable_snapshots(store.db_path)
        try:
            assert child.resume_from_snapshot(interrupted['approval:0'], resume_input='YES')['final_message'] == 'approval done'
        finally:
            child.shutdown()
        result = other.resume_from_snapshot(info['last_snapshot_id'])
    else:
        assert other.resume_from_snapshot(info['last_snapshot_id']) == {}
        assert len(calls) == 1  # Unanswered resume must not ask a fresh clone again.
        result = other.resume_from_snapshot(other.snapshot_runtime_info()['last_snapshot_id'], resume_input='YES')
    assert set(result) == {'approval', 'final'}
    assert Counter(name for name, _ in calls) == {'approval': 2, 'final': 1}
    assert {'role': 'user', 'content': 'YES'} in calls[1][1]
    assert other.snapshot_runtime_info()['interrupted_instances'] == {}
    assert store.get_run(info['run_id'])['status'] == 'completed'
    assert 'approval done' in str(calls[-1][1])


def test_parallel_resume_keeps_each_interruption_and_completed_siblings(approval_workflow):
    workflow, store, calls = approval_workflow(parallel=True)
    assert set(workflow.run(use_async=True)) == {'root'}
    info = workflow.snapshot_runtime_info()
    assert set(info['interrupted_instances']) == {'left:0', 'right:0'}
    saved = store.load_snapshot(info['last_snapshot_id'])['state_json']
    assert set(saved['instance_results']) == {'root:0'}
    with pytest.raises(ValueError, match='mapping'):
        workflow.resume_from_snapshot(info['last_snapshot_id'], resume_input='YES')
    with pytest.raises(ValueError, match='keys'):
        workflow.resume_from_snapshot(info['last_snapshot_id'], resume_input={'unknown:0': 'YES'})
    partial = workflow.resume_from_snapshot(info['last_snapshot_id'], resume_input={'left:0': 'YES'})
    assert set(partial) == {'root', 'left'}
    info = workflow.snapshot_runtime_info()
    assert set(info['interrupted_instances']) == {'right:0'}
    result = workflow.resume_from_snapshot(info['last_snapshot_id'], resume_input={'right:0': 'YES'})
    assert set(result) == {'root', 'left', 'right', 'join'}
    assert Counter(name for name, _ in calls) == {'root': 1, 'left': 2, 'right': 2, 'join': 1}


def test_workflow_resume_preserves_effects_before_child_interruption(approval_workflow):
    effects = []
    tool = IkaTools('effect', 'test', {'type': 'object'}, execute_function=lambda args: effects.append(1), replay_policy='deny')
    workflow, _, calls = approval_workflow(tool=tool)
    workflow.run()
    result = workflow.resume_from_snapshot(workflow.snapshot_runtime_info()['last_snapshot_id'], resume_input='YES')
    assert set(result) == {'approval', 'final'}
    assert effects == [1] and len(calls) == 3


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('use_async', [False, True])
@pytest.mark.parametrize('source_completed', [False, True])
def test_workflow_fork_does_not_mutate_source_child(approval_workflow, parallel, use_async, source_completed):
    workflow, store, _ = approval_workflow(use_async=use_async)
    workflow.run(use_async=parallel)
    info = workflow.snapshot_runtime_info()
    child_id = info['interrupted_instances']['approval:0']
    child_root = store.load_snapshot(child_id)['state_json']['root_frame_id']
    if source_completed:
        assert workflow.resume_from_snapshot(info['last_snapshot_id'], resume_input='ANSWER-SOURCE')['approval'].final == 'ANSWER-SOURCE'
    before = store.get_frame(child_root)
    fork = store.fork_run_from_snapshot(info['last_snapshot_id'])
    fork_child_id = fork['state']['interrupted_instances']['approval:0']
    assert fork_child_id != child_id
    assert store.load_snapshot(fork_child_id)['run_id'] == fork['run_id']
    result = workflow.resume_from_snapshot(fork['snapshot_id'], resume_input='ANSWER-FORK')
    assert result['approval'].final == 'ANSWER-FORK'
    assert store.get_frame(child_root) == before
    source = workflow.resume_from_snapshot(info['last_snapshot_id'], resume_input='ANSWER-SOURCE')
    assert source['approval'].final == 'ANSWER-SOURCE'
    # Forking a fork must remap the continuation again.
    another = store.fork_run_from_snapshot(fork['snapshot_id'])
    result = workflow.resume_from_snapshot(another['snapshot_id'], resume_input='ANSWER-ANOTHER')
    assert result['approval'].final == 'ANSWER-ANOTHER'
    assert store.get_frame(child_root)['outputs_json']['final_message'] == 'ANSWER-SOURCE'


def test_workflow_rejects_child_snapshot_from_another_run(approval_workflow):
    workflow, store, _ = approval_workflow()
    workflow.run()
    original = workflow.snapshot_runtime_info()
    saved = store.load_snapshot(original['last_snapshot_id'])
    workflow.run()
    other = workflow.snapshot_runtime_info()
    saved['state_json']['interrupted_instances'] = other['interrupted_instances']
    malformed = store.create_snapshot(saved['run_id'], saved['frame_id'], 'workflow_paused', saved['state_json'])
    before = store.list_frames(other['run_id'])
    with pytest.raises(ValueError, match='workflow run'):
        workflow.resume_from_snapshot(malformed, resume_input='ANSWER-WRONG')
    assert store.list_frames(other['run_id']) == before


@pytest.mark.parametrize('parallel', [False, True])
def test_rejected_workflow_fork_leaves_no_partial_records(approval_workflow, parallel):
    workflow, store, _ = approval_workflow(parallel=True)
    workflow.run(use_async=parallel)
    saved = store.load_snapshot(workflow.snapshot_runtime_info()['last_snapshot_id'])
    workflow.run(use_async=parallel)
    foreign = next(iter(workflow.snapshot_runtime_info()['interrupted_instances'].values()))
    interrupted = saved['state_json']['interrupted_instances']
    interrupted[list(interrupted)[-1]] = foreign
    malformed = store.create_snapshot(saved['run_id'], saved['frame_id'], 'workflow_paused', saved['state_json'])
    def counts():
        with store._connect() as connection:
            return [connection.execute(f'select count(*) from {table}').fetchone()[0]
                    for table in ('runs', 'frames', 'snapshots', 'events')]
    before = counts()
    with pytest.raises(ValueError, match='workflow run'):
        store.fork_run_from_snapshot(malformed)
    assert counts() == before


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('reuse', [False, True])
def test_selective_replay_invalidates_interrupted_descendant(approval_workflow, parallel, reuse):
    workflow, _, calls = approval_workflow(parallel=True, version_root=True)
    workflow.run(use_async=parallel)
    old = workflow.snapshot_runtime_info()
    control = workflow._workflow_snapshot.control
    control.parallel_replay = {'root': [0]} if reuse else {}
    control.reuse_historical_parallel_results = reuse
    result = workflow.resume_from_snapshot(old['last_snapshot_id'], resume_input={'left:0': 'ANSWER-STALE'})
    assert set(result) == {'root'}
    approvals = ('left', 'right') if parallel else ('left',)
    assert Counter(name for name, _ in calls) == {'root': 2, **dict.fromkeys(approvals, 2)}
    for name in approvals:
        messages = [messages for node, messages in calls if node == name][-1]
        assert 'ROOT-VERSION-2' in str(messages)
        assert 'ROOT-VERSION-1' not in str(messages)
        assert 'ANSWER-STALE' not in str(messages)
    current = workflow.snapshot_runtime_info()
    assert set(current['interrupted_instances'].values()).isdisjoint(old['interrupted_instances'].values())
    control.parallel_replay = {}
    control.reuse_historical_parallel_results = True
    for _ in range(3):
        if not current['interrupted_instances']:
            break
        result = workflow.resume_from_snapshot(current['last_snapshot_id'],
            resume_input={key: 'ANSWER-FRESH-' + key for key in current['interrupted_instances']})
        current = workflow.snapshot_runtime_info()
    assert set(result) == {'root', 'left', 'right', 'join'}
    assert Counter(name for name, _ in calls)['root'] == 2


@pytest.mark.parametrize('parallel', [False, True])
def test_restarting_interrupted_descendant_checks_replay_safety(approval_workflow, parallel):
    effects = []
    tool = IkaTools('effect', 'test', {'type': 'object'}, execute_function=lambda args: effects.append(1), replay_policy='deny')
    workflow, _, calls = approval_workflow(parallel=True, version_root=True, tool=tool)
    workflow.run(use_async=parallel)
    old = workflow.snapshot_runtime_info()
    before = len(calls)
    workflow._workflow_snapshot.control.parallel_replay = {'root': [0]}
    with pytest.raises(ValueError, match='non-replayable'):
        workflow.resume_from_snapshot(old['last_snapshot_id'])
    assert len(calls) == before and len(effects) == (2 if parallel else 1)
    workflow.resume_from_snapshot(old['last_snapshot_id'], allow_unsafe_replay=True)
    assert len(effects) == (4 if parallel else 2)
    assert 'ROOT-VERSION-2' in str([messages for name, messages in calls if name == 'left'][-1])


def test_selecting_interrupted_instance_restarts_only_that_continuation(approval_workflow):
    workflow, store, calls = approval_workflow(parallel=True)
    workflow.run(use_async=True)
    old = workflow.snapshot_runtime_info()
    workflow._workflow_snapshot.control.parallel_replay = {'left': [0]}
    workflow.resume_from_snapshot(old['last_snapshot_id'], resume_input={'left:0': 'ANSWER-STALE'})
    current = workflow.snapshot_runtime_info()
    def roots(info):
        return {key: store.load_snapshot(value)['state_json']['root_frame_id']
                for key, value in info['interrupted_instances'].items()}
    assert roots(current)['right:0'] == roots(old)['right:0']
    assert roots(current)['left:0'] != roots(old)['left:0']
    assert Counter(name for name, _ in calls) == {'root': 1, 'left': 2, 'right': 1}
