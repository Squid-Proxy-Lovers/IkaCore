"""Workflow replay continues the saved child, including independently answered children."""

from collections import Counter
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from IkaCore import IkaBaseAgent, IkaStage, IkaTools
from IkaCore.workflow import IkaWorkflow, WorkflowEdge, WorkflowNode
from IkaModel.chat_interface.tool_execution_sync import execute_tool
from IkaTest.test_snapshot_resume_state import _turn


@pytest.fixture
def approval_workflow(tmp_path, monkeypatch):
    workflows, agents, calls = [], [], []
    def create(*, parallel=False, use_async=False, tool=None):
        names = ('root', 'left', 'right', 'join') if parallel else ('approval', 'final')
        approvals = {'left', 'right'} if parallel else {'approval'}
        nodes = []
        for name in names:
            stages = [IkaStage('approve', 'ask approval', [tool] if tool else [], hitl=True)] if name in approvals else []
            agent = IkaBaseAgent(name, 'test', 'task', model_id='gpt-4o', api_key='test', Stages=stages,
                                maxsteps=6, max_stage_extensions=0, show_usage_level0=False, use_async=use_async)
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
        if kwargs.get('current_stage_index') is not None and not any(m['content'] == 'YES' for m in messages):
            if 'effect' in kwargs['tool_executors']:
                execute_tool('effect', {}, kwargs['tool_executors'])
            messages.append({'role': 'assistant', 'content': 'Approve?'})
            return {**_turn('Approve?'), 'interrupted': True, 'status': 'awaiting_user_input'}
        return _turn(model.agent_name + ' done', 'agent_end')
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
