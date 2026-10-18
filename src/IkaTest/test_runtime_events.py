"""Scoped output keeps runtime events independent of optional presentation."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from IkaCore import IkaBaseAgent, OutputConfig, RuntimeOutput, events, output_scope
from IkaCore.cli_output import set_output_sink, set_stdout_enabled
from IkaCore.events import OutputType, current_output
from IkaCore.output_adapters import make_output
from IkaModel.base import BareBoneModel
from IkaModel.worker_context import submit_with_controls


def test_no_subscriber_does_not_allocate_events_and_scopes_restore(monkeypatch):
    output = RuntimeOutput()
    monkeypatch.setattr(events, 'RuntimeEvent', lambda *a, **k: pytest.fail('allocated an unused event'))
    with output_scope(output):
        assert current_output() is output
        output.publish('unused', {})
        output.set_step('a', 3)
        assert output.get_step('a') == 3 and output.get_step('missing') == 0
        assert not output.is_buffering()
        with pytest.raises(ValueError, match='stop'):
            with output_scope(RuntimeOutput()):
                raise ValueError('stop')
        assert current_output() is output
    output.flush()
    output.shutdown()


def test_events_are_structured_isolated_and_follow_worker_context():
    first, second = [], []
    outer, inner = RuntimeOutput((first.append,)), RuntimeOutput((second.append,))
    with output_scope(outer):
        outer.set_instance_id(7)
        with ThreadPoolExecutor(max_workers=1) as pool:
            submit_with_controls(pool, lambda: current_output().log_action('worker')).result()
        with output_scope(inner):
            current_output().log_action('nested')
        current_output().log_action('parent')
    assert [e.data['action'] for e in first] == ['worker', 'parent']
    assert [e.data['action'] for e in second] == ['nested']
    assert first[0].instance_id == 7
    with pytest.raises(TypeError):
        first[0].data['action'] = 'changed'

    async def run():
        async def emit(output, value):
            with output_scope(output):
                await asyncio.sleep(0)
                current_output().log_action(value)
        await asyncio.gather(emit(outer, 'async-a'), emit(inner, 'async-b'))
    asyncio.run(run())
    assert first[-1].data['action'] == 'async-a' and second[-1].data['action'] == 'async-b'


def test_default_cli_ignores_legacy_global_sinks_and_buffers(capsys):
    sink = MagicMock()
    set_stdout_enabled(False)
    set_output_sink(sink)
    try:
        output = make_output(OutputConfig())
        output.cli_adapter.cli.configure(use_colors=False)
        with output_scope(output, share_cli=True):
            output.set_step('a', 1)
            output.start_parallel()
            assert output.is_buffering()
            # A child publisher shares workflow buffering, while retaining its subscriptions.
            child = make_output(OutputConfig())
            with output_scope(child):
                child.agent_init('a', ['a'], description='task', model='fixture')
                child.tool_call('lookup', {'q': 'x'}, ['a'], 1)
                child.tool_result('lookup', 'found', ['a'], 1)
                child.agent_response('a', 'done', ['a'], 1, is_final=True)
                child.summarization('a', 'summary', ['a'], 1)
                child.workflow_status('flow', 'complete')
                child.emit(OutputType.AGENT_INIT, 'prompt', ['a'])
            assert not capsys.readouterr().out
            output.end_parallel()
        rendered = capsys.readouterr().out
        assert '[AGENT RESPONSE]' in rendered and 'done' in rendered
        assert 'model=fixture' in rendered and 'summary' in rendered
        assert not sink.called
        output.flush()
        output.shutdown()
    finally:
        set_output_sink(None)
        set_stdout_enabled(True)


def test_optional_file_adapter_receives_runtime_events(tmp_path):
    path = tmp_path / 'events.jsonl'
    subscriber = MagicMock()
    subscriber.shutdown = MagicMock()
    output = make_output(OutputConfig(cli=False, logging_level=2, logging_file=str(path), subscribers=(subscriber,)))
    file_adapter = output.subscribers[-1]
    file_adapter.logger.log_input_enabled = True
    output.log_input([{'role': 'user', 'content': 'task'}])
    output.log_output('answer', {'input_tokens': 2}, {'total_cost': 0}, {})
    output.log_tool_results([{'name': 'lookup'}], ['found'])
    output.log_summary('summary')
    output.log_action('started')
    output.log_stage_start('phase', False, 2, 2)
    output.log_step('phase', 0, 'progress', [], {}, {}, 0.1)
    output.log_stage_end('phase', 1)
    output.log_hitl_prompt('phase')
    output.log_hitl_question('phase', 'continue?')
    output.log_hitl_input('phase', 'yes')
    output.log_hitl_answer('phase', 'yes')
    output.publish('custom', {'value': 4})
    output.flush()
    assert '"event": "output"' in path.read_text() and 'continue?' in path.read_text()
    assert subscriber.call_args.args[0].kind == 'custom'
    assert output.compute_cost('unknown-model', {})['total_cost'] == 0
    assert output.compute_cost('gpt-4o', {'input_tokens': 100, 'output_tokens': 20})['total_cost'] > 0
    output.shutdown()
    subscriber.shutdown.assert_called_once()


def test_agent_publishes_without_rendering_and_default_scope_is_lazy(monkeypatch, capsys):
    captured = []
    agent = IkaBaseAgent('a', prompt='task', model_id='gpt-4o', api_key='key',
                         output=OutputConfig(cli=False, subscribers=(captured.append,)))
    with output_scope(agent.events):
        BareBoneModel('gpt-4o', 'key', 'https://fixture.invalid', content_prompt='task', suppress_init_output=False)
    assert any(e.kind == 'display' for e in captured)
    assert capsys.readouterr().out == ''
    token = events._active_output.set(None)
    try:
        assert current_output() is current_output()
    finally:
        events._active_output.reset(token)
    agent.shutdown()


def test_usage_adapter_is_an_optional_subscriber():
    from IkaCore.output_adapters import UsageAdapter
    rendered = []
    output = RuntimeOutput((UsageAdapter(rendered.append),))
    output.log_action('no usage')
    output.log_step('phase', 1, '', [], {}, {}, 0)
    assert rendered == []
    output.log_step('phase', 1, '', [], {'input_tokens': 3}, {}, 0)
    assert '[USAGE] stage=phase' in rendered[0] and '3' in rendered[0]
    assert callable(UsageAdapter().writer)


def test_silent_workflow_scopes_preserve_silent_and_observed_callers(capsys):
    from IkaCore import IkaWorkflow, WorkflowNode

    class Fixture:
        name = 'fixture'
        Stages = ()
        use_async = False
        def clone_for_run(self):
            return Fixture()
        def execution(self):
            return {'final_message': 'done', 'summary': 'done'}
        def inject_workflow_context(self, context):
            pass
    workflow = IkaWorkflow('flow', 'description', [WorkflowNode('fixture', Fixture())], [],
                           output=OutputConfig(cli=False))
    observed = []
    for parent in (RuntimeOutput(), RuntimeOutput((observed.append,))):
        with output_scope(parent):
            result = workflow.run(use_async=True)
            assert result['fixture'].final == 'done' and current_output() is parent
    assert observed == [] and capsys.readouterr().out == ''
    workflow.async_executor.shutdown()
