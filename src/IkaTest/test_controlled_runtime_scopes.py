"""Deadline workers preserve the composed runtime's output and snapshot contexts."""

import json
import threading
from contextvars import ContextVar

import httpx
import pytest

from IkaCore import (
    ExecutionLimits,
    ExecutionPolicy,
    IkaBaseAgent,
    OutputConfig,
    ProviderConfig,
    RuntimeControl,
    RuntimeOutput,
    output_scope,
)
from IkaCore.events import current_output
from IkaCore.runtime_control import emit_tool_checkpoint
from IkaModel import request_controls, runtime_options
from IkaModel.execution_hooks import current_execution_context
from IkaModel.request_control import effective_max_retries, remaining_request_time
from IkaModel.request_transport import controlled_post
from IkaModel.runtime_errors import IkaRequestDeadlineExceeded
from IkaModel.runtime_policy import current_runtime_options


@pytest.mark.parametrize('provider', ['openai-chat', 'codex'])
@pytest.mark.parametrize('pause_point', ['post_model', 'tool:transport'])
def test_controlled_request_keeps_output_hooks_and_runtime_control(tmp_path, provider, pause_point):
    captured, parent_events, observed = [], [], []
    caller = threading.get_ident()
    marker = ContextVar('controlled_fixture', default='outside')
    url = ('https://chatgpt.com/backend-api/codex/responses' if provider == 'codex'
           else 'https://api.openai.com/v1/chat/completions')

    def respond(request):
        function = {'name': 'agent_end', 'arguments': json.dumps({'input': 'DONE'})}
        usage = {'input_tokens': 2, 'output_tokens': 1, 'total_tokens': 3}
        if provider == 'codex':
            event = {'type': 'response.completed', 'response': {'status': 'completed', 'usage': usage,
                'output': [{'type': 'function_call', 'call_id': 'end', **function}]}}
            return httpx.Response(200, text='data: ' + json.dumps(event) + '\n\n')
        return httpx.Response(200, json={'choices': [{'message': {'tool_calls': [
            {'id': 'end', 'type': 'function', 'function': function}]}}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}})

    def response_hook(response):
        assert threading.get_ident() != caller
        assert marker.get() == 'inside'
        assert effective_max_retries(9) == 1
        assert 0 < remaining_request_time() <= 10
        assert current_runtime_options().compact_at_fraction == 0.7
        context = current_execution_context()
        assert context['control'] is control
        assert current_output() is agent.events
        observed.append(context['run_id'])
        current_output().log_action('controlled response')
        emit_tool_checkpoint('transport', {'status_code': response.status_code})

    control = RuntimeControl(pause_points={pause_point})
    parent = RuntimeOutput((parent_events.append,))
    agent = IkaBaseAgent('controlled', prompt='Finish.', model_id='gpt-5.6-sol' if provider == 'codex' else 'gpt-4o',
        api_key='fixture', provider=ProviderConfig(api_url=url, use_responses_api=False),
        limits=ExecutionLimits(maxsteps=1), policy=ExecutionPolicy(max_step_extensions=0),
        output=OutputConfig(cli=False, subscribers=(captured.append,)))
    store = agent.enable_snapshots(str(tmp_path / 'runtime.db'), control)
    try:
        with httpx.Client(transport=httpx.MockTransport(respond), event_hooks={'response': [response_hook]}) as client:
            agent.client = client
            with output_scope(parent), request_controls(timeout=10, max_retries=1), runtime_options(compact_at_fraction=0.7):
                parent.set_instance_id(17)
                token = marker.set('inside')
                try:
                    result = agent.execution()
                finally:
                    marker.reset(token)
                assert current_output() is parent and marker.get() == 'outside'
                parent.log_action('parent restored')
            assert not client.is_closed
            assert result['status'] == 'paused' and result['pause_point'] == pause_point
            assert observed == [result['runtime']['run_id']]
            snapshot = store.load_snapshot(result['snapshot_id'])
            assert snapshot['snapshot_kind'] == pause_point
            if pause_point == 'post_model':
                # The saved turn is consumed outside the worker with no second request.
                agent.clear_pause_requests()
                resumed = agent.resume_from_snapshot(result['snapshot_id'])
                assert resumed['final_message'] == 'DONE' and resumed['usage']['total_tokens'] == 3
                assert len(observed) == 1
        actions = [event for event in captured if event.kind == 'action' and event.data['action'] == 'controlled response']
        assert len(actions) == 1 and actions[0].instance_id == 17
        assert [event.data['action'] for event in parent_events] == ['parent restored']
        assert current_execution_context() == {} and remaining_request_time() is None
    finally:
        agent.shutdown()


def test_deadline_response_cleanup_preserves_scope_after_caller_returns():
    captured, observed = [], []
    entered, close_started, release_close = threading.Event(), threading.Event(), threading.Event()
    release_read, closed = threading.Event(), threading.Event()
    marker = ContextVar('cleanup_fixture', default='outside')
    output = RuntimeOutput((captured.append,))

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            entered.set()
            assert release_read.wait(2)
            yield b'{}'

        def close(self):
            close_started.set()
            try:
                assert release_close.wait(2)
                observed.append((marker.get(), effective_max_retries(9),
                                 current_runtime_options().compact_at_fraction))
                current_output().log_action('late cleanup')
            finally:
                release_read.set()
                closed.set()

    try:
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))) as client:
            with output_scope(output), request_controls(timeout=.1, max_retries=1), runtime_options(compact_at_fraction=.7):
                output.set_instance_id(23)
                token = marker.set('inside')
                try:
                    with pytest.raises(IkaRequestDeadlineExceeded):
                        controlled_post('https://fixture.invalid', {}, {}, 10, client)
                finally:
                    marker.reset(token)
            assert entered.is_set() and close_started.wait(1) and not client.is_closed
            assert marker.get() == 'outside' and remaining_request_time() is None
            release_close.set()
            assert closed.wait(1)
        assert observed == [('inside', 1, .7)]
        assert len(captured) == 1 and captured[0].data['action'] == 'late cleanup'
        assert captured[0].instance_id == 23
    finally:
        release_close.set()
        release_read.set()
        for thread in threading.enumerate():
            if thread.name.startswith('ika-http-'):
                thread.join(2)
                assert not thread.is_alive()
