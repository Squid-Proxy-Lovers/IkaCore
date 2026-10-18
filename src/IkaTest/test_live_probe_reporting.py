"""Offline tests of the live harness: routing, bounds, skips and report trust boundary."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from IkaModel.request_control import effective_max_retries, remaining_request_time


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / 'scripts'))
    return importlib.import_module('live_provider_check')


def test_live_probe_redacts_errors_and_captured_output(runner, capsys):
    private = 'sk-fake-key-NEVER-RECORD'
    raw_output = 'PRIVATE_MODEL_OUTPUT'
    def fail():
        print(private, raw_output)
        print(private, file=sys.stderr)
        raise RuntimeError('backend returned 401: ' + private + raw_output)
    report = runner.run_case('sync_text', fail)
    assert report['status'] == 'failed'
    assert report['error_type'] == 'RuntimeError' and report['http_status'] == 401
    assert private not in json.dumps(report) and raw_output not in json.dumps(report)
    assert capsys.readouterr() == ('', '')
    result = runner.run_case('sync_text', lambda: {'content': raw_output, 'api_key': private,
        'completed': raw_output, 'usage': {'total_tokens': 3, 'input_tokens': private, 'secret': private},
        'error_type': private, 'reason': raw_output, 'unknown': private})
    assert result['usage'] == {'total_tokens': 3}
    assert private not in json.dumps(result) and raw_output not in json.dumps(result)


def test_live_probe_bounds_requests_and_restores_controls(runner):
    def check():
        assert effective_max_retries(99) == 1
        assert 0 < remaining_request_time() <= 60
        return {'completed': True}
    report = runner.run_case('sync_text', check)
    assert report['status'] == 'passed' and report['completed']
    assert effective_max_retries(99) == 99 and remaining_request_time() is None


def test_case_selection_and_cli(runner):
    assert runner.select_cases(None) == list(runner.CASES)
    assert runner.select_cases(['async_tools', 'sync_text', 'async_tools']) == ['async_tools', 'sync_text']
    with pytest.raises(ValueError):
        runner.select_cases(['arbitrary'])
    with pytest.raises(SystemExit):
        runner.parser().parse_args(['--case-timeout', '121'])
    with pytest.raises(SystemExit):
        runner.parser().parse_args(['--provider', 'codex', '--all-configured'])


def test_all_configured_skip_schema_and_fail_fast(runner):
    calls = []
    def execute(provider, model, name, timeout, env):
        calls.append((provider, name))
        return {'case': name, 'status': 'failed', 'seconds': 0, 'usage': {}}
    identity = {'package_version': '2.0.0', 'git_revision': '0' * 40, 'runtime_sha256': '1' * 64}
    for provider, config in runner.PROVIDERS.items():
        skipped = runner.provider_report(provider, 'fixture', identity, list(runner.CASES), {}, 60, execute)
        assert skipped['schema_version'] == 1 and skipped['runtime'] == identity
        assert skipped['cases_requested'] == list(runner.CASES)
        assert len(skipped['results']) == 15
        assert all(row['status'] == 'skipped' and row['reason'] == 'not_configured' for row in skipped['results'])
        missing = runner.provider_report(provider, None, identity, ['sync_text'], {config['key']: 'fake'}, 60, execute)
        assert missing['results'][0]['reason'] == 'model_required'
    assert calls == []
    report = runner.provider_report('codex', 'fixture', identity, list(runner.CASES), {'CODEX_BEARER': 'fake'}, 60, execute)
    assert calls == [('codex', 'sync_text')]
    assert len(report['results']) == 15
    assert all(row['reason'] == 'connectivity_failed' for row in report['results'][1:])


def test_all_configured_main_prints_clear_skips_without_credentials(runner, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(runner, 'prepare_environment', lambda: {})
    monkeypatch.setattr(runner, 'installed_identity', lambda: {'package_version': '2.0.0'})
    monkeypatch.setattr(runner, 'get_bearer', lambda: pytest.fail('implicit local login'))
    assert runner.main(['--all-configured', '--output-dir', str(tmp_path)]) == 0
    assert len(list(tmp_path.glob('*.json'))) == 8
    output = capsys.readouterr().out
    assert all(f'{provider}: not configured' in output for provider in runner.PROVIDERS)
    assert 'sync_text' in (tmp_path / 'summary.md').read_text()


def test_model_identity_cannot_contain_secret_or_output(runner):
    for model in ['raw output\n\n', 'sk-fake-key', 'prefix-sk-fake-key']:
        with pytest.raises(ValueError):
            runner.validate_model(model, {'OPENAI_API_KEY': 'sk-fake-key'})
    assert runner.validate_model('vendor/model-1:free', {}) == 'vendor/model-1:free'


def test_diagnostics_are_disabled(runner, monkeypatch):
    for name in ('IKA_DUMP_REQUESTS', 'GLM_REASONING_LOG', 'CODEX_REFRESH_TOKEN_URL_OVERRIDE',
                 'IKACORE_ALLOW_CODEX_TOKEN_URL_OVERRIDE', 'SSLKEYLOGFILE', 'PYTHONPATH'):
        monkeypatch.setenv(name, '/private/secret-output')
    env = runner.prepare_environment()
    assert all(name not in env for name in ('IKA_DUMP_REQUESTS', 'GLM_REASONING_LOG', 'SSLKEYLOGFILE', 'PYTHONPATH'))


def test_isolated_case_enforces_hard_wall_clock_and_discards_output(runner, monkeypatch):
    def blocked(command, directory, env, timeout):
        assert timeout == 1
        assert 'sk-FAKE' not in ' '.join(command)
        raise subprocess.TimeoutExpired(command, 1, output='sk-FAKE raw output')
    monkeypatch.setattr(runner, 'case_process', blocked)
    row = runner.isolated_case('codex', 'fixture', 'sync_text', 1, {'CODEX_BEARER': 'sk-FAKE'})
    assert row['status'] == 'failed' and row['error_type'] == 'TimeoutExpired'
    assert 'sk-FAKE' not in json.dumps(row) and 'raw output' not in json.dumps(row)


def test_isolated_case_sanitizes_child_results(runner, monkeypatch):
    def child(command, directory, env, timeout):
        Path(command[command.index('--child-output') + 1]).write_text(json.dumps({
            'status': 'passed', 'raw_output': 'PRIVATE', 'usage': {'total_tokens': 7, 'PRIVATE': 'SECRET'},
            'content': 'PRIVATE', 'completed': True, 'usage_complete': True}))
        return 0
    monkeypatch.setattr(runner, 'case_process', child)
    row = runner.isolated_case('codex', 'fixture', 'sync_text', 1, {})
    assert row['completed'] and row['usage_complete'] and row['usage'] == {'total_tokens': 7}
    assert 'PRIVATE' not in json.dumps(row) and 'SECRET' not in json.dumps(row)


@pytest.mark.parametrize('provider,model', [
    ('codex', 'gpt-5.6-sol'), ('openai-responses', 'gpt-5'), ('openai-chat', 'gpt-4o-mini'),
    ('anthropic', 'claude-fixture'), ('deepseek', 'deepseek-chat'), ('gemini', 'gemini-fixture'),
    ('openrouter', 'vendor/fixture'), ('zai', 'glm-4.7')])
def test_provider_routing_and_token_caps(runner, provider, model):
    from IkaModel.chat_helpers_common import build_provider_request
    from IkaModel.request_interface import get_provider
    probe = runner.Probe(model, 'sk-fake', provider)
    actual = get_provider(model, probe.url, probe.model().use_responses_api)
    assert actual == runner.PROVIDERS[provider]['adapter']
    _, _, payload = build_provider_request(actual, probe.model(), [{'role': 'user', 'content': 'OK'}], runner.init_message_history())
    if provider == 'codex':
        assert payload['stream'] is True and 'max_output_tokens' not in payload
    elif provider == 'gemini':
        assert payload['generationConfig']['maxOutputTokens'] == 512
    else:
        assert payload.get('max_tokens', payload.get('max_output_tokens', payload.get('max_completion_tokens'))) == 512
    if provider == 'zai':
        assert '/api/coding/paas/v4/' in probe.url
        assert payload['thinking'] == {'type': 'enabled'}


def test_mocked_transport_text_usage_deadline_and_retry(runner, monkeypatch):
    client, async_client = httpx.Client, httpx.AsyncClient
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'OK'}}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}})
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: async_client(transport=httpx.MockTransport(respond), **kwargs))
    probe = runner.Probe('gpt-4o-mini', 'sk-fake', 'openai-chat')
    with runner.observe_usage(probe) as usage:
        for name in ('sync_text', 'async_text', 'usage_accounting'):
            assert runner.run_case(name, runner.operations(probe)[name])['status'] == 'passed'
        assert usage['total_tokens'] == 9
    assert len(calls) == 3
    assert runner.run_case('deadline', lambda: runner.deadline(probe))['status'] == 'passed'
    assert len(calls) == 3
    # Use the actual MockTransport case without the client-factory patch.
    monkeypatch.setattr(httpx, 'Client', client)
    row = runner.run_case('transient_retry', lambda: runner.transient_retry(probe))
    assert row['status'] == 'passed' and row['attempts'] == 2 and row['synthetic_transport']


def test_unsupported_capabilities_are_explicit_skips(runner):
    probe = runner.Probe('claude-fixture', 'fake', 'anthropic')
    assert runner.run_case('reasoning_tools', lambda: runner.reasoning_tools(probe))['reason'] == 'runtime_no_extended_thinking'
    assert runner.run_case('streaming_sse', lambda: runner.streaming_sse(probe))['reason'] == 'runtime_no_sse'


def test_markdown_summary_rejects_untrusted_cells(runner, tmp_path):
    from live_summary import write_summary
    (tmp_path / 'codex.json').write_text(json.dumps({'model': 'SECRET', 'results': [
        {'case': 'sync_text', 'status': 'passed', 'seconds': 1.25, 'usage': {'total_tokens': 5}},
        {'case': 'SECRET', 'status': 'passed'}, {'case': 'async_text', 'status': 'SECRET'}]}))
    write_summary(tmp_path)
    summary = (tmp_path / 'summary.md').read_text()
    assert 'SECRET' not in summary and '| codex | sync_text | passed | 1.25 | 5 |' in summary


def test_editable_install_is_rejected(runner):
    with pytest.raises(ValueError, match='installed wheel'):
        runner.installed_identity()


def test_mocked_tool_replay_compaction_and_workflow(runner, monkeypatch):
    client, async_client = httpx.Client, httpx.AsyncClient
    token_usage = {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}
    def respond(request):
        payload = json.loads(request.content)
        text = json.dumps(payload)
        tools = [entry['function']['name'] for entry in payload.get('tools', [])]
        results = [m['content'] for m in payload.get('messages', []) if m['role'] == 'tool' and str(m['content']).startswith('fixture_')]
        if results:
            message = {'content': results[-1]}
        elif not tools:
            message = {'content': 'The synthetic fixture answer is COMPACT_OK.'}
        elif 'lookup_fixture' in tools:
            messages = payload['messages']
            results = [m['content'] for m in messages if m['role'] == 'tool']
            if results:
                message = {'content': results[-1]}
            else:
                message = {'tool_calls': [{'id': 'fixture', 'type': 'function', 'function': {
                    'name': 'lookup_fixture', 'arguments': '{"label":"check"}'}}]}
        else:
            if 'ask_user' in tools and not any(m.get('content') == 'YES' for m in payload['messages']):
                function, arguments = 'ask_user', {'question': 'Approve?'}
            else:
                function = 'agent_end'
                arguments = {'input': 'UPSTREAM_OK' if 'ask_user' in tools else
                             'WORKFLOW_OK' if 'WORKFLOW_OK' in text else 'COMPACT_OK'}
            message = {'tool_calls': [{'id': 'fixture', 'type': 'function', 'function': {
                'name': function, 'arguments': json.dumps(arguments)}}]}
        return httpx.Response(200, json={'choices': [{'message': message}], 'usage': token_usage})
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: async_client(transport=httpx.MockTransport(respond), **kwargs))
    probe = runner.Probe('gpt-4o-mini', 'sk-fake', 'openai-chat')
    for name in ('sync_tools', 'async_tools', 'compaction', 'workflow_hitl'):
        row = runner.run_case(name, runner.operations(probe)[name])
        assert row['status'] == 'passed', row


def test_mocked_sse_collector_and_codex_retry(runner, monkeypatch):
    import contextlib

    from IkaModel.codex import chat_helpers_codex
    envelope = {'type': 'response.completed', 'response': {'status': 'completed', 'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'OK'}]}],
        'usage': {'input_tokens': 2, 'output_tokens': 1, 'total_tokens': 3}}}
    def respond(request):
        return httpx.Response(200, text='data: ' + json.dumps(envelope) + '\n\n')
    @contextlib.contextmanager
    def stream(url, headers, payload, timeout, client):
        with httpx.Client(transport=httpx.MockTransport(respond)) as transport, transport.stream('POST', url) as response:
            yield response
    monkeypatch.setattr(chat_helpers_codex, 'open_stream', stream)
    probe = runner.Probe('gpt-5.6-sol', 'fake', 'codex')
    row = runner.run_case('streaming_sse', lambda: runner.streaming_sse(probe))
    assert row['status'] == 'passed' and row['sse_parsed']
    # Restore the real stream selector so the retry case uses its own MockTransport.
    monkeypatch.undo()
    row = runner.run_case('transient_retry', lambda: runner.transient_retry(probe))
    assert row['status'] == 'passed' and row['attempts'] == 2


def test_reasoning_probe_checks_next_payload(runner, monkeypatch):
    from IkaModel.chat_interface import chat_request
    client = httpx.Client
    captured = []
    def respond(request):
        payload = json.loads(request.content)
        captured.append(payload)
        outputs = [item['output'] for item in payload['input'] if item['type'] == 'function_call_output']
        output = ([{'type': 'message', 'content': [{'type': 'output_text', 'text': outputs[-1]}]}] if outputs else [
            {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'PRIVATE_THINKING'}]},
            {'type': 'function_call', 'call_id': 'fixture', 'name': 'lookup_fixture', 'arguments': '{"label":"check"}'}])
        return httpx.Response(200, json={'output': output, 'usage': {'input_tokens': 2, 'output_tokens': 1, 'total_tokens': 3}})
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    probe = runner.Probe('gpt-5', 'fake', 'openai-responses')
    original = chat_request.build_provider_request
    # Deliberately remove reasoning to ensure the harness reports failure, not a false pass.
    def drop_reasoning(*args, **kwargs):
        url, headers, payload = original(*args, **kwargs)
        payload['input'] = [item for item in payload['input'] if item.get('type') != 'reasoning']
        for item in payload['input']:
            item.pop('reasoning_content', None)
        return url, headers, payload
    monkeypatch.setattr(chat_request, 'build_provider_request', drop_reasoning)
    result = runner.run_case('reasoning_tools', lambda: runner.reasoning_tools(probe))
    assert len(captured) == 2 and result['status'] == 'failed' and result['error_type'] == 'AssertionError'
    assert 'PRIVATE_THINKING' not in json.dumps(result)


@pytest.mark.parametrize('provider,model', [('zai', 'glm-4.7'), ('deepseek', 'deepseek-reasoner')])
def test_reasoning_probe_accepts_preserved_chat_reasoning(runner, monkeypatch, provider, model):
    client = httpx.Client
    payloads = []
    def respond(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        results = [m['content'] for m in payload['messages'] if m['role'] == 'tool']
        message = {'content': results[-1]} if results else {'reasoning_content': 'PRIVATE_THINKING', 'tool_calls': [{
            'id': 'fixture', 'type': 'function', 'function': {'name': 'lookup_fixture', 'arguments': '{"label":"check"}'}}]}
        return httpx.Response(200, json={'choices': [{'message': message}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}})
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    probe = runner.Probe(model, 'fake', provider)
    result = runner.run_case('reasoning_tools', lambda: runner.reasoning_tools(probe))
    assert result['status'] == 'passed' and result['reasoning_preserved'] and len(payloads) == 2
    assert 'PRIVATE_THINKING' not in json.dumps(result)


def test_case_timeout_kills_process_group_and_reaps_child(runner, monkeypatch, tmp_path):
    events = []
    class Process:
        pid = 12345
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def wait(self, timeout=None):
            events.append(('wait', timeout))
            if timeout:
                raise subprocess.TimeoutExpired('fixture', timeout)
            return -9
        def kill(self):
            events.append('kill')
    def launch(command, **kwargs):
        assert kwargs['stdout'] == kwargs['stderr'] == subprocess.DEVNULL
        assert kwargs['start_new_session'] is (runner.os.name == 'posix')
        return Process()
    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    if runner.os.name == 'posix':
        monkeypatch.setattr(runner.os, 'killpg', lambda pid, sig: events.append(('killpg', pid)))
    with pytest.raises(subprocess.TimeoutExpired):
        runner.case_process(['fixture'], tmp_path, {}, 1)
    assert events == [('wait', 1), ('killpg', 12345) if runner.os.name == 'posix' else 'kill', ('wait', None)]
