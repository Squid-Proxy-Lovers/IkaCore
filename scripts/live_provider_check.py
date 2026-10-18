#!/usr/bin/env python3
"""Bounded live provider checks using synthetic prompts and harmless local tools.

Run with an installed IkaCore wheel. Credentials and model output are never written
into the report. Local-login access requires --use-local-login explicitly.
"""

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
from live_provider_config import CASES, PROVIDERS, clean_usage, safe_details, validate_model

from IkaCore import (
    ExecutionLimits,
    ExecutionPolicy,
    IkaBaseAgent,
    IkaStage,
    OutputConfig,
    ProviderConfig,
    RuntimeControl,
)
from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_interface.chat_interface import async_chat, chat, init_message_history
from IkaModel.codex.auth import get_bearer
from IkaModel.request_control import request_controls
from IkaModel.request_interface import async_api_request_retry
from IkaModel.runtime_errors import IkaRequestCancelled, IkaRequestDeadlineExceeded


class Probe:
    def __init__(self, model_id, bearer, provider='codex'):
        self.model_id, self.bearer = model_id, bearer
        self.provider = provider
        self.config = PROVIDERS[provider]
        self.url = self.config['url'].format(model=model_id)
        self.reasoning = 'low' if provider in {'codex', 'openai-responses', 'openai-chat'} and (
            model_id.startswith(('gpt-5', 'o1', 'o3', 'o4'))) else None

    def model(self):
        model = BareBoneModel(self.model_id, self.bearer, self.url,
            system_prompt='Follow the test instructions exactly. Keep answers brief.',
            suppress_init_output=True, reasoning_effort=self.reasoning, max_tokens=512,
            use_responses_api=self.provider == 'openai-responses')
        return model

    def agent(self, prompt, **kwargs):
        name = kwargs.pop('name', 'live_fixture')
        return IkaBaseAgent(name, 'Synthetic integration test', prompt,
            model_id=self.model_id, api_key=self.bearer,
            provider=ProviderConfig(api_url=self.url, reasoning_effort=self.reasoning,
                max_tokens=512, use_responses_api=self.provider == 'openai-responses'),
            limits=ExecutionLimits(maxsteps=3, max_tool_rounds=3, max_tool_calls=3),
            policy=ExecutionPolicy(max_step_extensions=0, max_stage_extensions=0),
            output=OutputConfig(cli=False, summarize_final=False, show_usage_level0=False), **kwargs)

    def text(self, asynchronous=False):
        model = self.model()
        messages = [{'role': 'user', 'content': 'Reply with exactly OK.'}]
        async def run():
            async with httpx.AsyncClient() as client:
                result = await async_chat(model, messages, init_message_history(), timeout=45, client=client)
                assert not client.is_closed
                return result
        result = asyncio.run(run()) if asynchronous else chat(model, messages, init_message_history(), timeout=45)
        assert result['content'].strip().upper() == 'OK'
        assert result['usage']['total_tokens'] > 0
        return {'usage_reported': True, 'usage': result['usage']}

    def tool_round_trip(self, asynchronous=False, thinking=False):
        model = self.model()
        model.deepthinking = thinking
        sentinel = 'fixture_' + uuid4().hex[:12]
        executions = []
        def lookup(arguments):
            assert arguments == {'label': 'check'}
            executions.append(True)
            return sentinel
        model.agent_tools = [AgentTool('lookup', 'lookup_fixture', 'Return the test fixture value.',
            ToolArgs('object', 'Arguments', properties={'label': {'type': 'string'}}, required=['label']),
            required=not thinking, limit_calls=1)]
        messages = [{'role': 'user', 'content': 'Call lookup_fixture with label check. Then reply with only the returned value.'}]
        kwargs = {'tool_executors': {'lookup_fixture': lookup}, 'timeout': 45, 'max_tool_rounds': 3, 'max_tool_calls': 2}
        async def run():
            async with httpx.AsyncClient() as client:
                result = await async_chat(model, messages, init_message_history(), client=client, **kwargs)
                assert not client.is_closed
                return result
        result = asyncio.run(run()) if asynchronous else chat(model, messages, init_message_history(), **kwargs)
        assert executions == [True]
        assert result['content'].strip() == sentinel
        assert result['executed_tool_calls'] and result['usage']['total_tokens'] > 0
        return {'tool_executions': len(executions), 'tool_result_replayed': True, 'usage': result['usage']}

    def simple_agent(self):
        agent = self.agent('Call agent_end with input exactly SIMPLE_OK immediately.')
        try:
            result = agent.execution()
            assert result['final_message'] == 'SIMPLE_OK'
            assert result['usage']['total_tokens'] > 0
        finally:
            agent.shutdown()
        return {'completed': True, 'usage': result['usage']}

    def staged_agent(self):
        stages = [IkaStage('first', 'Call stage_end immediately to advance to the second stage. Do not call agent_end.', [], stage_max_step=2),
                  IkaStage('second', 'Call agent_end with input exactly STAGED_OK immediately.', [], stage_max_step=2)]
        agent = self.agent('Follow each stage instruction exactly.', Stages=stages)
        with tempfile.TemporaryDirectory() as directory:
            store = agent.enable_snapshots(str(Path(directory) / 'stages.db'))
            try:
                result = agent.execution()
                assert result['final_message'] == 'STAGED_OK'
                names = [f['frame_name'] for f in store.list_frames(result['runtime']['run_id']) if f['frame_type'] == 'stage']
                assert names == ['first', 'second']
            finally:
                agent.shutdown()
        return {'stages_executed': 2, 'usage': result['usage']}

    def fresh_process_resume(self):
        agent = self.agent('Call agent_end with input exactly RESUME_OK immediately.')
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'resume.db')
            agent.enable_snapshots(path, RuntimeControl(pause_points={'post_model'}))
            try:
                paused = agent.execution()
                assert paused['status'] == 'paused' and paused['pause_point'] == 'post_model'
                usage = dict(agent._total_usage)
            finally:
                agent.shutdown()
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--resume-child'],
                input=json.dumps({'path': path, 'snapshot_id': paused['snapshot_id'], 'model': self.model_id, 'provider': self.provider}),
                capture_output=True, text=True, timeout=30, check=False)
            assert process.returncode == 0
            resumed = json.loads(process.stdout)
            assert resumed['content_matches'] and resumed['usage'] == usage
        return {'fresh_process': True, 'network_blocked_during_resume': True, 'usage_preserved': True, 'usage': usage}

    def async_cancellation(self):
        async def run():
            headers_received = asyncio.Event()
            callbacks = []
            responses = []
            cancelled = False
            async def on_response(response):
                responses.append(response)
                headers_received.set()
                await asyncio.Event().wait()
            def register(callback):
                callbacks.append(callback)
                return lambda: callbacks.remove(callback)
            model = self.model()
            from IkaModel.chat_helpers_common import build_provider_request
            url, headers, payload = build_provider_request(self.config['adapter'], model,
                [{'role': 'user', 'content': 'Reply with exactly OK.'}], init_message_history())
            async with httpx.AsyncClient(event_hooks={'response': [on_response]}) as client:
                with request_controls(cancel_checker=lambda: cancelled, abort_registrar=register):
                    task = asyncio.create_task(async_api_request_retry(url, headers, payload, client=client))
                    try:
                        await asyncio.wait_for(headers_received.wait(), timeout=45)
                        cancelled = True
                        callbacks[0]()
                        try:
                            await task
                        except IkaRequestCancelled:
                            pass
                        else:
                            raise AssertionError('request was not cancelled')
                        assert not client.is_closed and responses[0].is_closed and not callbacks
                    finally:
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
        asyncio.run(run())
        return {'live_response_closed': True, 'caller_client_open': True}


def resume_child():
    data = json.load(sys.stdin)
    def deny_network(*args, **kwargs):
        raise RuntimeError('resume attempted an unexpected network request')
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    # Resuming a saved completed model turn must not need a credential or an API call.
    probe = Probe(data['model'], 'offline-resume-fixture', data['provider'])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        agent = probe.agent('Call agent_end with input exactly RESUME_OK immediately.')
        agent.enable_snapshots(data['path'])
        try:
            result = agent.resume_from_snapshot(data['snapshot_id'])
        finally:
            agent.shutdown()
    print(json.dumps({'content_matches': result['final_message'] == 'RESUME_OK', 'usage': result['usage']}))


def runtime_fingerprint():
    import IkaCore
    import IkaMem
    import IkaModel
    digest = hashlib.sha256()
    for package in (IkaCore, IkaModel, IkaMem):
        root = Path(package.__file__).parent
        for path in sorted(root.rglob('*.py')):
            digest.update((package.__name__ + '/' + path.relative_to(root).as_posix()).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()



class SkipCase(Exception):
    pass


def run_case(name, operation, timeout=60):
    started = time.monotonic()
    result = {'case': name, 'status': 'failed'}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            with request_controls(timeout=timeout, max_retries=1):
                result.update(safe_details(operation()))
            result['status'] = 'passed'
        except SkipCase as error:
            result.update(safe_details({'reason': str(error)}))
            result['status'] = 'skipped'
        except Exception as error:
            result.update(safe_details({'error_type': type(error).__name__,
                                        'check': error.args[0] if error.args else None}))
            result.update(safe_details(getattr(error, 'probe_details', {})))
            result.setdefault('error_type', 'RuntimeError')
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            match = re.search(r'returned (\d{3})', str(error))
            result.update(safe_details({'http_status': status if status is not None else int(match.group(1)) if match else None}))
    result.setdefault('usage', {})
    result['seconds'] = round(time.monotonic() - started, 2)
    return result


def reasoning_tools(probe):
    """Observe real parsed reasoning and require it in the next provider payload."""
    from IkaModel.chat_interface import chat_request
    if probe.provider == 'anthropic':
        raise SkipCase('runtime_no_extended_thinking')
    if probe.provider in {'gemini', 'openrouter'} or (probe.provider.startswith('openai') and not probe.reasoning):
        raise SkipCase('model_not_reasoning')
    if probe.provider == 'deepseek' and 'reasoner' not in probe.model_id.lower():
        raise SkipCase('model_not_reasoning')
    build, parse = chat_request.build_provider_request, chat_request.parse_provider_response
    observed, replayed = [], []
    def capture_parse(*args, **kwargs):
        parsed = parse(*args, **kwargs)
        if parsed[1] and parsed[2]:
            observed.append(parsed[1])
        return parsed
    def capture_build(*args, **kwargs):
        built = build(*args, **kwargs)
        if observed:
            # Only exact values in serialized request payload count, never output them.
            encoded = json.dumps(built[2], ensure_ascii=False)
            replayed.append(all(json.dumps(value, ensure_ascii=False)[1:-1] in encoded for value in observed))
        return built
    with patch.object(chat_request, 'build_provider_request', capture_build), patch.object(
            chat_request, 'parse_provider_response', capture_parse):
        result = probe.tool_round_trip(thinking=True)
    if not observed:
        raise SkipCase('reasoning_not_exposed')
    assert replayed and all(replayed), 'reasoning_replay'
    return {**result, 'reasoning_preserved': True}


def streaming_sse(probe):
    if probe.provider != 'codex':
        raise SkipCase('runtime_no_sse')
    # Observe the actual SSE collector, not a content-type supplied by a proxy.
    from IkaModel.codex import chat_helpers_codex
    original = chat_helpers_codex._collect_stream
    seen = []
    def observe(response):
        result = original(response)
        seen.append(bool(result.get('output')))
        return result
    with patch.object(chat_helpers_codex, '_collect_stream', observe):
        result = probe.text()
    assert seen and all(seen)
    return {**result, 'sse_parsed': True}



def usage_accounting(probe):
    result = probe.text()
    usage = clean_usage(result['usage'])
    assert all(key in usage for key in ('input_tokens', 'output_tokens', 'total_tokens'))
    assert usage['total_tokens'] >= usage['input_tokens'] + usage['output_tokens'] > 0
    assert usage.get('input_cached_tokens', 0) <= usage['input_tokens']
    return result

def deadline(probe):
    # Expired budget must fail before contacting the live provider. No timing race/bill.
    with request_controls(timeout=0):
        try:
            chat(probe.model(), [{'role': 'user', 'content': 'Reply OK.'}], init_message_history())
        except IkaRequestDeadlineExceeded:
            return {'deadline_enforced': True}
    raise AssertionError('deadline ignored')


def transient_retry(probe):
    from IkaModel.request_interface import api_request_retry
    attempts = []
    def respond(request):
        attempts.append(True)
        if len(attempts) == 1:
            return httpx.Response(503, headers={'Retry-After': '0'}, json={'error': {'type': 'server_error'}})
        if probe.provider == 'codex':
            event = {'type': 'response.completed', 'response': {'status': 'completed', 'output': []}}
            return httpx.Response(200, headers={'Content-Type': 'text/event-stream'},
                                  text='event: response.completed\ndata: ' + json.dumps(event) + '\n\n')
        return httpx.Response(200, json={'fixture': True})
    # Explicit exception to one attempt, restricted to a transport that cannot use network.
    with httpx.Client(transport=httpx.MockTransport(respond)) as client, request_controls(max_retries=2):
        response = api_request_retry(probe.url, {}, {}, client=client, wait_seconds=0)
    assert response.status_code == 200 and len(attempts) == 2
    return {'synthetic_transport': True, 'attempts': 2}


def compaction(probe):
    from IkaModel import runtime_options
    sentinel = 'compact_' + uuid4().hex[:12]
    agent = probe.agent('Call agent_end with input exactly the synthetic fixture value from earlier context.')
    agent.context_budget = 256
    agent.message_history['messages']['fixture'] = {
        'type': 'assistant', 'message': 'The synthetic fixture value is ' + sentinel + '.', 'tokens': 180}
    with tempfile.TemporaryDirectory() as directory:
        store = agent.enable_snapshots(str(Path(directory) / 'compact.db'))
        try:
            with runtime_options(compact_at_fraction=0.6):
                result = agent.execution()
            assert result['final_message'] == sentinel, 'compaction_answer'
            assert agent.message_history['summary']['message'].startswith('[SUMMARY]'), 'compaction_summary'
            snapshots = store.list_snapshots(result['runtime']['run_id'])
            count = sum(row['snapshot_kind'] == 'post_compaction' for row in snapshots)
            assert count >= 1, 'compaction_snapshot'
        finally:
            agent.shutdown()
    return {'compacted': True, 'compactions': count, 'coherent_answer': True, 'usage': result['usage']}


def workflow_hitl(probe):
    from IkaCore.workflow import IkaWorkflow, WorkflowEdge, WorkflowNode
    from IkaModel.chat_interface import chat_request
    sentinel = 'upstream_' + uuid4().hex[:12]
    first = probe.agent('Follow the approval stage.', name='approval', Stages=[IkaStage(
        'approve', 'If no approval was supplied, call ask_user with question exactly Approve? '
        'When the user says YES, call agent_end with input exactly ' + sentinel + '.', [], hitl=True, stage_max_step=2)])
    second = probe.agent('Find SYNTHETIC_UPSTREAM_VALUE in the upstream context. Call agent_end with input '
                         'exactly its value, without quotes, punctuation or explanation. '
                         'Use MISSING only if that named field is absent.', name='final')
    workflow = IkaWorkflow('live_flow', 'Synthetic dependency + HITL',
        [WorkflowNode('approval', first), WorkflowNode('final', second)], [WorkflowEdge('approval', 'final')],
        compress_hook=lambda contexts, agent: json.dumps({'SYNTHETIC_UPSTREAM_VALUE': contexts[-1]}))
    observed = []
    original = chat_request.build_provider_request
    def observe(*args, **kwargs):
        built = original(*args, **kwargs)
        if args[1].agent_name == 'final':
            observed.append(sentinel in json.dumps(built[2]))
        return built
    with tempfile.TemporaryDirectory() as directory, patch.object(chat_request, 'build_provider_request', observe):
        workflow.enable_snapshots(str(Path(directory) / 'workflow.db'))
        try:
            assert workflow.run() == {}, 'workflow_initial_pause'
            info = workflow.snapshot_runtime_info()
            assert list(info['interrupted_instances']) == ['approval:0'], 'workflow_interrupted_instance'
            result = workflow.resume_from_snapshot(info['last_snapshot_id'], resume_input='YES')
            assert result['approval'].final == sentinel, 'workflow_upstream_answer'
            assert observed and all(observed), 'workflow_dependency_context'
            answer = result['final'].final
            if answer != sentinel:
                error = AssertionError('workflow_dependency_answer')
                error.probe_details = {'dependency_answer_missing': answer == 'MISSING',
                    'dependency_answer_contains_fixture': sentinel in answer, 'dependency_answer_empty': not answer}
                raise error
            assert not workflow.snapshot_runtime_info()['interrupted_instances'], 'workflow_resumed'
        finally:
            workflow.async_executor.shutdown()
            first.shutdown()
            second.shutdown()
    return {'dependency_preserved': True, 'hitl_resumed': True}


def operations(probe):
    return {'sync_text': probe.text, 'async_text': lambda: probe.text(True),
            'sync_tools': probe.tool_round_trip, 'async_tools': lambda: probe.tool_round_trip(True),
            'simple_agent': probe.simple_agent, 'staged_agent': probe.staged_agent,
            'fresh_process_resume': probe.fresh_process_resume, 'async_cancellation': probe.async_cancellation,
            'reasoning_tools': lambda: reasoning_tools(probe), 'streaming_sse': lambda: streaming_sse(probe),
            'usage_accounting': lambda: usage_accounting(probe), 'deadline': lambda: deadline(probe),
            'transient_retry': lambda: transient_retry(probe), 'compaction': lambda: compaction(probe),
            'workflow_hitl': lambda: workflow_hitl(probe)}


@contextlib.contextmanager
def observe_usage(probe):
    """Count completed provider responses, including separate compaction requests."""
    from IkaModel import request_interface, summarization
    from IkaModel.chat_interface.response_usage import extract_usage
    totals = dict.fromkeys(('input_tokens', 'output_tokens', 'total_tokens', 'input_cached_tokens'), 0)
    original_sync, original_async = request_interface.api_request_retry, request_interface.async_api_request_retry
    def record(response, url):
        try:
            from IkaModel.request_interface import get_provider
            adapter = get_provider(probe.model_id, url, probe.provider == 'openai-responses')
            for key, count in clean_usage(extract_usage(adapter, response.json())).items():
                totals[key] += count
        except (ValueError, TypeError, KeyError):
            pass  # Cancelled/incomplete responses have no accountable body.
        return response
    def sync(url, *args, **kwargs):
        return record(original_sync(url, *args, **kwargs), url)
    async def asynchronous(url, *args, **kwargs):
        return record(await original_async(url, *args, **kwargs), url)
    with patch.object(request_interface, 'api_request_retry', sync), patch.object(
            request_interface, 'async_api_request_retry', asynchronous), patch.object(
            summarization, 'api_request_retry', sync), patch.object(summarization, 'async_api_request_retry', asynchronous):
        yield totals


def installed_identity():
    import IkaCore
    distribution = importlib.metadata.distribution('ikacore')
    direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    root = Path(IkaCore.__file__).resolve()
    if direct.get('dir_info', {}).get('editable') or 'site-packages' not in root.parts:
        raise ValueError('run via scripts/live_matrix.py: an installed wheel is required')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, check=True).stdout.strip()
    assert re.fullmatch('[0-9a-f]{40}', revision)
    return {'package_version': distribution.version, 'git_revision': revision,
            'runtime_sha256': runtime_fingerprint(), 'installation': 'wheel', 'python': sys.version.split()[0]}


def child_case(args):
    probe = Probe(args.model, os.environ[PROVIDERS[args.provider[0]]['key']], args.provider[0])
    with observe_usage(probe) as usage:
        result = run_case(args.case[0], operations(probe)[args.case[0]], args.case_timeout)
    # Observed usage includes compaction and model turns that fail subsequent assertions.
    result['usage'] = clean_usage(usage)
    result['usage_complete'] = result['status'] == 'passed' and args.case[0] != 'async_cancellation'
    args.child_output.write_text(json.dumps(result) + '\n')
    return 0



def case_process(command, directory, env, timeout):
    # A resume child belongs to the same process group, so hard timeouts stop it too.
    with subprocess.Popen(command, cwd=directory, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, start_new_session=os.name == 'posix') as process:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            raise

def isolated_case(provider, model, name, timeout, env):
    from live_provider_config import safe_seconds
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='ika-live-case-') as directory:
        output = Path(directory) / 'result.json'
        command = [sys.executable, str(Path(__file__).resolve()), '--provider', provider, '--model', model,
                   '--case', name, '--case-timeout', str(timeout), '--child-output', str(output)]
        try:
            returncode = case_process(command, directory, env, timeout)
            if returncode or not output.exists():
                raise ChildProcessError('case process failed')
            raw = json.loads(output.read_text())
            result = {'case': name, 'status': raw['status'] if raw.get('status') in {'passed', 'failed', 'skipped'} else 'failed',
                      **safe_details(raw)}
            result['usage_complete'] = raw.get('usage_complete') is True
        except (subprocess.TimeoutExpired, ChildProcessError, ValueError, KeyError) as error:
            result = {'case': name, 'status': 'failed', 'error_type': type(error).__name__,
                      'usage': {}, 'usage_complete': False}
    result['seconds'] = safe_seconds(time.monotonic() - started)
    return result


def select_cases(requested):
    selected = list(dict.fromkeys(requested or CASES))
    if any(name not in CASES for name in selected):
        raise ValueError('unknown case')
    return selected


def provider_report(provider, model, identity, requested, env, timeout, execute=isolated_case, progress=None):
    config = PROVIDERS[provider]
    report = {'schema_version': 1, 'provider': provider, 'model': model,
              'endpoint': config['url'].format(model=model or ''), 'runtime': identity,
              'timestamp': datetime.now(timezone.utc).isoformat(), 'case_deadline_seconds': timeout,
              'live_attempts_per_request': 1, 'cases_requested': requested, 'results': []}
    reason = 'not_configured' if not env.get(config['key']) else 'model_required' if not model else None
    for name in requested:
        row = {'case': name, 'status': 'skipped', 'reason': reason, 'seconds': 0, 'usage': {}, 'usage_complete': False} if reason else execute(
            provider, model, name, timeout, env)
        report['results'].append(row)
        if progress:
            progress(report, row)
        if row['status'] == 'failed' and name in {'sync_text', 'async_text'}:
            reason = 'connectivity_failed'
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    group = result.add_mutually_exclusive_group()
    group.add_argument('--provider', action='append', choices=PROVIDERS)
    group.add_argument('--all-configured', action='store_true')
    result.add_argument('--model', help='model for a single selected provider')
    result.add_argument('--models', default='{}', help='JSON object mapping provider IDs to model IDs')
    result.add_argument('--use-local-login', action='store_true')
    result.add_argument('--case', action='append', choices=CASES)
    result.add_argument('--case-timeout', type=int, choices=range(1, 121), default=60, metavar='1..120')
    result.add_argument('--output-dir', type=Path, default=Path('artifacts/live'))
    result.add_argument('--output', type=Path, help='legacy single-provider report path')
    result.add_argument('--child-output', type=Path, help=argparse.SUPPRESS)
    result.add_argument('--resume-child', action='store_true', help=argparse.SUPPRESS)
    return result


def prepare_environment():
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'IKA_DUMP_REQUESTS', 'CODEX_REFRESH_TOKEN_URL_OVERRIDE',
                 'IKACORE_ALLOW_CODEX_TOKEN_URL_OVERRIDE', 'GLM_REASONING_LOG', 'SSLKEYLOGFILE'):
        env.pop(name, None)
        os.environ.pop(name, None)
    return env


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    logging.disable(logging.CRITICAL)
    env = prepare_environment()
    if args.resume_child:
        resume_child()
        return 0
    if args.child_output:
        return child_case(args)
    providers = list(dict.fromkeys(args.provider or PROVIDERS))
    if args.model and len(providers) != 1:
        cli.error('--model requires exactly one --provider; use --models or per-provider model env vars')
    if args.output and len(providers) != 1:
        cli.error('--output requires exactly one --provider')
    try:
        models = json.loads(args.models)
        if not isinstance(models, dict) or any(key not in PROVIDERS for key in models):
            raise ValueError('invalid model mapping')
        resolved = {provider: validate_model(value, env) if value else None for provider in providers
                    for value in [args.model or models.get(provider) or env.get(PROVIDERS[provider]['model_env'])]}
        identity = installed_identity()
    except (ValueError, AssertionError):
        cli.error('invalid model configuration or runtime is not an installed wheel; use scripts/live_matrix.py')
    if args.use_local_login and 'codex' in providers and not env.get('CODEX_BEARER'):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                env['CODEX_BEARER'] = get_bearer()
        except Exception:
            cli.error('local Codex login failed; no credential details recorded')
    failed = False
    for provider in providers:
        path = args.output or args.output_dir / (provider + '.json')
        def progress(report, row, path=path, provider=provider):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + '\n')
            if row.get('reason') not in {'not_configured', 'model_required'}:
                print(f"{provider}/{row['case']}: {row['status']} ({row['seconds']}s)", flush=True)
        report = provider_report(provider, resolved[provider], identity, select_cases(args.case), env,
                                 args.case_timeout, progress=progress)
        statuses = [row['status'] for row in report['results']]
        reason = report['results'][0].get('reason')
        label = 'not configured' if reason == 'not_configured' else 'model required' if reason == 'model_required' else (
            f"{statuses.count('passed')} passed, {statuses.count('failed')} failed, {statuses.count('skipped')} skipped")
        print(f'{provider}: {label}', flush=True)
        failed |= 'failed' in statuses or reason == 'model_required'
    from live_summary import write_summary
    write_summary(args.output.parent if args.output else args.output_dir)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
