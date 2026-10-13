#!/usr/bin/env python3
"""Bounded live Codex checks using synthetic prompts and harmless local tools.

Run with an installed IkaCore wheel. Credentials and model output are never written
into the report. Local-login access requires --use-local-login explicitly.
"""

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

from IkaCore import IkaBaseAgent, IkaStage, RuntimeControl
from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
from IkaModel.chat_interface.chat_interface import async_chat, chat, init_message_history
from IkaModel.codex.auth import get_bearer
from IkaModel.codex_constants import CODEX_API_URL
from IkaModel.request_control import request_controls
from IkaModel.request_interface import async_api_request_retry
from IkaModel.runtime_errors import IkaRequestCancelled


class Probe:
    def __init__(self, model_id, bearer):
        self.model_id, self.bearer = model_id, bearer

    def model(self):
        return BareBoneModel(self.model_id, self.bearer, CODEX_API_URL,
                             system_prompt='Follow the test instructions exactly. Keep answers brief.',
                             suppress_init_output=True, reasoning_effort='low')

    def agent(self, prompt, **kwargs):
        return IkaBaseAgent('live_fixture', 'Synthetic integration test', prompt,
                            model_id=self.model_id, api_key=self.bearer, api_url=CODEX_API_URL,
                            reasoning_effort='low', maxsteps=3, max_tool_rounds=3, max_tool_calls=3,
                            max_step_extensions=0, max_stage_extensions=0,
                            summarize_final=False, show_usage_level0=False, **kwargs)

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
        return {'usage_reported': True}

    def tool_round_trip(self, asynchronous=False):
        model = self.model()
        sentinel = 'fixture_' + uuid4().hex[:12]
        executions = []
        def lookup(arguments):
            assert arguments == {'label': 'check'}
            executions.append(True)
            return sentinel
        model.agent_tools = [AgentTool('lookup', 'lookup_fixture', 'Return the test fixture value.',
            ToolArgs('object', 'Arguments', properties={'label': {'type': 'string'}}, required=['label']),
            required=True, limit_calls=1)]
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
        return {'tool_executions': len(executions), 'tool_result_replayed': True}

    def simple_agent(self):
        agent = self.agent('Call agent_end with input exactly SIMPLE_OK immediately.')
        try:
            result = agent.execution()
            assert result['final_message'] == 'SIMPLE_OK'
            assert result['usage']['total_tokens'] > 0
        finally:
            agent.shutdown()
        return {'completed': True}

    def staged_agent(self):
        stages = [IkaStage('first', 'Call stage_end immediately to advance to the second stage. Do not call agent_end.', []),
                  IkaStage('second', 'Call agent_end with input exactly STAGED_OK immediately.', [])]
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
        return {'stages_executed': 2}

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
                input=json.dumps({'path': path, 'snapshot_id': paused['snapshot_id'], 'model': self.model_id}),
                capture_output=True, text=True, timeout=30, check=False)
            assert process.returncode == 0
            resumed = json.loads(process.stdout)
            assert resumed['content_matches'] and resumed['usage'] == usage
        return {'fresh_process': True, 'network_blocked_during_resume': True, 'usage_preserved': True}

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
            url, headers, payload = build_provider_request('codex', model,
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
    probe = Probe(data['model'], 'offline-resume-fixture')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        agent = probe.agent('Call agent_end with input exactly RESUME_OK immediately.')
        agent.enable_snapshots(data['path'])
        try:
            result = agent.resume_from_snapshot(data['snapshot_id'])
        finally:
            agent.shutdown()
    print(json.dumps({'content_matches': result['final_message'] == 'RESUME_OK', 'usage': result['usage']}))


def run_case(name, operation):
    started = time.monotonic()
    result = {'case': name, 'status': 'failed'}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            with request_controls(timeout=60, max_retries=1):
                result.update(operation())
            result['status'] = 'passed'
        except Exception as error:
            result['error_type'] = type(error).__name__
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            match = re.search(r'returned (\d{3})', str(error))
            if status is not None or match:
                result['http_status'] = status if status is not None else int(match.group(1))
    result['seconds'] = round(time.monotonic() - started, 2)
    return result


def runtime_fingerprint():
    import IkaCore
    import IkaModel
    digest = hashlib.sha256()
    for package in (IkaCore, IkaModel):
        root = Path(package.__file__).parent
        for path in sorted(root.rglob('*.py')):
            digest.update((package.__name__ + '/' + path.relative_to(root).as_posix()).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--use-local-login', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts/live/codex.json'))
    parser.add_argument('--case', action='append', choices=['sync_text', 'async_text', 'sync_tools', 'async_tools',
        'simple_agent', 'staged_agent', 'fresh_process_resume', 'async_cancellation'])
    parser.add_argument('--resume-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    for name in ('IKA_DUMP_REQUESTS', 'CODEX_REFRESH_TOKEN_URL_OVERRIDE', 'IKACORE_ALLOW_CODEX_TOKEN_URL_OVERRIDE'):
        os.environ.pop(name, None)
    if args.resume_child:
        resume_child()
        return 0
    if not args.model:
        parser.error('--model is required')
    bearer = os.getenv('CODEX_BEARER')
    if not bearer and args.use_local_login:
        try:
            bearer = get_bearer()
        except Exception as error:
            parser.error('local login failed: ' + type(error).__name__)
    if not bearer:
        parser.error('set CODEX_BEARER or explicitly pass --use-local-login')
    probe = Probe(args.model, bearer)
    cases = {'sync_text': probe.text, 'async_text': lambda: probe.text(True),
             'sync_tools': probe.tool_round_trip, 'async_tools': lambda: probe.tool_round_trip(True),
             'simple_agent': probe.simple_agent, 'staged_agent': probe.staged_agent,
             'fresh_process_resume': probe.fresh_process_resume, 'async_cancellation': probe.async_cancellation}
    requested = list(args.case or cases)
    report = {'model': args.model, 'endpoint': CODEX_API_URL, 'runtime_sha256': runtime_fingerprint(),
              'timestamp': datetime.now(timezone.utc).isoformat(), 'cases_requested': requested, 'results': []}
    for name in requested:
        result = run_case(name, cases[name])
        report['results'].append(result)
        print(json.dumps(result), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        if result['status'] != 'passed' and name in {'sync_text', 'async_text'}:
            break
    return int(any(row['status'] != 'passed' for row in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
