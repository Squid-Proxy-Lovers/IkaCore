#!/usr/bin/env python3
"""Offline payload/encoding timings for all registered provider routes."""

import argparse
import hashlib
import json
import platform
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROVIDERS = {
    'openai': ('gpt-4o', 'https://api.openai.com/v1/chat/completions'),
    'openai_responses': ('gpt-4o', 'https://api.openai.com/v1/responses'),
    'anthropic': ('claude-sonnet-4-5', 'https://api.anthropic.com/v1/messages'),
    'deepseek': ('deepseek-chat', 'https://api.deepseek.com/chat/completions'),
    'gemini': ('gemini-2.5-flash', 'https://generativelanguage.googleapis.com/v1beta'),
    'openrouter': ('openai/gpt-4o', 'https://openrouter.ai/api/v1/chat/completions'),
    'codex': ('gpt-5.2-codex', 'https://chatgpt.com/backend-api/codex/responses'),
}


def fixture(provider, count, tool_count):
    from IkaModel.base import AgentTool, BareBoneModel, ToolArgs
    name, url = PROVIDERS[provider]
    model = BareBoneModel(name, 'benchmark-placeholder', url, system_prompt='Benchmark instructions',
                          suppress_init_output=True, use_responses_api=provider != 'openai')
    model.agent_tools = [AgentTool(str(i), 'lookup_' + str(i), 'Look up a fixture record',
                                  ToolArgs('object', 'parameters', properties={'query': {'type': 'string'}, '__required__': ['query']}))
                         for i in range(tool_count)]
    messages = [{'role': 'user', 'content': 'task'}]
    messages += [{'role': 'assistant' if i % 2 else 'user', 'content': f'Message {i}: ' + 'x' * 128} for i in range(count)]
    history = {'system': {'message': 'Benchmark instructions', 'tokens': 0},
               'first_input': {'message': 'task', 'tokens': 0}, 'summary': {'message': '', 'tokens': 0}, 'messages': {}}
    return model, messages, history


def measure(function, iterations):
    function()
    samples = []
    for _ in range(7):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - start) / iterations)
    return {'median_us': statistics.median(samples) / 1000, 'samples_ns': samples}


def deny_network(*args, **kwargs):
    raise RuntimeError('provider benchmark attempted a real network connection')


def compare(before, after):
    index = {(r['provider'], r['messages'], r['tools']): r for r in before}
    assert len(index) == len(after), 'benchmark fixture inventory changed'
    results = []
    for row in after:
        previous = index[(row['provider'], row['messages'], row['tools'])]
        assert previous['payload_sha256'] == row['payload_sha256'], 'payload changed: ' + row['provider']
        results.append({'provider': row['provider'], 'messages': row['messages'],
                        'build_ratio': row['build']['median_us'] / previous['build']['median_us']})
    return results


def main():
    from IkaModel.chat_helpers_common import PROVIDER_REQUEST_BUILDERS, build_provider_request
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Compare payload hashes and build timings with a previous report')
    parser.add_argument('--iterations', type=int, default=300)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    assert set(PROVIDERS) == set(PROVIDER_REQUEST_BUILDERS)
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    rows = []
    for provider in PROVIDERS:
        for count, tools in ((10, 0), (100, 10), (1000, 10)):
            model, messages, history = fixture(provider, count, tools)
            def build(provider=provider, model=model, messages=messages, history=history):
                return build_provider_request(provider, model, messages, history)[2]
            payload = build()
            canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
            rows.append({'provider': provider, 'messages': count, 'tools': tools,
                         'build': measure(build, args.iterations),
                         'encode': measure(lambda payload=payload: json.dumps(payload, separators=(',', ':')), args.iterations),
                         'payload_bytes': len(canonical.encode()), 'payload_sha256': hashlib.sha256(canonical.encode()).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'fixture': 'text transcript; no network or model inference', 'rows': rows,
              'python': sys.version, 'platform': platform.platform(), 'iterations': args.iterations,
              'samples': 7, 'timestamp': datetime.now(timezone.utc).isoformat(),
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.baseline:
        report['comparison'] = compare(json.loads(args.baseline.read_text())['rows'], rows)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        print(row['provider'], row['messages'], f"{row['build']['median_us']:.1f} us build", row['payload_bytes'], 'bytes')


if __name__ == '__main__':
    main()
