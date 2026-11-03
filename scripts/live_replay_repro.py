#!/usr/bin/env python3
"""Offline regression probe for complete reasoning replay; prints booleans only.

No credentials or network are used. Each provider must preserve its synthetic
reasoning through parsing, assistant/tool history, and the next request.
"""

import json

from live_provider_config import PROVIDERS

from IkaModel.base import BareBoneModel
from IkaModel.chat_helpers_common import (
    append_provider_tool_messages,
    build_provider_request,
    parse_provider_response,
)
from IkaModel.chat_interface.chat_interface import init_message_history
from IkaModel.chat_interface.tool_result_formatters import (
    format_gemini_results,
    format_provider_tool_results,
)

MARKER = 'synthetic_private_reasoning_fixture'
OPAQUE = 'synthetic_opaque_provider_state'


def fixture_response(provider):
    call = {'id': 'call_fixture', 'type': 'function',
            'function': {'name': 'lookup_fixture', 'arguments': '{}'}}
    if provider == 'anthropic':
        return {'content': [{'type': 'thinking', 'thinking': MARKER, 'signature': OPAQUE},
                            {'type': 'redacted_thinking', 'data': OPAQUE},
                            {'type': 'tool_use', 'id': 'call_fixture', 'name': 'lookup_fixture', 'input': {}}]}
    if provider == 'gemini':
        return {'candidates': [{'content': {'role': 'model', 'parts': [
            {'text': MARKER, 'thought': True, 'thoughtSignature': OPAQUE},
            {'functionCall': {'name': 'lookup_fixture', 'args': {}}, 'thoughtSignature': OPAQUE},
        ]}}]}
    if provider in {'zai', 'deepseek', 'openrouter'}:
        fields = {'reasoning_content': MARKER}
        if provider == 'openrouter':
            fields = {'reasoning': MARKER, 'reasoning_details': [
                {'type': 'reasoning.encrypted', 'data': OPAQUE, 'index': 0, 'id': 'rs_fixture'}]}
        return {'choices': [{'message': {'content': '', **fields, 'tool_calls': [call]}}]}
    return {'output': [{'type': 'reasoning', 'id': 'reason_fixture', 'encrypted_content': OPAQUE,
                        'summary': [{'type': 'summary_text', 'text': MARKER}]},
                       {'type': 'function_call', 'call_id': 'call_fixture',
                        'name': 'lookup_fixture', 'arguments': '{}'}]}


def replay_evidence(provider):
    config = PROVIDERS[provider]
    ids = {'anthropic': 'claude-sonnet-4-5', 'gemini': 'gemini-2.5-pro', 'zai': 'glm-4.7',
           'deepseek': 'deepseek-reasoner', 'openrouter': 'anthropic/claude-sonnet-4.5'}
    model = BareBoneModel(ids.get(provider, 'gpt-5.6-sol'), 'offline-fixture', config['url'],
                          use_responses_api=provider == 'openai-responses', max_tokens=4096,
                          thinking_budget_tokens=1024 if provider == 'anthropic' else None)
    model.deepthinking = True
    adapter = config['adapter']
    messages, history = [{'role': 'user', 'content': 'Use the synthetic tool.'}], init_message_history()
    content, reasoning, calls, tokens = parse_provider_response(adapter, fixture_response(provider), model.model_id, history)
    tools = format_provider_tool_results(adapter, calls, ['synthetic-result'])
    append_provider_tool_messages(adapter, messages, history, content, reasoning, calls, tools,
                                  ['synthetic-result'], tokens, format_gemini_results_fn=format_gemini_results)
    _, _, payload = build_provider_request(adapter, model, messages, history)
    encoded = json.dumps(payload)
    return {'provider': provider, 'reasoning_parsed': reasoning == MARKER,
            'reasoning_replayed': MARKER in encoded,
            'opaque_replayed': OPAQUE in encoded if provider not in {'deepseek', 'zai'} else True,
            'thinking_enabled': bool(payload.get('thinking'))}


if __name__ == '__main__':
    results = [replay_evidence(provider) for provider in
               ('codex', 'openai-responses', 'anthropic', 'deepseek', 'gemini', 'openrouter', 'zai')]
    for result in results:
        print(json.dumps(result))
    raise SystemExit(0 if all(row['reasoning_parsed'] and row['reasoning_replayed'] and row['opaque_replayed']
                              for row in results) else 1)
