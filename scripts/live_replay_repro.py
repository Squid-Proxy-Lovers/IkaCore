#!/usr/bin/env python3
"""Offline repro of reasoning replay gaps; prints only booleans, uses no credentials.

Run with the development venv. This diagnoses existing runtime behavior and does not
patch it. reasoning_parsed=true followed by reasoning_replayed=false demonstrates
loss between parsing and the next request. Anthropic additionally drops signed blocks.
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

MARKER = 'synthetic_private_reasoning_fixture'


def replay_evidence(provider):
    config = PROVIDERS[provider]
    model = BareBoneModel('glm-4.7' if provider == 'zai' else 'gpt-5.6-sol' if provider == 'codex' else 'gpt-5' if provider != 'anthropic' else 'claude-fixture',
        'offline-fixture', config['url'], use_responses_api=provider == 'openai-responses')
    model.deepthinking = True
    adapter = config['adapter']
    if provider == 'anthropic':
        data = {'content': [{'type': 'thinking', 'thinking': MARKER, 'signature': 'synthetic-signature'},
                            {'type': 'tool_use', 'id': 'call_fixture', 'name': 'lookup_fixture', 'input': {}}]}
    elif provider == 'zai':
        data = {'choices': [{'message': {'reasoning_content': MARKER, 'tool_calls': [{
            'id': 'call_fixture', 'type': 'function', 'function': {'name': 'lookup_fixture', 'arguments': '{}'}}]}}]}
    else:
        data = {'output': [{'type': 'reasoning', 'id': 'reason_fixture',
                            'summary': [{'type': 'summary_text', 'text': MARKER}]},
                           {'type': 'function_call', 'call_id': 'call_fixture',
                            'name': 'lookup_fixture', 'arguments': '{}'}]}
    content, reasoning, calls, tokens = parse_provider_response(adapter, data, model.model_id)
    messages, history = [{'role': 'user', 'content': 'Use the synthetic tool.'}], init_message_history()
    tools = [{'role': 'tool', 'tool_call_id': 'call_fixture', 'content': 'synthetic-result'}]
    append_provider_tool_messages(adapter, messages, history, content, reasoning, calls, tools,
                                  ['synthetic-result'], tokens)
    _, _, payload = build_provider_request(adapter, model, messages, history)
    return {'provider': provider, 'reasoning_parsed': reasoning == MARKER,
            'reasoning_replayed': MARKER in json.dumps(payload),
            'thinking_enabled': bool(payload.get('thinking'))}


if __name__ == '__main__':
    for provider in ('codex', 'openai-responses', 'anthropic', 'zai'):
        print(json.dumps(replay_evidence(provider)))
