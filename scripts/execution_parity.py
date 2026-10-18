#!/usr/bin/env python3
"""Emit comparable offline execution traces against dc9efbd and the grouped v2 API.

Run this same script with PYTHONPATH pointing at each source tree, then compare JSON.
Snapshot identities/timestamps are omitted; budgets, calls, results and boundaries remain.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import IkaCore
from IkaCore import IkaBaseAgent, IkaStage, RuntimeControl
from IkaModel.base import AgentEndException

GROUPED = hasattr(IkaCore, 'ExecutionPolicy')


def call(name='agent_end', text='done'):
    args = {'stage_index': 0, 'reason': 'recheck'} if name == 'change_stage' else {'input': text}
    return {'id': 'fixture', 'function': {'name': name, 'arguments': json.dumps(args)}}


def response(content='done', controls=()):
    return {'content': content, 'content_before_tools': content, 'tool_calls': list(controls),
            'executed_tool_calls': list(controls), 'usage': {'input_tokens': 2, 'output_tokens': 1, 'total_tokens': 3},
            'cost': {'total_cost': 0.01}}


def cases():
    basic = ['finish', 'exception', 'requested', 'empty', 'plain', 'transition', 'backtrack', 'structured']
    for name in basic:
        yield name, 4, 0, None
    for steps in (1, 2, 3):
        for extensions in (0, 1, 2):
            yield f'budget-{steps}-{extensions}', steps, extensions, None
    for boundary in ('pre_model', 'post_model', 'step_entry'):
        yield 'pause-' + boundary, 3, 0, boundary
    yield 'human-input', 3, 1, None


def make_agent(staged, maxsteps, extensions, hitl):
    stages = [IkaStage('first', 'prepare', [], stage_max_step=1, hitl=hitl),
              IkaStage('last', 'answer', [], stage_max_step=2, allowed_back_to=[0])] if staged else []
    options = {'maxsteps': maxsteps, 'max_step_extensions': extensions, 'max_stage_extensions': extensions,
               'extend_steps_by': 2, 'extend_stage_steps_by': 2, 'show_usage_level0': False}
    if GROUPED:
        options = {'limits': IkaCore.ExecutionLimits(maxsteps=maxsteps),
                   'policy': IkaCore.ExecutionPolicy(max_step_extensions=extensions, max_stage_extensions=extensions,
                                                     extend_steps_by=2, extend_stage_steps_by=2),
                   'output': IkaCore.OutputConfig(cli=False)}
    return IkaBaseAgent('fixture', 'description', 'task', model_id='gpt-4o', api_key='fixture', Stages=stages, **options)


def run_case(directory, staged, name, maxsteps, extensions, pause):
    agent = make_agent(staged, maxsteps, extensions, name == 'human-input')
    calls, summaries = [], []
    count = 0
    def chat(model, messages, history, **kwargs):
        nonlocal count
        count += 1
        calls.append({'model': model.model_id, 'prompt': model.content_prompt, 'messages': deepcopy(messages),
                      'stage': kwargs.get('current_stage_index'), 'max_tool_calls': kwargs.get('max_tool_calls')})
        end = call()
        if name.startswith('budget') or name == 'plain':
            return response('partial')
        if name == 'empty' and count == 1:
            return response('')
        if name == 'exception':
            raise AgentEndException(response('exception', [end]))
        if name == 'requested' and count == 1:
            result = response('partial', [end])
            result['executed_tool_calls'] = [call('fixture_tool')]
            return result
        if staged and name == 'transition' and count == 1:
            return response('stage done', [call('stage_end')])
        if staged and name == 'backtrack' and count < 4:
            return response('move', [call('change_stage' if count == 2 else 'stage_end')])
        if staged and name == 'human-input' and count == 1:
            return {**response('continue?'), 'interrupted': True, 'status': 'awaiting_user_input',
                    'interrupt_data': {'question': 'continue?'}}
        if name == 'structured':
            return response('{"result": 1}', [call('agent_end', '{"result": 1}')])
        return response('done', [end])
    def summary(*args, **kwargs):
        summaries.append(kwargs.get('prompt_kind'))
        return 'forced final' if kwargs.get('prompt_kind') == 'force_answer' else 'remaining work'
    module = 'IkaCore.agent_policy' if GROUPED else 'IkaCore.agent_runtime_foundation'
    store = agent.enable_snapshots(str(directory / 'runtime.db'), RuntimeControl(pause_points={pause} if pause else set()))
    with patch('IkaCore.agent_chat_support.chat', chat), patch(module + '.run_summarization', summary):
        result = agent.execution()
        if result.get('status') in {'paused', 'awaiting_user_input'}:
            result = agent.resume_from_snapshot(result['snapshot_id'], resume_input='YES' if name == 'human-input' else None)
    run = result['runtime']['run_id']
    snapshots = store.list_snapshots(run)
    boundaries = []
    for snapshot in snapshots:
        state = snapshot['state_json']
        boundaries.append({'kind': snapshot['snapshot_kind'], **{key: state[key] for key in
                           ('scope', 'stage_index', 'remaining_steps', 'maxsteps', 'loop_state') if key in state}})
    agent.shutdown()
    return {'result': {key: value for key, value in result.items() if key != 'runtime'},
            'calls': calls, 'summaries': summaries, 'maxsteps': agent.maxsteps, 'snapshots': boundaries}


def normalize(value):
    if isinstance(value, dict):
        # Snapshot UUID transcript keys carry order, not identity.
        return {key: [normalize(v) for v in item.values()] if key == 'messages' and isinstance(item, dict)
                else normalize(item) for key, item in value.items() if key not in {'snapshot_id', 'runtime'}}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {}
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
        for staged in (False, True):
            for index, (name, maxsteps, extensions, pause) in enumerate(cases()):
                folder = Path(directory) / f'{staged}-{index}'
                folder.mkdir()
                report[f'{"stage" if staged else "simple"}:{name}'] = normalize(
                    run_case(folder, staged, name, maxsteps, extensions, pause))
    assert len(report) == 42
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'{len(report)} execution scenarios: {args.output}')


if __name__ == '__main__':
    main()
