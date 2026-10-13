"""Offline benchmark fixtures; no real model or application tools are invoked."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


def setup(case: str, directory: Path):
    import httpx

    from IkaCore.cli_output import set_stdout_enabled
    set_stdout_enabled(False)
    if case == 'agent_construct':
        def construct():
            agent = make_agent()
            agent.shutdown()
        return construct, lambda: None
    if case in {'httpx_roundtrip', 'agent_sync', 'agent_async', 'agent_snapshots', 'agent_soak'}:
        return request_case(case, directory, httpx), lambda: None
    if case == 'payload_100_messages':
        return payload_case(), lambda: None
    if case in {'tool_sync', 'tool_async'}:
        return tool_case(case), lambda: None
    if case.startswith('workflow_'):
        return workflow_case(case)
    if case == 'snapshot_append':
        from IkaCore import SnapshotStore
        store = SnapshotStore(str(directory / 'append.db'))
        run = store.create_run('agent', 'bench')
        frame = store.create_frame(run, 'agent', 'bench')
        return lambda: store.create_snapshot(run, frame, 'boundary', {'content': 'x' * 4096}), lambda: None
    raise ValueError(f'unknown case: {case}')


def make_agent(asynchronous=False):
    from IkaCore import IkaBaseAgent
    return IkaBaseAgent(name='benchmark', description='offline benchmark', prompt='finish', model_id='gpt-4o',
                        api_key='benchmark-placeholder', api_url='https://benchmark.invalid/v1/responses',
                        logging_level=0, show_usage_level0=False, use_async=asynchronous, maxsteps=1,
                        max_step_extensions=0)


def request_case(case, directory, httpx):
    payload = {'id': 'bench', 'output': [{'type': 'function_call', 'call_id': 'end', 'name': 'agent_end',
               'arguments': json.dumps({'input': 'done'})}], 'usage': {'input_tokens': 1, 'output_tokens': 1}}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    def operation():
        if case == 'httpx_roundtrip':
            with httpx.Client(transport=transport) as client:
                result = client.post('https://benchmark.invalid', json={}).json()
                if result['id'] != 'bench':
                    raise AssertionError('invalid mock response')
            return
        agent = make_agent(asynchronous=case == 'agent_async')
        agent.client = (httpx.AsyncClient if case == 'agent_async' else httpx.Client)(transport=transport)
        if case == 'agent_snapshots':
            agent.enable_snapshots(str(directory / 'agents.db'))
        try:
            result = agent.execution()
            if result['final_message'] != 'done':
                raise AssertionError('agent did not complete')
        finally:
            agent.shutdown()
    return operation


def payload_case():
    from types import SimpleNamespace

    from IkaModel.anthropic.claude import anthropic_fill_payload
    model = SimpleNamespace(model_id='claude-sonnet-4-5', max_tokens=100, temperature=0, system_prompt='system',
                            agent_tools=[], parallel_tool_calls=False)
    history = {'system': {'message': 'system'}, 'first_input': {'message': 'task'}, 'summary': {'message': ''},
               'messages': {str(i): {'message': f'Observation {i}: ' + 'x' * 128} for i in range(100)}}
    def operation():
        payload = anthropic_fill_payload(model, [{'role': 'user', 'content': 'task'}], history)
        if len(payload['messages']) != 101:
            raise AssertionError('payload lost history')
    return operation


def tool_case(case):
    from IkaModel.chat_interface.tool_execution_async import async_execute_tool
    from IkaModel.chat_interface.tool_execution_sync import execute_tool
    def echo(args):
        return args['value']
    async def async_echo(args):
        await asyncio.sleep(0)
        return args['value']
    if case == 'tool_async':
        async def operation():
            if await async_execute_tool('echo', {'value': 'ok'}, {'echo': async_echo}) != 'ok':
                raise AssertionError('tool result mismatch')
        return operation
    def operation():
        if execute_tool('echo', {'value': 'ok'}, {'echo': echo}) != 'ok':
            raise AssertionError('tool result mismatch')
    return operation


class FixtureAgent:
    """Pure scheduler fixture; deliberately performs no model/tool/network work."""
    Stages = ()
    use_async = False

    def __init__(self, name):
        self.name = self.prompt = name

    def clone_for_run(self):
        return FixtureAgent(self.name)

    def inject_workflow_context(self, context):
        self.context = context

    def execution(self):
        return {'final_message': self.name, 'summary': self.name}


def workflow_case(case):
    from IkaCore import IkaWorkflow, WorkflowEdge, WorkflowNode
    nodes = [WorkflowNode(name, FixtureAgent(name)) for name in ('root', 'left', 'right', 'join')]
    edges = [WorkflowEdge(a, b) for a, b in [('root', 'left'), ('root', 'right'), ('left', 'join'), ('right', 'join')]]
    workflow = IkaWorkflow('benchmark', 'diamond', nodes, edges, compress_hook=lambda contexts, agent: '\n'.join(contexts),
                           max_parallel_workers=4)
    def operation():
        if set(workflow.run(use_async=case == 'workflow_parallel')) != {'root', 'left', 'right', 'join'}:
            raise AssertionError('workflow lost a node')
    return operation, workflow.async_executor.shutdown
