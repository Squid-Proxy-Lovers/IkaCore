from dataclasses import FrozenInstanceError

import pytest

from IkaCore import AgentTopology, ExecutionLimits, ExecutionPolicy, IkaBaseAgent, ProviderConfig


def test_grouped_configuration_is_frozen_and_run_state_is_independent():
    rates = {'lookup': 10.0}
    limits = ExecutionLimits(maxsteps=2, per_tool_rate_limit=rates)
    agent = IkaBaseAgent('a', 'description', 'task', model_id='gpt-4o', api_key='secret', limits=limits)
    rates['lookup'] = 100
    assert agent.config.limits.per_tool_rate_limit['lookup'] == 10
    with pytest.raises(TypeError):
        agent.config.limits.per_tool_rate_limit['lookup'] = 11
    with pytest.raises(FrozenInstanceError):
        agent.config.limits.maxsteps = 9
    agent.maxsteps += 3
    assert agent.state.maxsteps == 5 and agent.config.limits.maxsteps == 2
    clone = agent.clone_for_run()
    assert clone.maxsteps == 5 and clone.state is not agent.state
    assert 'secret' not in repr(agent.config)
    agent.shutdown()
    clone.shutdown()


def test_no_duplicate_flat_constructor_options_and_collections_are_copied():
    with pytest.raises(TypeError, match='maxsteps'):
        IkaBaseAgent('a', 'description', 'task', model_id='gpt-4o', api_key='key', maxsteps=2)
    checks = []
    children = []
    policy = ExecutionPolicy(final_answer_checks=checks)
    topology = AgentTopology(subagents=children)
    checks.append(lambda _: False)
    children.append(object())
    assert policy.final_answer_checks == () and topology.subagents == ()
    assert ProviderConfig().use_responses_api


def test_dc9efbd_snapshot_resumes_saved_turn_without_provider_or_double_usage(tmp_path, monkeypatch):
    import json
    import sqlite3
    from contextlib import closing
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / 'fixtures/dc9efbd-post-model.json').read_text())
    path = tmp_path / 'legacy.db'
    with closing(sqlite3.connect(path)) as db:
        db.executescript(fixture['sql'])
    agent = IkaBaseAgent('legacy', 'description', 'task', model_id='gpt-4o', api_key='fixture',
                         limits=ExecutionLimits(maxsteps=2))
    agent.enable_snapshots(str(path))
    monkeypatch.setattr('IkaCore.agent_chat_support.chat', lambda *a, **k: pytest.fail('saved turn contacted provider'))
    result = agent.resume_from_snapshot(fixture['snapshot_id'])
    assert result['final_message'] == 'legacy complete'
    assert result['usage']['input_tokens'] == 2 and result['usage']['output_tokens'] == 3
    agent.shutdown()
