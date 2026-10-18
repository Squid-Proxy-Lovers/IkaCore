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
