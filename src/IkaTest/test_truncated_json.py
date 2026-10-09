import pytest

from IkaCore.agent_parse import JsonToolArgumentParseMixin
from IkaCore.truncated_json import _close_truncated_json_object
from IkaModel import runtime_options

_parse_tool_arguments = JsonToolArgumentParseMixin()._parse_tool_arguments


@pytest.mark.parametrize("raw, expected", [
    ('{"input":"half', {"input": "half"}),
    ('{"input":"slash\\', {"input": "slash"}),
    ('{"input":"unicode\\u12', {"input": "unicode"}),
    ('{"input":{"items":[1,2', {"input": {"items": [1, 2]}}),
    ('{"input":"escaped\\\"quote', {"input": 'escaped"quote'}),
    ('[1,2', None), ('{"input":', None), ('{"x":} ', None),
    ('{"input":"complete"}', {"input": "complete"}),
])
def test_recovery_only_returns_valid_objects(raw, expected):
    assert _close_truncated_json_object(raw) == expected


def test_recovery_is_explicit_and_only_for_control_tools():
    raw = '{"input":"part'
    assert _parse_tool_arguments(raw, "agent_end") == {}
    with runtime_options(recover_truncated_control_calls=True):
        assert _parse_tool_arguments(raw, "agent_end") == {"input": "part"}
        assert _parse_tool_arguments(raw, "ordinary") == {}
