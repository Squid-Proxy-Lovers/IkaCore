import pytest

from IkaCore.tools import IkaTools, validate_required_fields


def test_validate_required_fields_reports_missing_field_name():
    with pytest.raises(ValueError, match="description is required"):
        validate_required_fields({"name": "tool", "description": ""})


def test_ikatools_accepts_explicit_id_and_executes_callable():
    tool = IkaTools(
        "lookup",
        "Lookup data",
        {"query": "string"},
        execute_function=lambda params: f"found:{params['query']}",
        id="fixed-id",
    )

    assert tool.id == "fixed-id"
    assert tool.execute({"query": "abc"}) == "found:abc"


def test_ikatools_execute_without_callable_returns_error_message():
    tool = object.__new__(IkaTools)
    tool.execute_function = None

    assert tool.execute({}) == '{"error": "No execute function defined for this tool"}'
