import asyncio
import random

import pytest

from IkaModel.chat_interface.tool_execution_async import async_execute_tool
from IkaModel.chat_interface.tool_execution_sync import execute_tool
from IkaModel.codex.codex_responses import _codex_tool_output_item, _codex_user_items
from IkaModel.request_control import request_controls
from IkaModel.tool_output import truncate_codex_tool_output, truncate_tool_result


@pytest.fixture(autouse=True)
def clear_limits(monkeypatch):
    monkeypatch.delenv("IKA_TOOL_RESULT_MAX_CHARS", raising=False)
    monkeypatch.delenv("IKA_CODEX_TOOL_OUTPUT_MAX_CHARS", raising=False)


def test_output_limit_properties_across_unicode_and_boundaries():
    generator = random.Random(7342)
    cases = [("", 1), ("abc", 1), ("abc", 3), ("abcdef", 5), ("x" * 1000, 69)]
    cases += [
        ("".join(chr(generator.randrange(0x110000)) for _ in range(generator.randrange(300))),
         generator.randrange(1, 400))
        for _ in range(1000)
    ]
    for text, limit in cases:
        assert truncate_tool_result(text) == text
        assert truncate_codex_tool_output(text) == text
        with request_controls(tool_result_max_chars=limit, codex_tool_output_max_chars=limit):
            for truncate in (truncate_tool_result, truncate_codex_tool_output):
                result = truncate(text)
                assert len(result) <= limit
                assert truncate(result) == result
                if len(text) <= limit:
                    assert result == text


def test_large_output_preserves_head_and_tail():
    text = "START" + "x" * 10000 + "END"
    with request_controls(tool_result_max_chars=256):
        result = truncate_tool_result(text)
    assert result.startswith("START") and result.endswith("END")
    assert "truncated" in result and len(result) == 256


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "secret-looking-invalid-configuration"])
def test_invalid_limits_fail_before_execution(limit):
    with pytest.raises(ValueError, match="positive integer"), request_controls(tool_result_max_chars=limit):
        pass


def test_explicit_environment_limit_and_context_override(monkeypatch):
    monkeypatch.setenv("IKA_TOOL_RESULT_MAX_CHARS", "128")
    assert len(truncate_tool_result("x" * 1000)) == 128
    with request_controls(tool_result_max_chars=256):
        assert len(truncate_tool_result("x" * 1000)) == 256


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_invalid_environment_limit_does_not_echo_its_value(monkeypatch, value):
    monkeypatch.setenv("IKA_TOOL_RESULT_MAX_CHARS", value)
    with pytest.raises(ValueError, match="positive integer") as error:
        truncate_tool_result("output")
    assert error.value.__cause__ is None


def test_tool_execution_limits_are_opt_in_for_sync_and_async():
    output = "x" * 1000
    tools = {"emit": lambda args: output}
    assert execute_tool("emit", {}, tools) == output
    assert asyncio.run(async_execute_tool("emit", {}, tools)) == output
    with request_controls(tool_result_max_chars=128):
        assert len(execute_tool("emit", {}, tools)) == 128
        assert len(asyncio.run(async_execute_tool("emit", {}, tools))) == 128


def test_codex_tool_field_limits_do_not_change_default_payload_types():
    block = {"type": "tool_result", "tool_use_id": "call", "content": {"value": "x" * 1000}}
    message = {"role": "user", "content": [block]}
    assert _codex_user_items(message)[0]["output"] == block["content"]
    with request_controls(codex_tool_output_max_chars=128):
        result = _codex_user_items(message)[0]["output"]
        assert isinstance(result, str) and len(result) <= 128
        assert len(_codex_tool_output_item({"content": "x" * 1000})["output"]) == 128
