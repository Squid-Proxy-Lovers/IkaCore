"""Unfixed user-facing validation contracts found using the installed v2 wheel."""
from unittest.mock import patch

import httpx
import pytest

from IkaCore import ExecutionPolicy, IkaBaseAgent, IkaStage, OutputConfig, ProviderConfig


def make_agent(check, **kwargs):
    return IkaBaseAgent("repro", prompt="Answer briefly", model_id="gpt-4o-mini", api_key="fixture",
                        provider=ProviderConfig(use_responses_api=False), output=OutputConfig(cli=False),
                        policy=ExecutionPolicy(final_answer_checks=(check,), validation_retries=0), **kwargs)


@pytest.mark.xfail(strict=True, reason="DOGFOOD-1: constructor calls final-answer check with history, missing final_message")
def test_final_check_can_read_final_message_without_constructor_keyerror():
    def check(result) -> bool:
        """The final answer must be OK."""
        return result["final_message"] == "OK"

    helper = make_agent(check)
    helper.shutdown()


@pytest.mark.xfail(strict=True, reason="DOGFOOD-2: staged execution bypasses configured final_answer_checks")
def test_staged_execution_invokes_final_answer_check():
    checked_final = []

    def check(result) -> bool:
        """Record actual final answers (ignore constructor's history probe)."""
        if "final_message" in result:
            checked_final.append(result["final_message"])
        return True

    response = httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "",
        "tool_calls": [{"id": "end", "type": "function", "function": {"name": "agent_end",
        "arguments": '{"input": "OK"}'}}]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    helper = make_agent(check, Stages=[IkaStage("answer", "Finish with OK", [], stage_max_step=1)])
    try:
        with patch("IkaModel.request_interface.api_request_retry", return_value=response):
            assert helper.execution()["final_message"] == "OK"
        assert checked_final == ["OK"]
    finally:
        helper.shutdown()
