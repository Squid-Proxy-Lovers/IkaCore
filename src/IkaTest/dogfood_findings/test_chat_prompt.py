"""A second execution silently reuses the first task after prompt assignment."""
import json
from unittest.mock import patch

import httpx
import pytest

from IkaCore import IkaBaseAgent, OutputConfig, ProviderConfig


@pytest.mark.xfail(strict=True, reason="DOGFOOD-3: execution prefers stale first_input over newly assigned agent.prompt")
def test_updated_prompt_reaches_second_execution_request():
    payloads = []

    def respond(url, headers, payload, **kwargs):
        payloads.append(payload)
        return httpx.Response(200, json={"output": [{"type": "function_call", "id": "x", "call_id": "x",
            "name": "agent_end", "arguments": '{"input":"OK"}'}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}})

    helper = IkaBaseAgent("chat", prompt="FIRST_TURN", model_id="gpt-4o-mini", api_key="fixture",
                          provider=ProviderConfig(), output=OutputConfig(cli=False))
    try:
        with patch("IkaModel.request_interface.api_request_retry", side_effect=respond):
            helper.execution()
            helper.prompt = "SECOND_TURN_NEW_REQUEST"
            helper.execution()
        assert "SECOND_TURN_NEW_REQUEST" in json.dumps(payloads[-1])
    finally:
        helper.shutdown()
