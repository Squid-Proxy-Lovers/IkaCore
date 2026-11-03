"""
Codex provider — summarization path tests.

Covers:
  - _normalize_provider_for_summary keeps "codex" as itself (does NOT collapse
    to "openai" like openai_responses does).
  - get_summary_model("codex") returns a codex slug + codex URL.
  - create_summary_payload("codex", ...) emits the codex Responses shape
    (stream:true, store:false, structured input, no max_output_tokens).
  - parse_summary_response("codex", ...) walks output[] for output_text.
  - run_summarization with codex provider does NOT rewrite codex URL to
    api.openai.com/v1/chat/completions (the old silent-URL-rewrite bug).
"""
from unittest.mock import MagicMock, patch

import pytest

from IkaModel.base import BareBoneModel
from IkaModel.codex import CODEX_API_URL
from IkaModel.summarization import (
    _normalize_provider_for_summary,
    create_summary_payload,
    get_summary_model,
    parse_summary_response,
    run_summarization,
)

# ----------------------------------------------------------------------
# Provider normalization
# ----------------------------------------------------------------------

@pytest.mark.parametrize("provider_in,provider_out", [
    ("codex",            "codex"),
    ("openai_responses", "openai"),
    ("openai",           "openai"),
    ("deepseek",         "deepseek"),
    ("anthropic",        "anthropic"),
    ("gemini",           "gemini"),
])
def test_normalize_provider_for_summary(provider_in, provider_out):
    assert _normalize_provider_for_summary(provider_in) == provider_out


# ----------------------------------------------------------------------
# get_summary_model
# ----------------------------------------------------------------------

class TestGetSummaryModel:

    def test_codex_returns_codex_slug_and_url(self):
        mid, url = get_summary_model("codex")
        assert mid == "gpt-5.4-mini"
        assert url == CODEX_API_URL

    def test_openai_unchanged(self):
        mid, url = get_summary_model("openai")
        assert "gpt" in mid.lower()
        assert "api.openai.com" in url

    def test_unknown_returns_none_pair(self):
        assert get_summary_model("nonexistent_provider") == (None, None)


# ----------------------------------------------------------------------
# create_summary_payload — codex branch
# ----------------------------------------------------------------------

class TestCodexSummaryPayload:

    def setup_method(self):
        self.payload, self.headers = create_summary_payload(
            "codex", "gpt-5.4-mini", "fake-bearer",
            conversation_text="user: hi\nassistant: hello",
            system_prompt="Summarize.",
            user_prompt_prefix="History:\n",
        )

    def test_stream_true(self):
        assert self.payload["stream"] is True

    def test_store_false(self):
        assert self.payload["store"] is False

    def test_no_max_output_tokens(self):
        assert "max_output_tokens" not in self.payload

    def test_instructions_is_system_prompt(self):
        assert self.payload["instructions"] == "Summarize."

    def test_input_is_structured_user_message(self):
        msg = self.payload["input"][0]
        assert msg["type"] == "message"
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "input_text"
        # User prefix + conversation text
        assert "History:" in msg["content"][0]["text"]
        assert "assistant: hello" in msg["content"][0]["text"]

    def test_authorization_bearer_header(self):
        assert self.headers["Authorization"] == "Bearer fake-bearer"

    def test_accept_event_stream(self):
        assert self.headers["Accept"] == "text/event-stream"


# ----------------------------------------------------------------------
# parse_summary_response — codex branch
# ----------------------------------------------------------------------

class TestParseCodexSummaryResponse:

    def _make_response(self, body):
        r = MagicMock()
        r.json.return_value = body
        return r

    def test_extracts_text_from_output_messages(self):
        body = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "summary part 1 "},
                        {"type": "output_text", "text": "and part 2"},
                    ],
                }
            ]
        }
        assert parse_summary_response("codex", self._make_response(body)) == "summary part 1 and part 2"

    def test_falls_back_to_output_text_field(self):
        body = {"output": [], "output_text": "convenience field"}
        assert parse_summary_response("codex", self._make_response(body)) == "convenience field"

    def test_empty_response_returns_empty_string(self):
        body = {"output": []}
        assert parse_summary_response("codex", self._make_response(body)) == ""


# ----------------------------------------------------------------------
# run_summarization — codex URL must NOT be silently rewritten
# ----------------------------------------------------------------------

class TestCodexURLPreservedInSummarization:

    def _model(self):
        return BareBoneModel(
            model_id="gpt-5.4-mini", api_key="fake-bearer", api_url=CODEX_API_URL,
            system_prompt="You answer.", max_tokens=200, temperature=0,
            reasoning_effort="low", suppress_init_output=True,
        )

    def _mh(self):
        return {
            "system":     {"message": "You answer.", "tokens": 0},
            "first_input":{"message": "User: tell me about X.", "tokens": 0},
            "summary":    {"message": "", "tokens": 0},
            "messages": {
                "id1": {"message": "Assistant: X is a thing.", "tokens": 10, "type": "text"},
            },
        }

    def test_url_stays_codex_in_use_same_model_mode(self):
        """The old code rewrote any URL containing 'responses' to OpenAI
        chat completions — that would have broken codex. Ensure codex URL
        survives."""
        captured = {}

        def fake_retry(url, headers, payload, **kw):
            captured["url"] = url
            captured["payload"] = payload
            r = MagicMock()
            r.json.return_value = {
                "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": "summary"}
                ]}],
                "usage": {"total_tokens": 10},
            }
            r.raise_for_status = MagicMock()
            return r

        with patch("IkaModel.summarization.api_request_retry", side_effect=fake_retry):
            summary = run_summarization(self._model(), self._mh(),
                                        prompt_kind="default",
                                        write_to_history=False, use_same_model=True)

        assert summary == "summary"
        assert captured["url"] == CODEX_API_URL  # the bug fix
        assert captured["payload"]["stream"] is True
        assert "max_output_tokens" not in captured["payload"]
