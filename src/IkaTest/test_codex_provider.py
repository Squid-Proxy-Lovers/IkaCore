"""
Codex provider — provider dispatch & URL routing tests.

Covers:
  - get_provider() recognizes the codex URL.
  - get_provider() recognizes unambiguous "-codex" suffixed model IDs.
  - get_provider() does NOT auto-route bare gpt-5.x slugs to codex (those
    overlap with the standard OpenAI Responses API).
  - IkaBaseAgent.geturl() picks CODEX_API_URL for "-codex" model IDs.
  - Bare codex slugs without api_url still go to OpenAI by default.
"""
import pytest

from IkaCore.agent_helpers import AgentHelpersMixin
from IkaModel.codex import CODEX_API_URL, is_codex_url
from IkaModel.model_metadata import CODEX_KNOWN_MODELS
from IkaModel.request_interface import get_provider

geturl = AgentHelpersMixin.geturl


def test_gpt_5_6_variants_are_known_codex_slugs():
    assert {"gpt-5.6-sol", "gpt-5.6-terra"} <= CODEX_KNOWN_MODELS


# ----------------------------------------------------------------------
# get_provider — URL-based detection
# ----------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://chatgpt.com/backend-api/codex/responses",     "codex"),
    ("https://CHATGPT.com/backend-api/codex/responses",     "codex"),  # case-insensitive
    ("https://api.openai.com/v1/responses",                 "openai_responses"),
    ("https://api.openai.com/v1/chat/completions",          "openai"),
    ("https://api.deepseek.com/chat/completions",           "deepseek"),
    ("https://openrouter.ai/api/v1/chat/completions",       "openrouter"),
    ("https://api.anthropic.com/v1/messages",               "anthropic"),
])
def test_get_provider_by_url(url, expected):
    assert get_provider("gpt-anything", url) == expected


# ----------------------------------------------------------------------
# get_provider — model_id-based detection
# ----------------------------------------------------------------------

@pytest.mark.parametrize("model_id,expected", [
    ("gpt-5.3-codex",   "codex"),
    ("gpt-5.2-codex",   "codex"),
    ("GPT-5.3-CODEX",   "codex"),
    # Bare codex slugs are ambiguous (also valid OpenAI Responses models)
    # — should NOT auto-route to codex.
    ("gpt-5.5",         "openai_responses"),
    ("gpt-5.4",         "openai_responses"),
    ("gpt-5.4-mini",    "openai_responses"),
])
def test_get_provider_by_model_id(model_id, expected):
    """No api_url; provider determined from model_id alone."""
    assert get_provider(model_id, None) == expected


# ----------------------------------------------------------------------
# IkaBaseAgent.geturl — URL auto-routing for -codex slugs
# ----------------------------------------------------------------------

def test_geturl_codex_slug_routes_to_codex_backend():
    url = geturl("gpt-5.3-codex")
    assert url == CODEX_API_URL
    assert is_codex_url(url)


def test_geturl_codex_slug_case_insensitive():
    assert geturl("GPT-5.2-CODEX") == CODEX_API_URL


def test_geturl_bare_gpt5_stays_openai():
    """Bare gpt-5.x slugs must NOT auto-route to codex (overlap with OpenAI)."""
    assert geturl("gpt-5.5") == "https://api.openai.com/v1/responses"
    assert geturl("gpt-5.4-mini") == "https://api.openai.com/v1/responses"
    assert geturl("gpt-4o") == "https://api.openai.com/v1/responses"


def test_geturl_openrouter_format_unchanged():
    """OpenRouter slash format must still win over codex routing."""
    # "x/gpt-5.3-codex" is OpenRouter syntax — should NOT be sent to codex
    assert geturl("openrouter/gpt-5.3-codex") == "https://openrouter.ai/api/v1/chat/completions"


# ----------------------------------------------------------------------
# is_codex_url helper
# ----------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://chatgpt.com/backend-api/codex/responses", True),
    ("HTTPS://CHATGPT.COM/backend-api/codex/responses", True),
    ("https://api.openai.com/v1/responses", False),
    ("", False),
    (None, False),
])
def test_is_codex_url(url, expected):
    assert is_codex_url(url) is expected
