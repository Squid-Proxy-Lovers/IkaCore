"""Optional live provider smoke tests.

These tests are intentionally skipped unless IKACORE_LIVE_PROVIDER_TESTS=1 is set.
They validate provider plumbing only; deterministic contract tests remain the normal
CI gate.
"""

import os

import pytest

from IkaModel.base import BareBoneModel
from IkaModel.chat_interface.chat_interface import chat, init_message_history

_LIVE_PROVIDER_REQUIREMENTS = {
    "OpenAI": ("OPENAI_API_KEY", None),
    "Anthropic": ("ANTHROPIC_API_KEY", "IKACORE_LIVE_ANTHROPIC_MODEL"),
    "DeepSeek": ("DEEPSEEK_API_KEY", None),
    "Gemini": ("GEMINI_API_KEY", "IKACORE_LIVE_GEMINI_MODEL"),
    "OpenRouter": ("OPENROUTER_API_KEY", "IKACORE_LIVE_OPENROUTER_MODEL"),
}

pytestmark = pytest.mark.skipif(
    os.getenv("IKACORE_LIVE_PROVIDER_TESTS") != "1",
    reason="set IKACORE_LIVE_PROVIDER_TESTS=1 to run live provider smoke tests",
)


def _has_env(name: str) -> bool:
    return bool(os.getenv(name))


def _configured_live_providers() -> list[str]:
    configured = []
    for provider, (key_env, model_env) in _LIVE_PROVIDER_REQUIREMENTS.items():
        if _has_env(key_env) and (model_env is None or _has_env(model_env)):
            configured.append(provider)
    return configured


def _required_env(name: str, provider: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is required for live {provider} smoke test")
    return value


def _run_smoke(model: BareBoneModel) -> None:
    model.agent_tools = []
    out = chat(
        model,
        [{"role": "user", "content": "Reply with exactly: ok"}],
        message_history=init_message_history(),
        timeout=30.0,
    )
    assert out["content"]
    assert "usage" in out


def test_live_smoke_has_at_least_one_configured_provider():
    configured = _configured_live_providers()
    if not configured:
        required = ", ".join(
            f"{provider}: {key_env}" + (f" + {model_env}" if model_env else "")
            for provider, (key_env, model_env) in _LIVE_PROVIDER_REQUIREMENTS.items()
        )
        pytest.fail(f"IKACORE_LIVE_PROVIDER_TESTS=1 but no live provider is configured. Expected one of: {required}")


def test_openai_live_smoke_parses_minimal_response():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for live OpenAI smoke test")

    model = BareBoneModel(
        model_id=os.getenv("IKACORE_LIVE_OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        api_url="https://api.openai.com/v1/chat/completions",
        suppress_init_output=True,
        use_responses_api=False,
        max_tokens=32,
        temperature=0.0,
    )
    _run_smoke(model)


def test_anthropic_live_smoke_parses_minimal_response():
    model_id = _required_env("IKACORE_LIVE_ANTHROPIC_MODEL", "Anthropic")
    model = BareBoneModel(
        model_id=model_id,
        api_key=_required_env("ANTHROPIC_API_KEY", "Anthropic"),
        api_url="https://api.anthropic.com/v1/messages",
        suppress_init_output=True,
        max_tokens=32,
        temperature=0.0,
    )
    _run_smoke(model)


def test_deepseek_live_smoke_parses_minimal_response():
    model = BareBoneModel(
        model_id=os.getenv("IKACORE_LIVE_DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=_required_env("DEEPSEEK_API_KEY", "DeepSeek"),
        api_url="https://api.deepseek.com/chat/completions",
        suppress_init_output=True,
        max_tokens=32,
        temperature=0.0,
    )
    _run_smoke(model)


def test_gemini_live_smoke_parses_minimal_response():
    model_id = _required_env("IKACORE_LIVE_GEMINI_MODEL", "Gemini")
    model = BareBoneModel(
        model_id=model_id,
        api_key=_required_env("GEMINI_API_KEY", "Gemini"),
        api_url=f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
        suppress_init_output=True,
        max_tokens=32,
        temperature=0.0,
    )
    _run_smoke(model)


def test_openrouter_live_smoke_parses_minimal_response():
    model_id = _required_env("IKACORE_LIVE_OPENROUTER_MODEL", "OpenRouter")
    model = BareBoneModel(
        model_id=model_id,
        api_key=_required_env("OPENROUTER_API_KEY", "OpenRouter"),
        api_url="https://openrouter.ai/api/v1/chat/completions",
        suppress_init_output=True,
        max_tokens=32,
        temperature=0.0,
    )
    _run_smoke(model)
