"""
Codex (ChatGPT-backed Responses API) provider.

Sibling to the ``openai/`` provider. Talks to
``https://chatgpt.com/backend-api/codex/responses`` and bills against the
caller's ChatGPT plan instead of an OpenAI API key. Lives in its own
subpackage because the codex backend has its own constraints (streaming-only,
codex-specific model slugs) that would pollute the generic OpenAI path.

The provider treats ``BareBoneModel.api_key`` as a literal bearer, matching
every other IkaCore provider. Callers source the bearer however they like:
env var, secret store, or the ``codex_auth.get_bearer()`` opt-in helper that
reads ``~/.codex/auth.json`` and refreshes the JWT when stale.

Public surface used by IkaCore dispatch:
- ``build_codex_request``        build (url, headers, payload)
- ``parse_codex_response``       convert final response dict to internal tuple
- ``append_codex_tool_messages`` record assistant + tool-output turns
- ``request_codex``              POST + SSE collect + return Response-shim

Opt-in convenience:
- ``codex_auth.get_bearer``      read & refresh the Codex CLI's stored token
"""

from ..codex_constants import CODEX_API_URL
from . import auth as codex_auth
from .chat_helpers_codex import (
    append_codex_tool_messages,
    build_codex_request,
    is_codex_url,
    parse_codex_response,
    request_codex,
)
from .codex_responses import codex_responses_fill_payload

__all__ = [
    "codex_responses_fill_payload",
    "build_codex_request",
    "parse_codex_response",
    "append_codex_tool_messages",
    "request_codex",
    "codex_auth",
    "CODEX_API_URL",
    "is_codex_url",
]
