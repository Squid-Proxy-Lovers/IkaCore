"""
Codex provider example: drive an IkaBaseAgent through OpenAI's ChatGPT-backed
``codex`` Responses endpoint, billed against your ChatGPT plan instead of an
OpenAI API key.

Prerequisites
-------------
- Install the official Codex CLI and run ``codex login`` once (writes
  ``~/.codex/auth.json``).
- A ChatGPT Plus/Pro/Business/Enterprise/Edu plan with codex access.

Sourcing the bearer
-------------------
The codex provider treats ``api_key`` as a literal bearer, same as every
other IkaCore provider. There are two common ways to get one:

1. Env var (preferred, matches other providers):
       export CODEX_BEARER=<paste your token here>
2. Opt-in helper that reads ``~/.codex/auth.json`` and refreshes the JWT
   automatically when it's within 8 minutes of expiry:
       from IkaModel.codex import codex_auth
       bearer = codex_auth.get_bearer()

This example tries the env var first, then falls back to the helper.

How the wire works
------------------
- ``api_url`` points at the codex backend. IkaCore's URL dispatcher routes
  the request through the codex sibling provider.
- The codex backend rejects ``stream: false``; IkaCore handles SSE collection
  internally and surfaces a single final response, so callers don't see
  streaming events.
- ``model_id`` must be a codex-known slug: ``gpt-5.5``, ``gpt-5.4``,
  ``gpt-5.4-mini``, ``gpt-5.3-codex``, ``gpt-5.2``, ``gpt-5.2-codex``.
- ``reasoning_effort`` accepts the codex-extended set
  ``low | medium | high | xhigh`` on gpt-5.5.
"""

from __future__ import annotations

import os

from IkaCore import IkaBaseAgent, IkaTools
from IkaModel.codex import CODEX_API_URL, codex_auth

SYSTEM_PROMPT = """You are a concise agent. Use the tools provided when they
help, and call agent_end with a short final answer when you're done."""

PROMPT = """Use the list_files tool to look at the current directory, then
tell me what kind of project this looks like in one sentence."""


def _resolve_bearer() -> str:
    """Env var first; fall back to reading ~/.codex/auth.json."""
    env_bearer = os.getenv("CODEX_BEARER")
    if env_bearer:
        return env_bearer
    return codex_auth.get_bearer()


def main() -> None:
    bearer = _resolve_bearer()
    print(f"[codex] bearer={bearer[:12]}...  plan={codex_auth.get_chatgpt_plan()}")

    get_pwd = IkaTools(
        name="get_pwd",
        description="Return the absolute path of the current working directory.",
        parameters={"None": "No parameters are required"},
        execute_function=lambda _: os.getcwd(),
    )

    list_files = IkaTools(
        name="list_files",
        description="List entries in a directory",
        parameters={
            "directory_path": {
                "type": "string",
                "description": "Absolute path to list",
                "required": True,
            },
        },
        execute_function=lambda x: os.listdir(x["directory_path"]),
    )

    model_id = os.getenv("CODEX_MODEL_ID", "gpt-5.4-mini")
    reasoning_effort = os.getenv("CODEX_REASONING_EFFORT", "low")

    agent = IkaBaseAgent(
        name="codex_example",
        description="Tiny codex-provider smoke test",
        system_prompt=SYSTEM_PROMPT,
        prompt=PROMPT,
        tools=[get_pwd, list_files],
        model_id=model_id,
        api_url=CODEX_API_URL,
        api_key=bearer,
        reasoning_effort=reasoning_effort,
        maxsteps=8,
        logging_level=2,
    )
    agent.execution()


if __name__ == "__main__":
    main()
