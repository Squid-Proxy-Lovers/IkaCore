"""CLI, bounded configuration and scripted HTTP responses shared by the apps.

Offline mode replaces the request boundary, not agents, tools, history or snapshots.
The supported IkaCore.testing adapter scopes offline requests to this session.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from IkaCore import ExecutionLimits, ExecutionPolicy, IkaBaseAgent, OutputConfig, ProviderConfig
from IkaCore.testing import MockProvider, provider_response
from IkaModel import request_controls
from IkaModel.codex import CODEX_API_URL

Action = tuple[str, dict[str, Any]] | str


def call(name: str, **args: Any) -> Action:
    return name, args


def end(text: str) -> Action:
    return call("agent_end", input=text)


def parser(description: str | None) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="script provider responses; default")
    mode.add_argument("--live", action="store_true", help="make bounded real Codex requests")
    result.add_argument("--use-local-login", action="store_true", help="opt in to Codex CLI credential loading")
    result.add_argument("--model", default="gpt-5.6-sol")
    return result


def credentials(args: argparse.Namespace) -> str:
    if not args.live:
        return "offline-fixture"
    bearer = os.getenv("CODEX_BEARER")
    if not bearer and args.use_local_login:
        from IkaModel.codex.auth import get_bearer
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bearer = get_bearer()
    if not bearer:
        raise ValueError("Use --use-local-login or set CODEX_BEARER for --live")
    return bearer


def agent(args: argparse.Namespace, name: str, prompt: str, **kwargs: Any) -> IkaBaseAgent:
    agent_type = kwargs.pop("agent_type", IkaBaseAgent)
    return agent_type(
        name, prompt=f"TASK:{name}\n{prompt}", model_id=args.model, api_key=credentials(args),
        provider=kwargs.pop("provider", ProviderConfig.bounded(api_url=CODEX_API_URL, max_tokens=700, reasoning_effort="low")),
        limits=kwargs.pop("limits", ExecutionLimits(maxsteps=4, step_timeout=45, max_tool_rounds=4, max_tool_calls=12)),
        policy=kwargs.pop("policy", ExecutionPolicy(max_step_extensions=0, max_stage_extensions=0,
                                                   validation_retries=1, validation_extra_steps=1)),
        output=OutputConfig(cli=False), **kwargs,
    )


class ScriptedProvider(MockProvider):
    def __init__(self, scripts: dict[str, list[Action]]):
        self.scripts = {key: deque(value) for key, value in scripts.items()}
        super().__init__(responder=self._respond)
        self.summary_calls = 0

    def _respond(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        if not payload.get("tools"):
            self.summary_calls += 1
            action: Action = "Preference: blue. Research: source A supports 20% savings; source B says pilot first."
        else:
            raw = json.dumps(payload)
            matches = [key for key in self.scripts if f"TASK:{key}" in raw]
            if len(matches) != 1:
                raise AssertionError(f"Expected one fixture task, found {matches}")
            queue = self.scripts[matches[0]]
            if not queue:
                raise AssertionError(f"Unexpected extra model call for {matches[0]}")
            action = queue.popleft()
        return provider_response(url, payload, action, len(self.requests))

    def assert_consumed(self) -> None:
        assert all(not queue for queue in self.scripts.values()), "Some scripted model turns never ran"



@contextlib.contextmanager
def session(args: argparse.Namespace, scripts: dict[str, list[Action]]) -> Iterator[ScriptedProvider | None]:
    with contextlib.ExitStack() as stack:
        mock = None
        if not args.live:
            mock = ScriptedProvider(scripts)
            stack.enter_context(mock)
            # Any missed request path fails closed instead of accidentally reaching a provider.
            stack.enter_context(patch.object(httpx.Client, "send", side_effect=AssertionError("offline network access")))
            stack.enter_context(patch.object(httpx.AsyncClient, "send", side_effect=AssertionError("offline network access")))
        elif os.getenv("IKACORE_APP_METRICS"):
            from IkaModel import request_interface, summarization
            metrics = Path(os.environ["IKACORE_APP_METRICS"])
            lock = threading.Lock()
            original_sync = request_interface.api_request_retry
            original_async = request_interface.async_api_request_retry

            def record() -> None:
                with lock, metrics.open("a") as stream:
                    stream.write('{"request_started": true}\n')

            def observed_sync(*args: Any, **kwargs: Any):
                record()
                return original_sync(*args, **kwargs)

            async def observed_async(*args: Any, **kwargs: Any):
                record()
                return await original_async(*args, **kwargs)

            for module in (request_interface, summarization):
                stack.enter_context(patch.object(module, "api_request_retry", observed_sync))
                stack.enter_context(patch.object(module, "async_api_request_retry", observed_async))
        stack.enter_context(request_controls(timeout=getattr(args, "session_timeout", 90), max_retries=1, tool_result_max_chars=1200,
                                            codex_tool_output_max_chars=1200))
        yield mock
        if mock:
            mock.assert_consumed()


def emit(app: str, **checks: Any) -> None:
    # Deliberately print checks rather than model output or request payloads.
    print(json.dumps({"app": app, "status": "passed", **checks}, sort_keys=True))
