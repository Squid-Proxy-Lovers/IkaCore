"""Offline provider fixtures exercising the real serializers, tools and runtime.

Use MockProvider as a context manager. It substitutes model *and* summary request
boundaries locally; unexpected requests fail rather than making network calls.
"""
from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from copy import deepcopy
from threading import Lock
from typing import Any
from uuid import uuid4

import httpx

from IkaModel.request_adapter import request_adapter

Action = tuple[str, dict[str, Any]] | str
__all__ = ["MockProvider", "provider_response"]


class MockProvider:
    def __init__(self, responses: Sequence[dict[str, Any] | httpx.Response] = (), *,
                 responder: Callable[[str, dict[str, Any]], dict[str, Any] | httpx.Response] | None = None):
        if responses and responder is not None:
            raise ValueError("Choose responses or a responder")
        self._responses = deque(responses)
        self._responder = responder
        self.requests: list[dict[str, Any]] = []
        self.urls: list[str] = []
        self._lock = Lock()
        self._scope: AbstractContextManager[None] | None = None

    def request(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        with self._lock:
            self.requests.append(deepcopy(payload))
            self.urls.append(url)
            if self._responder is None:
                if not self._responses:
                    raise AssertionError("Unexpected provider request: scripted responses exhausted")
                response = self._responses.popleft()
            else:
                response = self._responder(url, payload)
        if isinstance(response, dict):
            response = httpx.Response(200, json=response)
        try:
            _ = response.request
        except RuntimeError:
            response.request = httpx.Request("POST", url)
        return response

    def assert_consumed(self) -> None:
        assert not self._responses, "Some scripted provider responses were not consumed"

    def __enter__(self) -> MockProvider:
        if self._scope is not None:
            raise RuntimeError("A MockProvider instance cannot be entered twice")
        self._scope = request_adapter(self.request)
        self._scope.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        assert self._scope is not None
        self._scope.__exit__(*exc)
        self._scope = None
        if exc[0] is None:
            self.assert_consumed()


def provider_response(url: str, payload: dict[str, Any], action: Action, index: int | None = None) -> httpx.Response:
    """Provider-native fixture envelopes exercise real payload and response adapters."""
    text = action if isinstance(action, str) else ""
    name, arguments = action if isinstance(action, tuple) else ("", {})
    uid = f"fixture_{index if index is not None else uuid4().hex}"
    usage = {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
    if "anthropic.com" in url:
        content = ([{"type": "tool_use", "id": uid, "name": name, "input": arguments}] if name
                   else [{"type": "text", "text": text}])
        data = {"id": uid, "type": "message", "role": "assistant", "content": content,
                "stop_reason": "tool_use" if name else "end_turn", "usage": usage}
    elif "generativelanguage.googleapis.com" in url:
        parts = [{"functionCall": {"name": name, "args": arguments}}] if name else [{"text": text}]
        data = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 20, "totalTokenCount": 100}}
    elif "responses" in url:
        output = ([{"type": "function_call", "id": uid, "call_id": uid, "name": name,
                    "arguments": json.dumps(arguments), "status": "completed"}] if name else
                  [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}])
        data = {"id": uid, "status": "completed", "output": output, "usage": usage}
    else:
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if name:
            message["tool_calls"] = [{"id": uid, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)}}]
        data = {"choices": [{"message": message, "finish_reason": "tool_calls" if name else "stop"}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}}
    return httpx.Response(200, json=data, request=httpx.Request("POST", url))

