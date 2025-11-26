import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union

from .tools import Tool

_LOG = logging.getLogger(__name__)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    input_cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    input_cost: float = 0.0
    output_cost: float = 0.0

@dataclass
class Message:
    role: MessageRole
    content: Optional[str] = None
    reasoning: Optional[List[str]] = None
    responses_output: Optional[List[Any]] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    token_usage: Optional[TokenUsage] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

def parse_json_blob(text: str) -> dict[str, Any]:
    _LOG.debug("Attempting to parse JSON from text (length: %d)", len(text))
    patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1) if '```' in pattern else match.group(0)
            try:
                result = json.loads(json_str)
                _LOG.debug("Successfully parsed JSON blob")
                return result
            except json.JSONDecodeError as e:
                _LOG.debug("Failed to parse JSON with pattern %s: %s", pattern[:20], e)
                continue
    _LOG.debug("No valid JSON found in text")
    return {}


class RateLimiter:
    def __init__(self, requests_per_minute: Optional[float] = None):
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0.0
        if requests_per_minute:
            _LOG.debug("Rate limiter initialized with %s requests per minute", requests_per_minute)

    def wait_if_needed(self):
        if not self.requests_per_minute:
            return
        min_interval = 60.0 / self.requests_per_minute
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            _LOG.debug("Rate limiting: sleeping for %.2f seconds", sleep_time)
            time.sleep(sleep_time)
        self.last_request_time = time.time()


class Model(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
        stop_sequences: Optional[list[str]] = None,
        response_format: Optional[dict[str, Any]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs
    ) -> Message:
        pass

    def __call__(self, *args, **kwargs) -> Message:
        return self.generate(*args, **kwargs)

    def _get_model_cost(self, model_id: str) -> Optional[tuple[float, float, float]]:
        return {
            "gpt-5": (1.250, 0.125, 10.000),
            "gpt-5-mini": (0.250, 0.025, 2.000),
            "gpt-5-nano": (0.050, 0.005, 0.400),
        }.get(model_id)

    def _prepare_messages(self, messages: list[Message], role_conversions: Optional[dict[str, str]] = None) -> list[dict[str, Any]]:
        _LOG.debug("Preparing %d messages for API call", len(messages))
        result: list[dict[str, Any]] = []
        merged_count = 0

        for msg in messages:
            msg_dict = msg.to_dict()
            original_role = msg_dict["role"]

            if role_conversions and msg_dict["role"] in role_conversions:
                msg_dict["role"] = role_conversions[msg_dict["role"]]
                _LOG.debug("Converted role %s → %s", original_role, msg_dict["role"])

            if (result and result[-1]["role"] == msg_dict["role"] and 
                msg_dict.get("content") and msg_dict["role"] != "tool"):
                result[-1]["content"] = f"{result[-1].get('content', '')}\n{msg_dict['content']}".strip()
                merged_count += 1
            else:
                result.append(msg_dict)

        if merged_count > 0:
            _LOG.debug("Merged %d consecutive messages with same role", merged_count)

        _LOG.debug("Prepared %d messages for API call", len(result))
        return result

    def _calculate_cost(self, usage: TokenUsage) -> TokenUsage:
        if not (usage.input_cost == 0.0 and usage.output_cost == 0.0):
            return usage

        model_name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", getattr(self, 'model_id', ''))
        model_cost = self._get_model_cost(model_name)
        if model_cost is None:
            _LOG.warning("No cost data for model: %s", model_name)
            return usage

        input_cost_per_m, cached_input_cost_per_m, output_cost_per_m = model_cost

        cached_tokens = usage.input_cached_tokens or 0
        uncached_tokens = max(usage.input_tokens - cached_tokens, 0)

        input_cost = (uncached_tokens / 1_000_000) * input_cost_per_m + (cached_tokens / 1_000_000) * cached_input_cost_per_m
        output_cost = (usage.output_tokens / 1_000_000) * output_cost_per_m

        return TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            input_cached_tokens=usage.input_cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            reasoning_tokens=usage.reasoning_tokens
        )
