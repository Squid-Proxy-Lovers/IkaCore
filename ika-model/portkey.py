import json
import logging
from typing import Any, Optional, Union

from portkey_ai import Portkey

from .base import Model, Message, MessageRole, RateLimiter, TokenUsage, ToolCall
from .tools import Tool

_LOG = logging.getLogger(__name__)


class PortKeyModel(Model):
    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1",
        requests_per_minute: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.client = Portkey(base_url=base_url, api_key=api_key)
        _LOG.info("Initialized PortKey model: %s", model_id)
        _LOG.debug("Using PortKey base_url: %s", base_url)

    def generate(
        self,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
        stop_sequences: Optional[list[str]] = None,
        response_format: Optional[dict[str, Any]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs
    ) -> Message:
        self.rate_limiter.wait_if_needed()

        pk_messages = [m.to_dict() for m in messages]

        completion_kwargs = {
            "model": self.model_id,
            "messages": pk_messages,
            **kwargs
        }

        if stop_sequences:
            completion_kwargs["stop"] = stop_sequences

        if tools:
            completion_kwargs["tools"] = [t.to_schema() for t in tools]
            if tool_choice is not None:
                completion_kwargs["tool_choice"] = tool_choice

        _LOG.debug("Calling PortKey API with %d messages", len(messages))
        response = self.client.chat.completions.create(**completion_kwargs)

        tool_calls = []
        if getattr(response.choices[0].message, "tool_calls", None):
            for tc in response.choices[0].message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        choice = response.choices[0].message
        return Message(
            role=MessageRole.ASSISTANT,
            content=choice.content,
            tool_calls=tool_calls or None,
            reasoning=getattr(choice, "reasoning", None),
            token_usage=TokenUsage(
                input_tokens=getattr(response.usage, "prompt_tokens", 0),
                output_tokens=getattr(response.usage, "completion_tokens", 0)
            )
        )

