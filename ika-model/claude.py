import json
import logging
from typing import Any, Optional, Union

try:
    import anthropic
except ImportError:
    anthropic = None

from .base import Model, Message, MessageRole, RateLimiter, TokenUsage, ToolCall
from .tools import Tool

_LOG = logging.getLogger(__name__)


class ClaudeModel(Model):
    def __init__(
        self,
        model_id: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if anthropic is None:
            raise ImportError("anthropic package is required. Install it with: pip install anthropic")
        
        self.model_id = model_id
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.client = anthropic.Anthropic(api_key=api_key)
        
        _LOG.info("Initialized Claude model: %s", model_id)

    def _prepare_messages(self, messages: list[Message]) -> tuple[Optional[str], list[dict[str, Any]]]:
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                if system_message is not None:
                    _LOG.warning("Multiple system messages found, merging them")
                    system_message = f"{system_message}\n{msg.content or ''}"
                else:
                    system_message = msg.content
            elif msg.role == MessageRole.TOOL:
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content or ""
                        }
                    ]
                })
            elif msg.role == MessageRole.ASSISTANT:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments
                        })
                if content:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content
                    })
            elif msg.role == MessageRole.USER:
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.content or ""
                })
        
        return system_message, anthropic_messages

    def _tools_to_anthropic(self, tools: list[Tool]) -> list[dict[str, Any]]:
        anthropic_tools = []
        for tool in tools:
            try:
                schema = tool.to_schema()
                if isinstance(schema, dict):
                    func = schema.get("function", {})
                    name = func.get("name")
                    description = func.get("description", "")
                    parameters = func.get("parameters", {})
                else:
                    name = getattr(tool, "name", "tool")
                    description = getattr(tool, "description", "")
                    inputs = getattr(tool, "inputs", {}) or {}
                    properties = {}
                    required = []
                    if isinstance(inputs, dict):
                        for k, spec in inputs.items():
                            if not isinstance(spec, dict):
                                continue
                            typ = spec.get("type", "string")
                            if isinstance(typ, list):
                                typ = typ[0] if typ else "string"
                            if typ == "any":
                                typ = "string"
                            prop = {"type": typ}
                            if typ == "array":
                                prop["items"] = spec.get("items", {"type": "string"})
                            if "description" in spec:
                                prop["description"] = spec["description"]
                            if not spec.get("nullable", False):
                                required.append(k)
                            if "enum" in spec:
                                prop["enum"] = spec["enum"]
                            properties[k] = prop
                    parameters = {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                
                anthropic_tools.append({
                    "name": name,
                    "description": description,
                    "input_schema": parameters
                })
            except Exception as e:
                _LOG.warning("Failed to convert tool %s to Anthropic format: %s", getattr(tool, "name", "unknown"), e)
                continue
        
        return anthropic_tools

    def generate(
        self,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
        stop_sequences: Optional[list[str]] = None,
        response_format: Optional[dict[str, Any]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs
    ) -> Message:
        _LOG.debug("Generating completion with Claude model %s for %d messages", self.model_id, len(messages))
        
        system_message, anthropic_messages = self._prepare_messages(messages)
        
        request_kwargs = {
            "model": self.model_id,
            "messages": anthropic_messages,
            **self.config,
            **kwargs
        }
        
        if system_message:
            request_kwargs["system"] = system_message
        
        if stop_sequences:
            request_kwargs["stop_sequences"] = stop_sequences
        
        if tools:
            request_kwargs["tools"] = self._tools_to_anthropic(tools)
            if tool_choice is not None:
                if tool_choice == "required":
                    request_kwargs["tool_choice"] = {"type": "any"}
                elif tool_choice == "none":
                    request_kwargs["tool_choice"] = {"type": "none"}
                elif isinstance(tool_choice, dict):
                    request_kwargs["tool_choice"] = tool_choice
        
        if response_format and response_format.get("type") == "json_schema":
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": response_format.get("json_schema", {})
            }
        
        self.rate_limiter.wait_if_needed()
        
        response = self.client.messages.create(**request_kwargs)
        
        content_text = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content_text.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else json.loads(block.input) if isinstance(block.input, str) else {}
                ))
        
        usage = response.usage
        
        return Message(
            role=MessageRole.ASSISTANT,
            content="".join(content_text) if content_text else None,
            tool_calls=tool_calls if tool_calls else None,
            token_usage=self._calculate_cost(TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens
            ))
        )

    def _get_model_cost(self, model_id: str) -> Optional[tuple[float, float, float]]:
        model_name_lower = model_id.lower()
        
        if "opus-4-5" in model_name_lower or "opus-4.5" in model_name_lower:
            return (5.00, 25.00, 0.00)
        
        if "opus" in model_name_lower:
            if "3" in model_name_lower:
                return (15.00, 75.00, 0.00)
            return (5.00, 25.00, 0.00)
        
        model_name = model_id.split("-")[0:3]
        if len(model_name) >= 3:
            model_name = "-".join(model_name)
        else:
            model_name = model_id
        
        return {
            "claude-3-5-sonnet": (3.00, 15.00, 0.00),
            "claude-3-5-haiku": (1.00, 5.00, 0.00),
            "claude-3-7-sonnet": (3.00, 15.00, 0.00),
            "claude-3-opus": (15.00, 75.00, 0.00),
            "claude-3-sonnet": (3.00, 15.00, 0.00),
            "claude-3-haiku": (0.25, 1.25, 0.00),
            "claude-3": (3.00, 15.00, 0.00),
            "claude-2": (8.00, 24.00, 0.00),
            "claude-2.1": (8.00, 24.00, 0.00),
        }.get(model_name)

