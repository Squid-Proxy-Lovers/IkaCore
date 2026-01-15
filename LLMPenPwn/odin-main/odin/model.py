import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, List, Optional, Union
from uuid import uuid4

import openai

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
        d = {"role": self.role.value}
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
        result = []
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

        # Remove any date-version pinning from model name
        model_name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", self.model_id)
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

class OpenAIModel(Model):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        custom_role_conversions: Optional[dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.custom_role_conversions = custom_role_conversions
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=10)

        _LOG.info("Initialized OpenAI model: %s", model_id)
        if base_url:
            _LOG.debug("Using custom base URL: %s", base_url)
        if custom_role_conversions:
            _LOG.debug("Using custom role conversions: %s", custom_role_conversions)

    def generate(
        self,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
        stop_sequences: Optional[list[str]] = None,
        response_format: Optional[dict[str, Any]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs
    ) -> Message:
        _LOG.debug("Generating completion with model %s for %d messages", self.model_id, len(messages))

        completion_kwargs = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages, self.custom_role_conversions),
            **self.config,
            **kwargs
        }

        if stop_sequences and self._supports_stop():
            completion_kwargs["stop"] = stop_sequences
            _LOG.debug("Using stop sequences: %s", stop_sequences)

        if response_format:
            completion_kwargs["response_format"] = response_format
            _LOG.debug("Using response format: %s", response_format)

        if tools:
            completion_kwargs["tools"] = [t.to_schema() for t in tools]
            if tool_choice is not None:
                completion_kwargs["tool_choice"] = tool_choice
            _LOG.debug("Using %d tools with tool_choice=%s", len(tools), tool_choice)

        self.rate_limiter.wait_if_needed()

        response = self.client.chat.completions.create(**completion_kwargs)

        choice = response.choices[0].message
        tool_calls = None

        if choice.tool_calls:
            tool_calls = []
            _LOG.debug("Processing %d tool calls from response", len(choice.tool_calls))
            for tc in choice.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError as e:
                        _LOG.warning("Failed to parse tool call arguments as JSON: %s", e)
                        args = {"raw": args}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
                _LOG.debug("Added tool call: %s", tc.function.name)
        elif choice.content and tools and not response_format:
            parsed = parse_json_blob(choice.content)
            if parsed:
                name = parsed.get("tool") or parsed.get("name")
                args = parsed.get("arguments", {})
                if name:
                    tool_calls = [ToolCall(id=str(uuid4()), name=name, arguments=args)]
                    _LOG.debug("Extracted tool call from content: %s", name)

        reasoning = getattr(choice, "reasoning", None)
        return Message(
            role=MessageRole.ASSISTANT,
            content=choice.content,
            tool_calls=tool_calls,
            reasoning=[reasoning] if reasoning else None,
            token_usage=self._calculate_cost(TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                input_cached_tokens=getattr(response.usage.prompt_tokens_details, "cached_tokens", 0),
                output_tokens=response.usage.completion_tokens,
                reasoning_tokens=getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0)
        )))

    def _supports_stop(self) -> bool:
        model_name = self.model_id.split("/")[-1]
        supports_stop = not re.match(r"^(o3[-\d]*|o4-mini[-\d]*|gpt\-5)$", model_name)
        _LOG.debug("Model %s supports stop sequences: %s", model_name, supports_stop)
        return supports_stop

class OpenAIResponsesModel(Model):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        use_instructions: bool = True,
        add_search_tool: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=kwargs.get("max_retries", 5))
        self.use_instructions = use_instructions
        self.add_search_tool = add_search_tool
        _LOG.info("Initialized OpenAI Responses model: %s", model_id)

    def _tools_to_responses(self, tools: list[Tool]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in tools:
            try:
                s = t.to_schema()
            except Exception:
                s = None

            if isinstance(s, dict):
                f = s.get("function", {})
                name = f.get("name")
                description = f.get("description", "")
                parameters = f.get("parameters", {"type": "object", "properties": {}})
            else:
                # Fallback: build parameters schema from attributes directly (robust to custom tool wrappers)
                name = getattr(t, "name", "tool")
                description = getattr(t, "description", "")
                inputs = getattr(t, "inputs", {}) or {}
                properties: dict[str, Any] = {}
                required: list[str] = []
                if isinstance(inputs, dict):
                    for k, spec in inputs.items():
                        if not isinstance(spec, dict):
                            continue
                        typ = spec.get("type", "string")
                        if isinstance(typ, list):
                            typ = typ[0] if typ else "string"
                        if typ == "any":
                            typ = "string"
                        prop: dict[str, Any] = {"type": typ}
                        if typ == "array":
                            prop["items"] = spec.get("items", {"type": "string"})
                        if "description" in spec:
                            prop["description"] = spec["description"]
                        if spec.get("nullable", False):
                            pass
                        else:
                            required.append(k)
                        if "enum" in spec:
                            prop["enum"] = spec["enum"]
                        properties[k] = prop
                parameters = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                }

            out.append({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
            })
        return out

    def _messages_to_input(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        items: list[dict[str, Any]] = []
        instructions: str = ""

        assert len([m for m in messages if m.role == MessageRole.SYSTEM]) <= 1, "There could be at most one SYSTEM message"

        if messages and messages[0].role == MessageRole.SYSTEM:
            instructions = messages[0].content

        for m in messages[1:]:
            if m.role == MessageRole.TOOL:
                if m.tool_call_id:
                    items.append({
                        "type": "function_call_output",
                        "call_id": m.tool_call_id,
                        "output": m.content or "",
                    })
                continue

            if m.role == MessageRole.ASSISTANT:
                items.extend(m.responses_output or [])
                continue

            # backwards compatibility
            if m.role == MessageRole.ASSISTANT and m.tool_calls:
                for tc in m.tool_calls:
                    items.append({
                        "type": "function_call",
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) or "{}",
                        "call_id": tc.id,
                    })
                if m.content:
                    items.append({"role": "assistant", "content": m.content})
                continue

            items.append({"type": "message", "role": m.role.value, "content": m.content or ""})

        return instructions, items

    def generate(
        self,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
        stop_sequences: Optional[list[str]] = None,
        response_format: Optional[dict[str, Any]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs
    ) -> Message:
        _LOG.debug("Generating response with model %s for %d messages", self.model_id, len(messages))

        instructions, input_items = self._messages_to_input(messages)

        if not self.use_instructions and instructions:
            input_items.insert(0, {"type": "message", "role": "system", "content": instructions})
            instructions = None

        req: dict[str, Any] = {
            "model": self.model_id,
            "input": input_items if input_items else "",
            **self.config,
            **kwargs,
            "prompt_cache_key": "odin-cache",
            "store": True,
            "include": [
                "reasoning.encrypted_content",
            ]
        }

        req.setdefault("tools", [])

        if self.add_search_tool:
            req["tools"].append({"type": "web_search"})
            req["include"].append("web_search_call.action.sources")

        if tools:
            req["tools"].extend(self._tools_to_responses(tools))

        n_tools = len(req['tools'])
        _LOG.debug(f"Using {n_tools} tools (Responses schema)")

        if instructions:
            req["instructions"] = instructions

        if response_format and response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {})
            req["text"] = {"format": {"type": "json_schema", **schema}}

        self.rate_limiter.wait_if_needed()
        resp = self.client.responses.create(**req)

        tool_calls: list[ToolCall] = []
        texts: List[str] = []
        summaries = []
        for output in resp.output or []:
            if output.type == "function_call":
                args = output.arguments or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                call_id = getattr(output, "call_id", getattr(output, "id", str(uuid4())))
                tool_calls.append(ToolCall(id=call_id, name=output.name, arguments=args))
            elif output.type == "message":
                for content in output.content:
                    if content.type == "output_text":
                        texts.append(content.text)
            elif output.type == "reasoning":
                for content in output.summary:
                    if content.type == "summary_text":
                        summaries.append(content.text)

        return Message(
            role=MessageRole.ASSISTANT,
            content="".join(texts),
            reasoning=summaries if summaries else None,
            responses_output=[o.to_dict() for o in resp.output] if resp.output else None,
            tool_calls=tool_calls or None,
            token_usage=self._calculate_cost(TokenUsage(
                input_tokens=resp.usage.input_tokens,
                input_cached_tokens=resp.usage.input_tokens_details.cached_tokens,
                output_tokens=resp.usage.output_tokens,
                reasoning_tokens=resp.usage.output_tokens_details.reasoning_tokens
            )),
        )

class OpenRouterAPIModel(OpenAIModel):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        custom_role_conversions: Optional[dict[str, str]] = None,
        **kwargs
    ):
        base_url = "https://openrouter.ai/api/v1"
        super().__init__(
            model_id, api_key=api_key, base_url=base_url,
            requests_per_minute=requests_per_minute,
            custom_role_conversions=custom_role_conversions,
            **kwargs
        )

    def _get_model_cost(self, model_id):
        return {
            'x-ai/grok-4-fast:free': (0, 0, 0),
            'x-ai/grok-code-fast-1': (0.20, 0.02, 1.50),
            'google/gemini-2.5-flash': (0.30, 0.075, 2.50),
            'google/gemini-2.5-pro': (2.50, 0.625, 15.00),
        }.get(model_id)

class XAIAPIModel(OpenAIModel):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        custom_role_conversions: Optional[dict[str, str]] = None,
        **kwargs
    ):
        base_url = "https://api.x.ai/v1"
        super().__init__(
            model_id, api_key=api_key, base_url=base_url,
            requests_per_minute=requests_per_minute,
            custom_role_conversions=custom_role_conversions,
            **kwargs
        )

    def _get_model_cost(self, model_id):
        return {
            'grok-4-fast': (0.40, 0.05, 0.40),
        }.get(model_id)
    
class XAIAPIResponsesModel(OpenAIResponsesModel):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        **kwargs
    ):
        base_url = "https://api.x.ai/v1"
        super().__init__(
            model_id,
            api_key=api_key,
            base_url=base_url,
            requests_per_minute=requests_per_minute,
            use_instructions=False, # unsupported: https://docs.x.ai/docs/guides/responses-api
            **kwargs
        )

    def _get_model_cost(self, model_id):
        return {
            'grok-4-fast': (0.40, 0.05, 0.40),
        }.get(model_id)

class GeminiAPIModel(OpenAIModel):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        custom_role_conversions: Optional[dict[str, str]] = None,
        **kwargs
    ):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        super().__init__(
            model_id, api_key=api_key, base_url=base_url,
            requests_per_minute=requests_per_minute,
            custom_role_conversions=custom_role_conversions,
            **kwargs
        )

    def _get_model_cost(self, model_id):
        return { # TODO
        }.get(model_id)

class DeepSeekAPIModel(OpenAIModel):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        custom_role_conversions: Optional[dict[str, str]] = None,
        **kwargs
    ):
        base_url = "https://api.deepseek.com/v1"
        super().__init__(
            model_id, api_key=api_key, base_url=base_url,
            requests_per_minute=requests_per_minute,
            custom_role_conversions=custom_role_conversions,
            **kwargs
        )

    def _get_model_cost(self, model_id):
        return {
            "deepseek-chat": (0.14, 0.14, 0.28),
            "deepseek-coder": (0.14, 0.14, 0.28),
        }.get(model_id)


try:
    import anthropic
except ImportError:
    anthropic = None


class AnthropicModel(Model):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        **kwargs
    ):
        if anthropic is None:
            raise ImportError(
                "Please install 'anthropic' package to use AnthropicModel: `pip install anthropic`"
            )
        super().__init__(**kwargs)
        self.model_id = model_id
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.client = anthropic.Anthropic(api_key=api_key)
        _LOG.info("Initialized Anthropic model: %s", model_id)

    def generate(
        self,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
        stop_sequences: Optional[list[str]] = None,
        response_format: Optional[dict[str, Any]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs
    ) -> Message:
        _LOG.debug("Generating completion with model %s for %d messages", self.model_id, len(messages))

        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            elif msg.role == MessageRole.TOOL:
                # Anthropic uses tool_result blocks in user messages
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content or ""
                    }]
                })
            elif msg.role == MessageRole.ASSISTANT:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments
                        })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content_blocks if content_blocks else [{"type": "text", "text": ""}]
                })
            else:  # USER
                anthropic_messages.append({"role": "user", "content": msg.content or ""})

        completion_kwargs = {
            "model": self.model_id,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": anthropic_messages,
            **{k: v for k, v in self.config.items() if k != "model"},
            **{k: v for k, v in kwargs.items() if k not in ["max_tokens"]}
        }

        if system_message:
            completion_kwargs["system"] = system_message

        if stop_sequences:
            completion_kwargs["stop_sequences"] = stop_sequences
            _LOG.debug("Using stop sequences: %s", stop_sequences)

        if tools:
            # Convert tools to Anthropic format
            anthropic_tools = []
            for tool in tools:
                schema = tool.to_schema()
                func_schema = schema.get("function", {}) if isinstance(schema, dict) else {}
                anthropic_tools.append({
                    "name": func_schema.get("name", tool.name),
                    "description": func_schema.get("description", getattr(tool, "description", "")),
                    "input_schema": func_schema.get("parameters", {})
                })
            completion_kwargs["tools"] = anthropic_tools
            if tool_choice is not None:
                completion_kwargs["tool_choice"] = tool_choice
            _LOG.debug("Using %d tools with tool_choice=%s", len(tools), tool_choice)

        self.rate_limiter.wait_if_needed()

        response = self.client.messages.create(**completion_kwargs)

        tool_calls = None
        content_text = ""
        if response.content:
            tool_calls = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input
                    ))
                    _LOG.debug("Added tool call: %s", block.name)
                elif block.type == "text":
                    content_text += block.text

        return Message(
            role=MessageRole.ASSISTANT,
            content=content_text if content_text else None,
            tool_calls=tool_calls if tool_calls else None,
            token_usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
        )


from portkey_ai import Portkey

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

        # Convert Message objects into Portkey format
        pk_messages = [m.to_dict() for m in messages]

        completion_kwargs = {
            "model": self.model_id,
            "messages": pk_messages,
            **kwargs
        }

        if stop_sequences:
            completion_kwargs["stop"] = stop_sequences

        if tools:
            # Optional: inject tools into Portkey call if needed
            # Portkey may support a 'tools' or 'functions' param similar to OpenAI
            completion_kwargs["tools"] = [t.to_schema() for t in tools]
            if tool_choice is not None:
                completion_kwargs["tool_choice"] = tool_choice

        _LOG.debug("Calling PortKey API with %d messages", len(messages))
        response = self.client.chat.completions.create(**completion_kwargs)

        # Extract tool calls if any
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
