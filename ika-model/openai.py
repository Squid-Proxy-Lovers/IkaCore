import json
import logging
import re
from typing import Any, List, Optional, Union
from uuid import uuid4

import openai

from .base import Model, Message, MessageRole, RateLimiter, TokenUsage, ToolCall, parse_json_blob
from .tools import Tool

_LOG = logging.getLogger(__name__)


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

    def _messages_to_input(self, messages: list[Message]) -> tuple[Optional[str], list[dict[str, Any]]]:
        items: list[dict[str, Any]] = []
        instructions: Optional[str] = None

        assert len([m for m in messages if m.role == MessageRole.SYSTEM]) <= 1, "There could be at most one SYSTEM message"

        if messages and messages[0].role == MessageRole.SYSTEM:
            instructions = messages[0].content or None

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
            use_instructions=False,
            **kwargs
        )

    def _get_model_cost(self, model_id):
        return {
            'grok-4-fast': (0.40, 0.05, 0.40),
        }.get(model_id)
