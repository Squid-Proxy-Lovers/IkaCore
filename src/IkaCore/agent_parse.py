from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Tuple

from IkaCore.execution_types import StageTarget
from IkaCore.stages import IkaStage


class JsonToolArgumentParseMixin:
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None

        stripped = text.strip()

        # Try direct JSON parse
        try:
            json.loads(stripped)
            return stripped
        except (TypeError, json.JSONDecodeError):
            pass

        # Try to find JSON in code blocks or after markers
        patterns = [
            r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",  # Markdown code blocks
            r"(\{.*?\}|\[.*?\])",  # Any JSON-like structure
        ]

        for pattern in patterns:
            matches = re.findall(pattern, stripped, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)
                    return match
                except (TypeError, json.JSONDecodeError):
                    continue
        
        return None

    def _parse_tool_arguments(self, args_raw: Any) -> dict[str, Any]:
        if not isinstance(args_raw, str):
            return args_raw if isinstance(args_raw, dict) else {}
        try:
            args = json.loads(args_raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        return args if isinstance(args, dict) else {}


class AgentEndParseMixin(JsonToolArgumentParseMixin):
    def _agent_end_text_from_args(self, args: dict[str, Any], response_content: Optional[str]) -> Optional[str]:
        agent_end_text = args.get("input", "")
        if agent_end_text and str(agent_end_text).strip() not in [".", ""]:
            return str(agent_end_text)

        for value in args.values():
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    return value
                if len(stripped) > 10 and stripped != ".":
                    return value
            elif value and not isinstance(value, (dict, list)) and str(value).strip() not in [".", ""]:
                return str(value)

        if response_content:
            return self._extract_json_from_text(response_content)
        return None

    def _validate_agent_end_text(self, agent_end_text: Optional[str]) -> str:
        if not agent_end_text or agent_end_text.strip() in [".", ""]:
            raise ValueError(
                "agent_end was called with empty or invalid arguments. "
                "The agent MUST provide a final answer/output when calling agent_end. "
                "Use the 'input' parameter to pass your response."
            )

        stripped_text = agent_end_text.strip()
        if stripped_text in ["{}", "[]", "null", '""', "''"]:
            raise ValueError(
                f"agent_end was called with invalid/empty content: '{stripped_text}'. "
                "You MUST provide a meaningful final answer, not empty JSON objects, arrays, or null values."
            )

        try:
            parsed = json.loads(stripped_text)
        except json.JSONDecodeError:
            return agent_end_text

        if isinstance(parsed, dict):
            if len(parsed) == 0:
                raise ValueError(
                    f"agent_end was called with empty JSON object: '{stripped_text}'. "
                    "You MUST provide a meaningful final answer."
                )
            if (
                len(parsed) == 1
                and "functions" in parsed
                and isinstance(parsed["functions"], list)
                and len(parsed["functions"]) == 0
            ):
                raise ValueError(
                    f"agent_end was called with empty functions list: '{stripped_text}'. "
                    "You MUST provide a meaningful final answer."
                )
        elif isinstance(parsed, list) and len(parsed) == 0:
            raise ValueError(
                f"agent_end was called with empty JSON array: '{stripped_text}'. "
                "You MUST provide a meaningful final answer."
            )
        return agent_end_text

    def _handle_agent_end_call(
        self,
        args: dict[str, Any],
        response_content: Optional[str],
    ) -> str:
        return self._validate_agent_end_text(self._agent_end_text_from_args(args, response_content))


class StageControlParseMixin(AgentEndParseMixin):
    def _target_stage_from_change_call(
        self,
        args: dict[str, Any],
        allowed_back: list[int],
        current_stage_idx: int,
    ) -> Optional[int]:
        stage_idx = next(
            (args[key] for key in ("stage_index", "stage", "to") if key in args and args[key] is not None),
            None,
        )
        if stage_idx is None:
            return None
        try:
            parsed_stage_idx = int(stage_idx)
        except (TypeError, ValueError):
            return None
        if parsed_stage_idx in allowed_back and parsed_stage_idx < current_stage_idx:
            return parsed_stage_idx
        return None

    def parse_control_calls(
        self,
        tool_calls: List[dict],
        stage: Optional[IkaStage],
        current_stage_idx: int = 0,
        response_content: Optional[str] = None,
    ) -> Tuple[Optional[StageTarget], bool, Optional[str]]:
        target_stage: Optional[StageTarget] = None
        agent_end_called = False
        agent_end_text: Optional[str] = None

        allowed_back = getattr(stage, "allowed_back_to", []) if stage else []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name") or call.get("name")
            args_raw = fn.get("arguments") or "{}"
            args = self._parse_tool_arguments(args_raw)

            # Log memory tool usage (tools are executed by chat function via tool_executors)
            if name in ["short_term_save", "short_term_search", "long_term_save", "long_term_search"]:
                logger = getattr(self, "logger", None)
                if logger:
                    logger.log_action(f"{name} called")
                continue

            if name == "agent_end":
                agent_end_called = True
                agent_end_text = self._handle_agent_end_call(args, response_content)
                break

            if name == "stage_end" and stage is not None:
                target_stage = "next"
            if name == "change_stage" and stage is not None:
                changed_stage = self._target_stage_from_change_call(args, allowed_back, current_stage_idx)
                if changed_stage is not None:
                    target_stage = changed_stage

        return target_stage, agent_end_called, agent_end_text


class AgentParseMixin(StageControlParseMixin):
    pass
