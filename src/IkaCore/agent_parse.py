from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from IkaCore.stages import IkaStage


class AgentParseMixin:

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        
        stripped = text.strip()
        
        # Try direct JSON parse
        try:
            json.loads(stripped)
            return stripped
        except Exception:
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
                except Exception:
                    continue
        
        return None

    def parse_control_calls(
        self,
        tool_calls: List[dict],
        stage: Optional[IkaStage],
        current_stage_idx: int = 0,
        response_content: Optional[str] = None,
    ) -> Tuple[Optional[int], bool, Optional[str]]:
        target_stage: Optional[int | str] = None
        agent_end_called = False
        agent_end_text: Optional[str] = None

        allowed_back = getattr(stage, "allowed_back_to", []) if stage else []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name") or call.get("name")
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}

            # Log memory tool usage (tools are executed by chat function via tool_executors)
            if name in ["short_term_save", "short_term_search", "long_term_save", "long_term_search"]:
                if getattr(self, "logger", None):
                    self.logger.log_action(f"{name} called")
                continue

            if name == "agent_end":
                agent_end_called = True
                agent_end_text = args.get("input", "")
                if not agent_end_text or agent_end_text.strip() in [".", ""]:
                    for key, value in args.items():
                        if isinstance(value, str):
                            stripped = value.strip()
                            if stripped.startswith("{") or stripped.startswith("["):
                                agent_end_text = value
                                break
                            elif len(stripped) > 10 and stripped != ".":
                                agent_end_text = value
                                break
                        elif value and not isinstance(value, (dict, list)) and str(value).strip() not in [".", ""]:
                            agent_end_text = str(value)
                            break
                
                # If still empty, try to extract from response content
                if not agent_end_text or agent_end_text.strip() in [".", ""]:
                    if response_content:
                        extracted_json = self._extract_json_from_text(response_content)
                        if extracted_json:
                            agent_end_text = extracted_json
                
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
                except json.JSONDecodeError:
                    pass
                
                break

            if name == "stage_end" and stage is not None:
                target_stage = "next"
            if name == "change_stage" and stage is not None:
                stage_idx = args.get("stage_index") or args.get("stage") or args.get("to")
                try:
                    stage_idx = int(stage_idx)
                except Exception:
                    stage_idx = None
                if stage_idx is not None and stage_idx in allowed_back and stage_idx < current_stage_idx:
                    target_stage = stage_idx

        # target_stage can be an int or the literal string "next"; type ignore for that union here,
        # callers already handle both cases.
        return target_stage, agent_end_called, agent_end_text
