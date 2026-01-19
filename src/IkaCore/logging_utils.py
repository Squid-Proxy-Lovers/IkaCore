import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class IkaLogger:
    def __init__(self, level: int = 0, log_file: str = "logs.txt", use_colors: bool = True, show_usage_level0: bool = True):
        self.level = level
        self.log_file = Path(log_file)
        self.use_colors = use_colors
        self.show_usage_level0 = show_usage_level0

    def _color(self, text: str, color: str) -> str:
        if not self.use_colors or self.level != 0:
            return text
        colors = {
            "cyan": "\033[96m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "magenta": "\033[95m",
            "reset": "\033[0m",
        }
        return f"{colors.get(color,'')}{text}{colors['reset']}"

    def write_line(self, line: str) -> None:
        if self.level == 0:
            print(line)
        elif self.level == 1:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        elif self.level == 2:
            # level 2 uses JSON lines, handled by log_json
            self.log_json({"text": line})

    def log_json(self, payload: Dict[str, Any]) -> None:
        if self.level != 2:
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        payload_with_ts = {"ts": time.time(), **payload}
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload_with_ts, ensure_ascii=True) + "\n")

    def log_input(self, messages: List[dict]) -> None:
        if not messages:
            return
        user_msg = messages[-1]
        line = f"[INPUT] role={user_msg.get('role')} content={user_msg.get('content')}"
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "input", "message": user_msg})

    def log_tool_results(self, tool_calls: List[dict], tool_results: List[str]) -> None:
        if not tool_calls:
            return
        for idx, call in enumerate(tool_calls):
            result = tool_results[idx] if idx < len(tool_results) else ""
            name = call.get("name") or call.get("function", {}).get("name", "")
            line = f"[TOOL] name={name} result={result}"
            self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "tool_results", "tool_calls": tool_calls, "tool_results": tool_results})

    def log_output(self, content: str, usage: Optional[Dict[str, Any]] = None, cost: Optional[Dict[str, float]] = None, message_history: Optional[dict] = None) -> None:
        line = f"[OUTPUT] content={content}"
        if usage:
            line += f" usage={usage}"
        if cost:
            line += f" cost={cost}"
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "output", "content": content, "usage": usage, "cost": cost, "message_history": message_history})

    def log_summary(self, summary: str) -> None:
        line = f"[SUMMARY] {summary}"
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "summary", "summary": summary})

    def log_action(self, action: str) -> None:
        line = f"[ACTION] {action}"
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "action", "action": action})

    def log_stage_start(self, stage_name: str, hitl: bool, remaining_steps: int, step_limit: int) -> None:
        line = self._color(f"[STAGE START] {stage_name} hitl={hitl} remaining={remaining_steps} limit={step_limit}", "cyan")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "stage_start", "stage": stage_name, "hitl": hitl, "remaining": remaining_steps, "limit": step_limit})

    def log_stage_end(self, stage_name: str, used_steps: int) -> None:
        line = self._color(f"[STAGE END] {stage_name} used_steps={used_steps}", "magenta")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "stage_end", "stage": stage_name, "used_steps": used_steps})

    def log_step(self, stage_name: str, step_idx: int, output: str, tool_calls: list, usage: dict, cost: dict, elapsed: float) -> None:
        tool_names = [t.get("name") or t.get("function", {}).get("name", "") for t in (tool_calls or [])]
        preview = (output or "")[:200].replace("\n", " ")
        usage_part = f" tokens={usage} cost={cost}" if (self.level != 0 or self.show_usage_level0) else ""
        line = self._color(f"[STEP] stage={stage_name} step={step_idx} tools={tool_names} elapsed={elapsed:.2f}s{usage_part} out='{preview}'", "green")
        self.write_line(line)
        if self.level == 2:
            self.log_json({
                "event": "step",
                "stage": stage_name,
                "step": step_idx,
                "tools": tool_names,
                "elapsed_sec": elapsed,
                "usage": usage,
                "cost": cost,
                "output_preview": preview,
            })

    def log_hitl_prompt(self, stage_name: str) -> None:
        line = self._color(f"[HITL] Stage '{stage_name}' awaiting user input. Type your message or 'stage_end' to finish.", "yellow")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "hitl_prompt", "stage": stage_name})

    def log_hitl_input(self, stage_name: str, user_text: str) -> None:
        line = self._color(f"[HITL INPUT] stage={stage_name} user='{user_text}'", "yellow")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "hitl_input", "stage": stage_name, "user_text": user_text})

    @staticmethod
    def get_model_cost(model_id: str) -> Optional[tuple[float, float, float]]:
        cost_map = {
            "gpt-4o": (5.00, 5.00, 15.00),
            "gpt-4.1": (5.00, 5.00, 15.00),
            "gpt-4.1-mini": (0.150, 0.150, 0.600),
            "gpt-4.1-nano": (0.050, 0.050, 0.400),
            "claude-3-opus": (15.00, 15.00, 75.00),
            "claude-3-sonnet": (3.00, 3.00, 15.00),
            "claude-3-haiku": (0.250, 0.250, 1.250),
            "claude-sonnet-4": (3.00, 3.00, 15.00),
            "deepseek-chat": (0.140, 0.140, 0.280),
            "gemini-1.5-pro": (3.50, 3.50, 10.50),
        }
        return cost_map.get(model_id.lower(), None)

    def compute_cost(self, model_id: str, usage: Dict[str, Any]) -> Dict[str, float]:
        model_cost = self.get_model_cost(model_id)
        if model_cost is None:
            return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

        input_cost_per_m, cached_input_cost_per_m, output_cost_per_m = model_cost
        input_tokens = usage.get("input_tokens", 0) or 0
        cached_tokens = usage.get("input_cached_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or usage.get("total_tokens", 0) or 0

        uncached_tokens = max(input_tokens - cached_tokens, 0)
        input_cost = (uncached_tokens / 1_000_000) * input_cost_per_m + (cached_tokens / 1_000_000) * cached_input_cost_per_m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_m
        total_cost = input_cost + output_cost

        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
        }
