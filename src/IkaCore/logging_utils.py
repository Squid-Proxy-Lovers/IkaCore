import atexit
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from IkaCore.cli_output import _stdout_lock

# Type alias for queue items
_WriteItem = Tuple[str, Union[str, Dict[str, Any]]]  # ("line", str) or ("json", dict)


class IkaLogger:
    """level 0: no logging; 1: file; 2: JSON file."""

    def __init__(self, level: int = 0, log_file: str = "logs.txt", use_colors: bool = True, show_usage_level0: bool = True, log_input_enabled: bool = False):
        self.level = level
        self.log_file = Path(log_file)
        self.use_colors = use_colors
        self.show_usage_level0 = show_usage_level0
        self.log_input_enabled = log_input_enabled

        # Queue-based async writer
        self._queue: queue.Queue[Optional[_WriteItem]] = queue.Queue()
        self._shutdown = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="IkaLogger-Writer")
        self._writer_thread.start()

        # Register shutdown handler to flush on exit
        atexit.register(self.shutdown)

    def _writer_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:  # Shutdown signal
                    break
                self._process_item(item)
                self._queue.task_done()
            except queue.Empty:
                continue

        # Drain remaining items on shutdown
        while True:
            try:
                item = self._queue.get_nowait()
                if item is not None:
                    self._process_item(item)
                self._queue.task_done()
            except queue.Empty:
                break

    def _process_item(self, item: _WriteItem) -> None:
        """Process a single queue item (runs in writer thread)."""
        item_type, data = item

        if item_type == "line" and isinstance(data, str):
            if self.level == 0:
                pass
            elif self.level == 1:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(data + "\n")
            elif self.level == 2:
                self._write_json({"text": data})

        elif item_type == "json" and isinstance(data, dict):
            if self.level == 2:
                self._write_json(data)

    def _write_json(self, payload: Dict[str, Any]) -> None:
        """Write JSON payload to file (runs in writer thread)."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        payload_with_ts = {"ts": time.time(), **payload}
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload_with_ts, ensure_ascii=True) + "\n")

    def _color(self, text: str, color: str) -> str:
        if not self.use_colors or self.level != 0:
            return text
        colors = {
            "cyan": "\033[96m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "orange": "\033[38;5;208m",
            "red": "\033[91m",
            "magenta": "\033[95m",
            "reset": "\033[0m",
        }
        return f"{colors.get(color,'')}{text}{colors['reset']}"

    def write_line(self, line: str) -> None:
        """Queue a line for writing (non-blocking). No-op when level is 0."""
        if self.level == 0:
            return
        self._queue.put(("line", line))

    def log_json(self, payload: Dict[str, Any]) -> None:
        """Queue a JSON payload for writing (non-blocking)."""
        if self.level != 2:
            return
        self._queue.put(("json", payload))

    def flush(self) -> None:
        """Block until all queued writes are complete."""
        self._queue.join()

    def shutdown(self) -> None:
        """Gracefully shutdown the writer thread."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self._queue.put(None)  # Signal writer to exit
        if self._writer_thread.is_alive() and self._writer_thread != threading.current_thread():
            self._writer_thread.join(timeout=1.0)
        try:
            atexit.unregister(self.shutdown)
        except (AttributeError, ValueError):
            pass

    def __deepcopy__(self, memo: Dict[int, Any]) -> "IkaLogger":
        """Return self on deepcopy - all copies share the same writer thread."""
        memo[id(self)] = self
        return self

    def __copy__(self) -> "IkaLogger":
        """Return self on copy - all copies share the same writer thread."""
        return self

    def log_input(self, messages: List[dict]) -> None:
        if not self.log_input_enabled or not messages:
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
        line = self._color(f"[HITL] Stage '{stage_name}' awaiting user input. Type your message or 'stage_end' to finish.", "orange")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "hitl_prompt", "stage": stage_name})

    def log_hitl_input(self, stage_name: str, user_text: str) -> None:
        line = self._color(f"[HITL INPUT] stage={stage_name} user='{user_text}'", "orange")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "hitl_input", "stage": stage_name, "user_text": user_text})

    def log_hitl_question(self, stage_name: str, question: str) -> None:
        line = self._color(f"[HITL] stage={stage_name} question='{question}'", "orange")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "hitl_question", "stage": stage_name, "question": question})

    def log_hitl_answer(self, stage_name: str, user_text: str) -> None:
        line = self._color(f"[HITL ANSWER] stage={stage_name} user='{user_text}'", "orange")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "hitl_answer", "stage": stage_name, "user_text": user_text})

    @staticmethod
    def get_model_cost(model_id: str) -> Optional[tuple[float, float, float]]:
        # (input_cost_per_M, cached_input_cost_per_M, output_cost_per_M)
        cost_map = {
            "gpt-4o": (2.50, 1.25, 10.00),
            "gpt-4.1": (2.00, 0.50, 8.00),
            "gpt-4.1-mini": (0.40, 0.10, 1.60),
            "gpt-4.1-nano": (0.10, 0.025, 0.40),
            "gpt-5": (2.50, 0.25, 20.00),
            "gpt-5-mini": (0.45, 0.045, 3.60),
            "gpt-5-nano": (0.10, 0.01, 0.40),
            "gpt-5.2": (1.75, 0.175, 14.00),
            "gpt-5.3-codex": (1.75, 0.175, 14.00),
            "gpt-5.4": (2.50, 0.25, 15.00),
            "gpt-5.4-mini": (0.75, 0.075, 4.50),
            "gpt-5.4-nano": (0.20, 0.02, 1.25),
            "gpt-5.4-pro": (30.00, 30.00, 180.00),
            "claude-3-opus": (15.00, 1.50, 75.00),
            "claude-3-sonnet": (3.00, 0.30, 15.00),
            "claude-3-haiku": (0.250, 0.03, 1.250),
            "claude-3-5-haiku": (0.80, 0.08, 4.00),
            "claude-3-5-sonnet": (3.00, 0.30, 15.00),
            "claude-3-7-sonnet": (3.00, 0.30, 15.00),
            "claude-sonnet-4": (3.00, 0.30, 15.00),
            "claude-sonnet-4-5": (3.00, 0.30, 15.00),
            "claude-sonnet-4-6": (3.00, 0.30, 15.00),
            "claude-opus-4": (15.00, 1.50, 75.00),
            "claude-opus-4-1": (15.00, 1.50, 75.00),
            "claude-opus-4-5": (5.00, 0.50, 25.00),
            "claude-opus-4-6": (5.00, 0.50, 25.00),
            "claude-haiku-4": (0.80, 0.08, 4.00),
            "claude-haiku-4-5": (1.00, 0.10, 5.00),
            "deepseek-chat": (0.280, 0.028, 0.420),
            "deepseek-reasoner": (0.280, 0.028, 0.420),
            "gemini-1.5-pro": (3.50, 3.50, 10.50),
            "gemini-2.0-flash": (0.10, 0.025, 0.40),
            "gemini-2.5-flash-preview-05-20": (0.15, 0.0375, 0.60),
            "gemini-3-flash": (0.50, 0.05, 3.00),
            "gemini-3-flash-preview": (0.50, 0.05, 3.00),
            "gemini-3-pro": (2.00, 0.20, 12.00),
            "gemini-3.1-pro": (2.00, 0.20, 12.00),
            "gemini-3.1-pro-preview": (2.00, 0.20, 12.00),
            "gemini-3.1-pro-preview-customtools": (2.00, 0.20, 12.00),
            # Released Mar 18, 2026. 196,608 context.
            "minimax-m2.7": (0.30, 0.30, 1.20),
        }
        mid = (model_id or "").lower().strip()
        if not mid:
            return None

        # Strip provider prefixes like "openai/gpt-5.4".
        if "/" in mid:
            mid = mid.split("/", 1)[1]

        result = cost_map.get(mid)
        if result:
            return result

        # Resolve provider-specific model IDs, for example:
        # "anthropic.claude-sonnet-4-6", "us.anthropic.claude-opus-4-6-v1:0", "claude-opus-4-5@20251101"
        candidates = [mid]
        if "anthropic." in mid:
            candidates.append(mid.split("anthropic.", 1)[1])
        if "." in mid:
            candidates.append(mid.split(".")[-1])
        for delim in ("@", ":"):
            if delim in mid:
                candidates.append(mid.split(delim, 1)[0])
        # De-duplicate while preserving order.
        seen = set()
        normalized = []
        for c in candidates:
            c = c.strip()
            if c and c not in seen:
                normalized.append(c)
                seen.add(c)

        for cand in normalized:
            result = cost_map.get(cand)
            if result:
                return result

        # Try prefix matching for versioned model IDs.
        sorted_keys = sorted(cost_map.keys(), key=len, reverse=True)
        for cand in normalized:
            for key in sorted_keys:
                if cand.startswith(key):
                    return cost_map[key]
        return None

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
