import atexit
import json
import queue
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from IkaModel.model_metadata import resolve_model_cost

WriteItem = Tuple[str, Union[str, Dict[str, Any]]]  # ("line", str) or ("json", dict)
_WriteItem = WriteItem


class _LoggerState(Protocol):
    level: int
    log_file: Path
    use_colors: bool
    show_usage_level0: bool
    log_input_enabled: bool
    _queue: Optional[queue.Queue[Optional[_WriteItem]]]
    _shutdown: threading.Event
    _writer_thread: Optional[threading.Thread]
    _atexit_registered: bool

    def _ensure_writer(self) -> None:
        ...

    def _writer_loop(self) -> None:
        ...

    def _process_item(self, item: _WriteItem) -> None:
        ...

    def _write_json(self, payload: Dict[str, Any]) -> None:
        ...

    def _color(self, text: str, color: str) -> str:
        ...

    def write_line(self, line: str) -> None:
        ...

    def log_json(self, payload: Dict[str, Any]) -> None:
        ...

    def shutdown(self) -> None:
        ...


class LoggerWriterProcessingMixin:
    def _process_item(self: _LoggerState, item: _WriteItem) -> None:
        item_type, data = item

        if item_type == "line" and isinstance(data, str):
            if self.level == 1:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(data + "\n")
            elif self.level == 2:
                self._write_json({"text": data})
        elif item_type == "json" and isinstance(data, dict) and self.level == 2:
            self._write_json(data)

    def _write_json(self: _LoggerState, payload: Dict[str, Any]) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        payload_with_ts = {"ts": time.time(), **payload}
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload_with_ts, ensure_ascii=True) + "\n")


class LoggerWriterThreadMixin(LoggerWriterProcessingMixin):
    def _ensure_writer(self: _LoggerState) -> None:
        if self._writer_thread is not None:
            return
        self._queue = queue.Queue[Optional[_WriteItem]]()
        self._shutdown.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="IkaLogger-Writer")
        self._writer_thread.start()
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True

    def _writer_loop(self: _LoggerState) -> None:
        q = self._queue
        if q is None:
            return
        while not self._shutdown.is_set():
            try:
                item = q.get(timeout=0.1)
                if item is None:
                    break
                self._process_item(item)
                q.task_done()
            except queue.Empty:
                continue

        while True:
            try:
                item = q.get_nowait()
                if item is not None:
                    self._process_item(item)
                q.task_done()
            except queue.Empty:
                break

    def flush(self: _LoggerState) -> None:
        if self._queue is not None:
            self._queue.join()

    def shutdown(self: _LoggerState) -> None:
        if self._writer_thread is None or self._shutdown.is_set():
            return
        self._shutdown.set()
        if self._queue is not None:
            self._queue.put(None)
        if self._writer_thread.is_alive() and self._writer_thread != threading.current_thread():
            self._writer_thread.join(timeout=1.0)
        self._writer_thread = None
        self._queue = None
        if self._atexit_registered:
            try:
                atexit.unregister(self.shutdown)
            except (AttributeError, ValueError):
                pass
            self._atexit_registered = False


class LoggerWriteApiMixin(LoggerWriterThreadMixin):
    def _color(self: _LoggerState, text: str, color: str) -> str:
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
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def write_line(self: _LoggerState, line: str) -> None:
        if self.level == 0:
            return
        self._ensure_writer()
        assert self._queue is not None
        self._queue.put(("line", line))

    def log_json(self: _LoggerState, payload: Dict[str, Any]) -> None:
        if self.level != 2:
            return
        self._ensure_writer()
        assert self._queue is not None
        self._queue.put(("json", payload))

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        memo[id(self)] = self
        return self

    def __copy__(self) -> Any:
        return self


class LoggerConversationEventsMixin:
    def log_input(self: _LoggerState, messages: List[dict]) -> None:
        if not self.log_input_enabled or not messages:
            return
        user_msg = messages[-1]
        line = f"[INPUT] role={user_msg.get('role')} content={user_msg.get('content')}"
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "input", "message": user_msg})

    def log_tool_results(self: _LoggerState, tool_calls: List[dict], tool_results: List[str]) -> None:
        if not tool_calls:
            return
        for idx, call in enumerate(tool_calls):
            result = tool_results[idx] if idx < len(tool_results) else ""
            name = call.get("name") or call.get("function", {}).get("name", "")
            self.write_line(f"[TOOL] name={name} result={result}")
        if self.level == 2:
            self.log_json({"event": "tool_results", "tool_calls": tool_calls, "tool_results": tool_results})

    def log_output(
        self: _LoggerState,
        content: str,
        usage: Optional[Dict[str, Any]] = None,
        cost: Optional[Dict[str, float]] = None,
        message_history: Optional[dict] = None,
    ) -> None:
        line = f"[OUTPUT] content={content}"
        if usage:
            line += f" usage={usage}"
        if cost:
            line += f" cost={cost}"
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "output", "content": content, "usage": usage, "cost": cost, "message_history": message_history})

    def log_summary(self: _LoggerState, summary: str) -> None:
        self.write_line(f"[SUMMARY] {summary}")
        if self.level == 2:
            self.log_json({"event": "summary", "summary": summary})


class LoggerStageEventsMixin:
    def log_action(self: _LoggerState, action: str) -> None:
        self.write_line(f"[ACTION] {action}")
        if self.level == 2:
            self.log_json({"event": "action", "action": action})

    def log_stage_start(self: _LoggerState, stage_name: str, hitl: bool, remaining_steps: int, step_limit: int) -> None:
        line = self._color(f"[STAGE START] {stage_name} hitl={hitl} remaining={remaining_steps} limit={step_limit}", "cyan")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "stage_start", "stage": stage_name, "hitl": hitl, "remaining": remaining_steps, "limit": step_limit})

    def log_stage_end(self: _LoggerState, stage_name: str, used_steps: int) -> None:
        line = self._color(f"[STAGE END] {stage_name} used_steps={used_steps}", "magenta")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "stage_end", "stage": stage_name, "used_steps": used_steps})

    def log_step(self: _LoggerState, stage_name: str, step_idx: int, output: str, tool_calls: list, usage: dict, cost: dict, elapsed: float) -> None:
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


class LoggerHitlEventsMixin:
    def log_hitl_prompt(self: _LoggerState, stage_name: str) -> None:
        line = self._color(f"[HITL] Stage '{stage_name}' awaiting user input. Type your message or 'stage_end' to finish.", "orange")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "hitl_prompt", "stage": stage_name})

    def log_hitl_input(self: _LoggerState, stage_name: str, user_text: str) -> None:
        line = self._color(f"[HITL INPUT] stage={stage_name} user='{user_text}'", "orange")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "hitl_input", "stage": stage_name, "user_text": user_text})

    def log_hitl_question(self: _LoggerState, stage_name: str, question: str) -> None:
        line = self._color(f"[HITL] stage={stage_name} question='{question}'", "orange")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "hitl_question", "stage": stage_name, "question": question})

    def log_hitl_answer(self: _LoggerState, stage_name: str, user_text: str) -> None:
        line = self._color(f"[HITL ANSWER] stage={stage_name} user='{user_text}'", "orange")
        self.write_line(line)
        if self.level == 2:
            self.log_json({"event": "hitl_answer", "stage": stage_name, "user_text": user_text})


class LoggerCostMixin:
    @staticmethod
    @lru_cache(maxsize=4096)
    def get_model_cost(model_id: str) -> Optional[tuple[float, float, float]]:
        return resolve_model_cost(model_id)

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
