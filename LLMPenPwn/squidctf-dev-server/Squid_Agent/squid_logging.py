#!/usr/bin/env python3
import json
import os
import time
import re
import sys
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

def _slugify(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9\-_. ]+", "", s)
    s = s.replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return s

class SquidAgentRunLogger:
    def __init__(
        self,
        challenge_title: str,
        base_dir: str = "logs_squidagent",
        timestamp: bool = False
    ) -> None:
        self.challenge_title = challenge_title
        self.base_dir = base_dir
        self.events: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.success: Optional[bool] = None
        self.exit_reason: Optional[str] = None
        self.error: Optional[str] = None

        os.makedirs(self.base_dir, exist_ok=True)

        slug = _slugify(self.challenge_title)
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = f"{ts}_{slug}.json"
        else:
            self.filename = f"{slug}.json"
            
        self.file_path = os.path.join(self.base_dir, self.filename)

    def start(self) -> None:
        self.start_time = time.time()
        self._write()

    def log_event(
        self,
        *,
        agent: Optional[str] = None,
        role: Optional[str] = None,
        content: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        index: Optional[int] = None,
    ) -> None:
        event: Dict[str, Any] = {}
        if agent is not None: event["agent"] = agent
        if role is not None: event["role"] = role
        if index is not None: event["index"] = index
        if content is not None: event["content"] = content
        
        # Try to clean up tool calls if they are passed as strings that look like JSON
        if tool_call is not None:
            event["tool_call"] = self._maybe_truncate(tool_call)
        
        if tool_result is not None:
            event["tool_result"] = self._maybe_truncate(tool_result)
        if meta is not None: event["meta"] = meta
        
        self.events.append(event)
        self._write()

    def set_outcome(
        self,
        *,
        success: Optional[bool] = None,
        exit_reason: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        if success is not None: self.success = success
        if exit_reason is not None: self.exit_reason = exit_reason
        if error is not None: self.error = error
        self._write()

    def finish(self) -> None:
        self.end_time = time.time()
        self._write()

    def _maybe_truncate(self, obj: Any, max_chars: int = 50000) -> Any:
        try:
            text = json.dumps(obj)
            if len(text) <= max_chars:
                return obj
            truncated = text[: max_chars - 3] + "..."
            try:
                return json.loads(truncated)
            except Exception:
                return {"truncated_json": truncated}
        except Exception:
            return obj

    def _write(self) -> None:
        current_end = self.end_time if self.end_time else time.time()
        duration = (current_end - self.start_time) if self.start_time else 0.0

        data: Dict[str, Any] = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "time_taken": duration,
            "status": "completed" if self.end_time else "running",
            "success": self.success,
            "exit_reason": self.exit_reason,
            "error": self.error,
            "transcript": self.events,
        }

        tmp_path = self.file_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.file_path)
        except Exception as e:
            # Avoid infinite recursion if print fails
            pass

class LogCaptureHandler(logging.Handler):
    """Intercepts Python logging messages and sends them to the SquidAgentRunLogger."""
    def __init__(self, run_logger):
        super().__init__()
        self.run_logger = run_logger

    def emit(self, record):
        try:
            msg = self.format(record)
            # Basic heuristic to categorize logs
            role = "system"
            if "Tool call" in msg:
                role = "assistant"
            elif "Tool response" in msg:
                role = "tool"
            
            self.run_logger.log_event(role=role, content=msg)
        except Exception:
            self.handleError(record)

class StreamToLogger:
    """Fake file-like stream that redirects writes to both stdout and the logger."""
    def __init__(self, original_stream, run_logger):
        self.original_stream = original_stream
        self.run_logger = run_logger

    def write(self, buf):
        # Write to the actual console so user sees it
        self.original_stream.write(buf)
        self.original_stream.flush()
        
        # Log to JSON if it's a meaningful line
        if buf.strip():
            # Heuristic: try to parse tool calls if they appear in stdout
            content = buf.strip()
            self.run_logger.log_event(role="console", content=content)

    def flush(self):
        self.original_stream.flush()
    
    def isatty(self):
        return self.original_stream.isatty()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

class run_log:
    """
    Context manager that:
    1. Initializes the JSON logger.
    2. Captures sys.stdout (print statements).
    3. Captures logging.info/warning/error.
    4. Writes everything to the JSON file live.
    """
    def __init__(
        self,
        challenge_title: str,
        base_dir: str = "logs_squidagent",
        timestamp: bool = False,
        capture_stdout: bool = True
    ) -> None:
        self._logger = SquidAgentRunLogger(challenge_title, base_dir=base_dir, timestamp=timestamp)
        self.capture_stdout = capture_stdout
        self.original_stdout = sys.stdout
        self.log_handler = None

    def __enter__(self) -> SquidAgentRunLogger:
        self._logger.start()
        
        # 1. Attach logging handler
        self.log_handler = LogCaptureHandler(self._logger)
        self.log_handler.setLevel(logging.INFO)
        # Attach to root logger to catch everything
        logging.getLogger().addHandler(self.log_handler)
        
        # 2. Capture stdout (for print statements)
        if self.capture_stdout:
            sys.stdout = StreamToLogger(self.original_stdout, self._logger)
            
        return self._logger

    def __exit__(self, exc_type, exc, tb) -> None:
        # Restore stdout
        if self.capture_stdout:
            sys.stdout = self.original_stdout

        # Remove logging handler
        if self.log_handler:
            logging.getLogger().removeHandler(self.log_handler)

        if exc is not None:
            self._logger.set_outcome(success=False, error=str(exc), exit_reason="exception")
        self._logger.finish()