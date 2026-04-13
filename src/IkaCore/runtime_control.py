from __future__ import annotations

from dataclasses import dataclass, field
from threading import local
from typing import Any, Callable, Optional


_TOOL_CONTEXT = local()


class RuntimePauseRequested(RuntimeError):
    def __init__(self, checkpoint_kind: str, snapshot_id: Optional[str] = None, frame_id: Optional[str] = None):
        super().__init__(f"Runtime paused at checkpoint '{checkpoint_kind}'")
        self.checkpoint_kind = checkpoint_kind
        self.snapshot_id = snapshot_id
        self.frame_id = frame_id


@dataclass
class RuntimeControl:
    pause_points: set[str] = field(default_factory=set)
    parallel_replay: dict[str, list[int]] = field(default_factory=dict)
    reuse_historical_parallel_results: bool = True

    def should_pause(self, checkpoint_kind: str) -> bool:
        if checkpoint_kind in self.pause_points:
            return True
        prefix, _, suffix = checkpoint_kind.partition(":")
        if prefix and f"{prefix}:*" in self.pause_points:
            return True
        if suffix and "*:" + suffix in self.pause_points:
            return True
        return "*" in self.pause_points

    def request_pause(self, *checkpoint_kinds: str) -> None:
        self.pause_points.update(checkpoint_kinds)

    def clear_pause(self, *checkpoint_kinds: str) -> None:
        if not checkpoint_kinds:
            self.pause_points.clear()
            return
        for checkpoint_kind in checkpoint_kinds:
            self.pause_points.discard(checkpoint_kind)


def set_tool_checkpoint_handler(handler: Optional[Callable[..., Any]]) -> None:
    _TOOL_CONTEXT.handler = handler


def emit_tool_checkpoint(label: str, payload: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    handler = getattr(_TOOL_CONTEXT, "handler", None)
    if handler is None:
        return None
    return handler(label=label, payload=payload or {})
