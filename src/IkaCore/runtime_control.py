"""Explicit snapshot pause points and replay-selection controls."""

# pyright: strict

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from IkaModel.execution_hooks import emit_boundary
from IkaModel.runtime_errors import IkaRequestControlError


class RuntimePauseRequested(IkaRequestControlError):
    def __init__(self, checkpoint_kind: str, snapshot_id: Optional[str] = None, frame_id: Optional[str] = None):
        super().__init__(f"Runtime paused at checkpoint '{checkpoint_kind}'")
        self.checkpoint_kind = checkpoint_kind
        self.snapshot_id = snapshot_id
        self.frame_id = frame_id


def _pause_points() -> set[str]:
    return set()


def _parallel_replay() -> dict[str, list[int]]:
    return {}


@dataclass
class RuntimeControl:
    pause_points: set[str] = field(default_factory=_pause_points)
    parallel_replay: dict[str, list[int]] = field(default_factory=_parallel_replay)
    reuse_historical_parallel_results: bool = True

    def should_pause(self, checkpoint_kind: str) -> bool:
        prefix, _, suffix = checkpoint_kind.partition(":")
        return (checkpoint_kind in self.pause_points or "*" in self.pause_points
                or f"{prefix}:*" in self.pause_points or (bool(suffix) and f"*:{suffix}" in self.pause_points))

    def request_pause(self, *checkpoint_kinds: str) -> None:
        self.pause_points.update(checkpoint_kinds)

    def clear_pause(self, *checkpoint_kinds: str) -> None:
        if checkpoint_kinds:
            self.pause_points.difference_update(checkpoint_kinds)
        else:
            self.pause_points.clear()


def emit_tool_checkpoint(label: str, payload: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    snapshot_id = emit_boundary("tool:" + label, payload)
    return {"snapshot_id": snapshot_id} if snapshot_id else None
