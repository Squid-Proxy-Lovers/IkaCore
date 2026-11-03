# pyright: strict

import queue
import threading
from pathlib import Path
from typing import Optional

from IkaCore.logging_utils_support import (
    LoggerConversationEventsMixin,
    LoggerCostMixin,
    LoggerHitlEventsMixin,
    LoggerStageEventsMixin,
    LoggerWriteApiMixin,
    WriteItem,
)


class IkaLogger(
    LoggerCostMixin,
    LoggerHitlEventsMixin,
    LoggerStageEventsMixin,
    LoggerConversationEventsMixin,
    LoggerWriteApiMixin,
):
    """level 0: no logging; 1: file; 2: JSON file."""

    def __init__(
        self,
        level: int = 0,
        log_file: str = "logs.txt",
        use_colors: bool = True,
        show_usage_level0: bool = True,
        log_input_enabled: bool = False,
    ) -> None:
        self.level = level
        self.log_file = Path(log_file)
        self.use_colors = use_colors
        self.show_usage_level0 = show_usage_level0
        self.log_input_enabled = log_input_enabled

        self._queue: Optional[queue.Queue[Optional[WriteItem]]] = None
        self._shutdown = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._atexit_registered = False

        if self.level != 0:
            self._ensure_writer()
