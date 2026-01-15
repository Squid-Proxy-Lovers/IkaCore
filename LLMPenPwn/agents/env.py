#!/usr/bin/env python3
"""
Simple environment wrapper for host-based execution (no containers).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HostEnvironment:
    """Simple environment for host-based execution (no containers)."""
    
    def __init__(self, workdir: Optional[str] = None):
        self.workdir = workdir or "."
    
    def get_environment_prompt(self) -> str:
        """Return environment description for agents."""
        return f"""You are running on the host system. Working directory: {self.workdir}
You have access to shell commands and can run tools like nmap, curl, etc. directly on the system."""

