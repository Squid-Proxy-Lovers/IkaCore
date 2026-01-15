#!/usr/bin/env python
from __future__ import annotations

import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import logging
from datetime import datetime
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from IkaCore.agents import IkaBaseAgent
from IkaCore.tools import IkaTools

from summarization import compress_messages, should_summarize

_LOG = logging.getLogger(__name__)

class RiskLevel(Enum):
    """risk levels for findings."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFORMATIONAL = "Informational"


class AgentType(Enum):
    """types of specialized pentest agents."""
    RECON = "recon"
    WEB = "web"
    WINDOWS = "windows"
    MISC = "misc"
    CODE_REVIEW = "code_review"


class AgentStage(Enum):
    """stages of pentesting for individual target agents."""
    SERVICE_DISCOVERY = "service_discovery"
    CREDENTIAL_TESTING = "credential_testing"
    CONFIG_ISSUES = "config_issues"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    POST_EXPLOITATION = "post_exploitation"
    SLEEPING = "sleeping"


@dataclass
class Finding:
    """represents a finding/vulnerability."""
    finding_id: str
    title: str
    risk: RiskLevel
    target_ip: Optional[str] = None
    service: Optional[str] = None
    description: str = ""
    remediation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    discovered_by: Optional[str] = None


@dataclass
class Target:
    """represents a network target."""
    ip: str
    hostname: Optional[str] = None
    open_ports: List[int] = field(default_factory=list)
    services: Dict[int, str] = field(default_factory=dict)
    os_guess: Optional[str] = None
    summary: str = ""
    findings: List[Finding] = field(default_factory=list)
    current_stage: AgentStage = AgentStage.SERVICE_DISCOVERY
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class NetworkSummary:
    """summary of the entire network being tested."""
    targets: Dict[str, Target]
    network_diagram: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)
    total_findings: int = 0
    
    def add_target(self, target: Target) -> None:
        """add a target to the network."""
        self.targets[target.ip] = target
    
    def get_target(self, ip: str) -> Optional[Target]:
        """get a target by IP."""
        return self.targets.get(ip)
    
    def get_findings(self) -> List[Finding]:
        """get all findings across all targets."""
        findings = []
        for target in self.targets.values():
            findings.extend(target.findings)
        return findings


@dataclass
class Task:
    """base task for agents."""
    target_path: str
    goal: str
    notes: str | None = None


class BaseSystem:
    def __init__(
        self,
        agent: IkaBaseAgent,
        env,
    ) -> None:
        self.agent = agent
        self.env = env

    @classmethod
    def from_model_and_tools(
        cls,
        model_id: str,
        api_key: str,
        tools: list[IkaTools],
        env,
        *,
        max_steps: int = 100,
        on_step_completed_callback: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        enable_summarization: bool = True,
        name: str = "Agent",
        description: str = "Agent",
        role: str = "agent",
    ) -> "BaseSystem":
        agent = IkaBaseAgent(
            name=name,
            description=description,
            prompt=system_prompt or "",
            role=role,
            tools=tools,
            model_id=model_id,
            api_key=api_key,
            maxsteps=max_steps,
            logging_level=1 if enable_summarization else 0,
        )
        return cls(agent=agent, env=env)

    def build_task_prompt(self, task: Task) -> str:
        env_prompt = self.env.get_environment_prompt() if hasattr(self.env, 'get_environment_prompt') else ""
        parts = [
            "<pentest-task>",
            f"Target path: {task.target_path}",
            f"Goal: {task.goal}",
        ]
        if task.notes:
            parts.append(f"Notes: {task.notes}")
        parts.append("</pentest-task>")
        if env_prompt:
            parts.append(env_prompt)
        return "\n".join(parts).strip()

    def run_task(self, task: Task) -> Dict[str, Any]:
        """run a task and return result."""
        prompt = self.build_task_prompt(task)
        self.agent.message_history["first_input"]["message"] = prompt
        result = self.agent.execution()
        return {
            "output": result.get("final_message", ""),
            "success": True,
            "steps": [],
        }
    
    def run(self, prompt: str) -> Dict[str, Any]:
        """run with a prompt string directly."""
        self.agent.message_history["first_input"]["message"] = prompt
        result = self.agent.execution()
        return {
            "output": result.get("final_message", ""),
            "success": True,
            "steps": [],
        }
