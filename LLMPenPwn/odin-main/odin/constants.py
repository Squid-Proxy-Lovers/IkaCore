import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

_LOG = logging.getLogger(__name__)

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CERTAIN = "certain"

class SeverityLevel(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExploitabilityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass(slots=True)
class VulnerabilityFinding:
    title: str
    report: str
    summary: str
    file_path: str
    confidence: ConfidenceLevel
    severity: SeverityLevel
    exploitability: ExploitabilityLevel