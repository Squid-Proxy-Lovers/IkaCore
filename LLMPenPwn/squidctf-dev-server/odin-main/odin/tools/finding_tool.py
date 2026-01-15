import logging
import textwrap
from typing import Callable, List

from ..constants import (ConfidenceLevel, ExploitabilityLevel, SeverityLevel,
                         VulnerabilityFinding)
from .base import Tool

_LOG = logging.getLogger(__name__)

class AddVulnerabilityFindingTool(Tool):
    name = "add_vulnerability_finding"
    description = textwrap.dedent("""
    Add a security vulnerability finding to the collection.

    Args:
        title (str): Brief title describing the vulnerability
        report (str): Detailed report of the vulnerability
        summary (str): A brief summary of the vulnerability
        file_path (str): Path to the vulnerable file (relative to code root)
        severity (str, Enum): Describes the severity of the vulnerability (INFO, LOW, MEDIUM, HIGH, CRITICAL)
        confidence (str, Enum): Describes the confidence level of the finding (LOW, MEDIUM, HIGH, CERTAIN)
            - IMPORTANT: Ensure that the finding includes a detailed description using markdown, and it includes code snippets or file paths where relevant. Include as much detail as possible to help a developer understand and fix the issue. Additionally include high level steps to exploit the vulnerability in order to steal information. Do not include a references section. A good description has about around 600 words with these required sections:
                - Overview
                - Where it occurs
                - Vulnerability Details
                - Impact
                - Steps to Reproduce / Exploit
                - Remediation
        exploitability (str, Enum): Describes the exploitability level of the finding, ensure you consider how easily the vulnerability can be exploited from an attacker's perspective (LOW, MEDIUM, HIGH, CERTAIN)

    Returns:
        Confirmation message that the finding was logged.
    """)
    inputs = {
        "title": {
            "type": "string",
            "description": "Brief title describing the vulnerability"
        },
        "report": {
            "type": "string",
            "description": "Detailed explanation of the vulnerability"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the vulnerability (1-3 sentences max)"
        },
        "file_path": {
            "type": "string",
            "description": "Path to the vulnerable file (relative to code root)"
        },
        "severity": {
            "type": "string",
            "description": "Describes the severity of the vulnerability (INFO, LOW, MEDIUM, HIGH, CRITICAL)",
            "enum": ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        },
        "confidence": {
            "type": "string",
            "description": "Describes the confidence level of the finding (LOW, MEDIUM, HIGH, CERTAIN)",
            "enum": ["LOW", "MEDIUM", "HIGH", "CERTAIN"]
        },
        "exploitability": {
            "type": "string",
            "description": "Describes the exploitability level of the finding, ensure you consider how easily the vulnerability can be exploited from an attacker's perspective (LOW, MEDIUM, HIGH, CERTAIN)",
            "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        }
    }
    output_type = "string"

    def __init__(self, finding_callback: Callable = None):
        super().__init__()
        self.findings = []
        # TODO: fallback need to be non-blocking
        self.finding_callback = finding_callback

    def forward(
        self,
        title: str,
        report: str,
        summary: str,
        file_path: str,
        severity: str,
        confidence: str,
        exploitability: str
    ) -> str:
        print(f"Adding finding: {title}, severity={severity}, confidence={confidence}, exploitability={exploitability}")
        finding = VulnerabilityFinding(
            title=title,
            report=report,
            summary=summary,
            file_path=file_path,
            severity=SeverityLevel[severity],
            confidence=ConfidenceLevel[confidence],
            exploitability=ExploitabilityLevel[exploitability]
        )

        if self.finding_callback:
            self.finding_callback(finding)

        self.findings.append(finding)

        _LOG.debug("Added finding: %s (severity=%s)", finding.title, finding.severity)
        return f"Added finding: {finding.title} (severity={finding.severity})"