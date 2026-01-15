import logging
import textwrap
from typing import Any, Callable, List, Optional

from ..constants import (ConfidenceLevel, ExploitabilityLevel, SeverityLevel,
                         VulnerabilityFinding)
from ..model import Model
from .base import Tool
from .finding_verifier import FindingVerifier, VerificationResult

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

    def __init__(
        self,
        finding_callback: Callable = None,
        verifier: Optional[FindingVerifier] = None,
        verification_model: Optional[Model] = None,
        code_root: str = "/opt/resources",
        enable_verification: bool = True,
        enable_kill_chain: bool = False,
        max_verification_retries: int = 3,
        python_tool: Any = None,
        bash_tool: Any = None,
    ):
        """
        Initialize the AddVulnerabilityFindingTool.

        Args:
            finding_callback: Callback function invoked when a finding is added
            verifier: Pre-configured FindingVerifier instance (optional)
            verification_model: LLM model for verification review (optional)
            code_root: Root path to source code being analyzed
            enable_verification: Whether to enable finding verification
            enable_kill_chain: Whether to execute exploit code for validation
            max_verification_retries: Max retry attempts before accepting finding
            python_tool: Tool for executing Python code (for kill chain)
            bash_tool: Tool for executing bash commands (for kill chain)
        """
        super().__init__()
        self.findings: List[VulnerabilityFinding] = []
        self.finding_callback = finding_callback
        self.enable_verification = enable_verification
        self.max_verification_retries = max_verification_retries

        # Track retry attempts per finding (by title+file_path hash)
        self._retry_counts: dict[int, int] = {}

        # Initialize verifier if verification is enabled
        if enable_verification:
            self.verifier = verifier or FindingVerifier(
                code_root=code_root,
                verification_model=verification_model,
                enable_kill_chain=enable_kill_chain,
                python_tool=python_tool,
                bash_tool=bash_tool,
            )
            _LOG.info("Finding verification enabled with max_retries=%d, kill_chain=%s",
                      max_verification_retries, enable_kill_chain)
        else:
            self.verifier = None
            _LOG.info("Finding verification disabled")

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
        """
        Add a vulnerability finding, optionally with verification.

        If verification is enabled, the finding will be validated before acceptance.
        If verification fails, an error message is returned prompting correction.
        After max_verification_retries failed attempts, the finding is accepted with warnings.

        Returns:
            Success message if finding accepted, or error message if verification failed
        """
        _LOG.info("Processing finding: %s, severity=%s, confidence=%s, exploitability=%s",
                  title, severity, confidence, exploitability)

        # Create the finding object
        finding = VulnerabilityFinding(
            title=title,
            report=report,
            summary=summary,
            file_path=file_path,
            severity=SeverityLevel[severity],
            confidence=ConfidenceLevel[confidence],
            exploitability=ExploitabilityLevel[exploitability]
        )

        # Run verification if enabled
        if self.verifier:
            # Create a hash key for tracking retries on this specific finding
            finding_key = hash((title, file_path))
            current_retries = self._retry_counts.get(finding_key, 0)

            verification_result = self.verifier.verify(finding)

            if not verification_result.passed:
                if current_retries >= self.max_verification_retries:
                    # Max retries reached - accept the finding but log warnings
                    _LOG.warning(
                        "Max verification retries (%d) reached for finding: %s. Accepting with warnings.",
                        self.max_verification_retries, title
                    )
                    warning_msg = self._format_verification_warnings(verification_result)
                    _LOG.warning("Verification warnings for '%s': %s", title, warning_msg)
                    # Fall through to accept the finding
                else:
                    # Increment retry count and return error message
                    self._retry_counts[finding_key] = current_retries + 1
                    return self._format_verification_failure(verification_result, current_retries + 1)

        # Verification passed, disabled, or max retries exceeded - add the finding
        if self.finding_callback:
            self.finding_callback(finding)

        self.findings.append(finding)

        _LOG.info("Successfully added finding: %s (severity=%s)", finding.title, finding.severity)
        return f"Successfully added finding: {finding.title} (severity={finding.severity})"

    def _format_verification_failure(self, result: VerificationResult, retry_num: int) -> str:
        """Format a verification failure message for the agent."""
        message_parts = [
            f"VERIFICATION FAILED (attempt {retry_num}/{self.max_verification_retries})",
            "",
            "The finding was NOT accepted due to the following issues:",
            ""
        ]

        if result.errors:
            message_parts.append("ERRORS (must fix):")
            for error in result.errors:
                message_parts.append(f"  - {error}")
            message_parts.append("")

        if result.warnings:
            message_parts.append("WARNINGS:")
            for warning in result.warnings:
                message_parts.append(f"  - {warning}")
            message_parts.append("")

        if result.suggestions:
            message_parts.append("SUGGESTIONS:")
            for suggestion in result.suggestions:
                message_parts.append(f"  - {suggestion}")
            message_parts.append("")

        message_parts.append("Please fix the issues above and resubmit the finding with corrected information.")

        return "\n".join(message_parts)

    def _format_verification_warnings(self, result: VerificationResult) -> str:
        """Format warnings for logging when max retries reached."""
        parts = []
        if result.errors:
            parts.append(f"Errors: {', '.join(result.errors)}")
        if result.warnings:
            parts.append(f"Warnings: {', '.join(result.warnings)}")
        return "; ".join(parts) if parts else "None"