"""
Finding Verifier Module

Provides comprehensive verification of vulnerability findings before submission,
including file path validation, required sections checking, code snippet verification,
LLM quality review, and optional kill chain validation.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ..constants import VulnerabilityFinding
from ..model import Message, MessageRole, Model

_LOG = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of a verification check."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class FindingVerifier:
    """
    Comprehensive verifier for vulnerability findings.

    Validates findings through multiple mechanisms:
    - File path existence
    - Required sections presence
    - Code snippet accuracy
    - LLM quality review
    - Kill chain execution (optional)
    """

    REQUIRED_SECTIONS = [
        "Overview",
        "Where it occurs",
        "Vulnerability Details",
        "Impact",
        "Steps to Reproduce",
        "Remediation"
    ]

    # Variants for each required section (case-insensitive matching)
    SECTION_VARIANTS = {
        "Overview": ["overview", "summary", "introduction", "description"],
        "Where it occurs": ["where it occurs", "location", "affected files", "affected code", "affected component"],
        "Vulnerability Details": ["vulnerability details", "technical details", "details", "vulnerability description"],
        "Impact": ["impact", "severity", "risk", "consequences", "business impact"],
        "Steps to Reproduce": ["steps to reproduce", "steps to exploit", "reproduction", "exploit steps", "poc", "proof of concept", "exploitation"],
        "Remediation": ["remediation", "fix", "mitigation", "recommendation", "solution", "how to fix"]
    }

    def __init__(
        self,
        code_root: str = "/opt/resources",
        verification_model: Optional[Model] = None,
        enable_kill_chain: bool = False,
        python_tool: Any = None,
        bash_tool: Any = None,
    ):
        """
        Initialize the FindingVerifier.

        Args:
            code_root: Root path to the source code being analyzed
            verification_model: LLM model for quality review (optional)
            enable_kill_chain: Whether to execute exploit code for validation
            python_tool: Tool for executing Python code (for kill chain)
            bash_tool: Tool for executing bash commands (for kill chain)
        """
        self.code_root = code_root
        self.verification_model = verification_model
        self.enable_kill_chain = enable_kill_chain
        self.python_tool = python_tool
        self.bash_tool = bash_tool

        _LOG.info("FindingVerifier initialized with code_root=%s, enable_kill_chain=%s",
                  code_root, enable_kill_chain)

    def verify(self, finding: VulnerabilityFinding) -> VerificationResult:
        """
        Run all verification checks on a finding.

        Args:
            finding: The vulnerability finding to verify

        Returns:
            Aggregated VerificationResult from all checks
        """
        _LOG.debug("Starting verification for finding: %s", finding.title)

        results = []

        # Run all verification checks
        results.append(self._verify_file_path(finding.file_path))
        results.append(self._verify_required_sections(finding.report))
        results.append(self._verify_code_snippets(finding.report, finding.file_path))

        if self.verification_model:
            results.append(self._llm_review(finding))
        else:
            _LOG.debug("Skipping LLM review - no verification model configured")

        if self.enable_kill_chain:
            results.append(self._verify_kill_chain(finding.report))
        else:
            _LOG.debug("Skipping kill chain validation - disabled")

        # Aggregate results
        aggregated = self._aggregate_results(results)

        _LOG.info("Verification completed for '%s': passed=%s, errors=%d, warnings=%d",
                  finding.title, aggregated.passed, len(aggregated.errors), len(aggregated.warnings))

        return aggregated

    def _aggregate_results(self, results: List[VerificationResult]) -> VerificationResult:
        """Aggregate multiple verification results into one."""
        all_errors = []
        all_warnings = []
        all_suggestions = []

        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_suggestions.extend(result.suggestions)

        return VerificationResult(
            passed=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            suggestions=all_suggestions
        )

    def _verify_file_path(self, file_path: str) -> VerificationResult:
        """
        Verify that the referenced file path exists in the codebase.

        Args:
            file_path: Path to the vulnerable file (relative to code_root)

        Returns:
            VerificationResult indicating if the file exists
        """
        _LOG.debug("Verifying file path: %s", file_path)

        if not file_path or file_path.strip() == "":
            return VerificationResult(
                passed=False,
                errors=["File path is empty or not provided"],
                suggestions=["Provide the relative path to the vulnerable file or network target identifier"]
            )

        # Check for network/infrastructure finding paths
        # These don't need actual file validation
        network_patterns = [
            r'^network/',  # network/192.168.1.10/smb
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP address
            r'^[a-zA-Z0-9.-]+:\d+',  # host:port
            r'^infrastructure/',  # infrastructure/dc01/
            r'^host/',  # host/server01/
            r'^target/',  # target/192.168.1.10
        ]

        for pattern in network_patterns:
            if re.match(pattern, file_path):
                _LOG.debug("Network/infrastructure path accepted: %s", file_path)
                return VerificationResult(
                    passed=True,
                    warnings=[f"Network path accepted without file validation: {file_path}"]
                )

        # Normalize the path
        clean_path = file_path.lstrip('/')
        full_path = os.path.join(self.code_root, clean_path)

        if os.path.exists(full_path):
            _LOG.debug("File path verified: %s", full_path)
            return VerificationResult(passed=True)

        # File not found - try to find similar files
        similar_files = self._find_similar_files(clean_path)

        suggestions = []
        if similar_files:
            suggestions = [f"Did you mean: {f}" for f in similar_files[:3]]

        return VerificationResult(
            passed=False,
            errors=[f"File not found: {file_path}"],
            suggestions=suggestions
        )

    def _find_similar_files(self, target_path: str) -> List[str]:
        """Find files with similar names to the target path."""
        similar = []
        target_name = os.path.basename(target_path)
        target_ext = os.path.splitext(target_name)[1]

        try:
            for root, dirs, files in os.walk(self.code_root):
                # Skip hidden directories and common non-source directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', '.git']]

                for file in files:
                    # Skip if different extension
                    if target_ext and not file.endswith(target_ext):
                        continue

                    # Calculate similarity
                    similarity = SequenceMatcher(None, target_name.lower(), file.lower()).ratio()
                    if similarity > 0.6:
                        rel_path = os.path.relpath(os.path.join(root, file), self.code_root)
                        similar.append((similarity, rel_path))

            # Sort by similarity and return top matches
            similar.sort(reverse=True, key=lambda x: x[0])
            return [path for _, path in similar[:5]]
        except Exception as e:
            _LOG.warning("Error searching for similar files: %s", e)
            return []

    def _verify_required_sections(self, report: str) -> VerificationResult:
        """
        Verify that all required sections are present in the report.

        Args:
            report: The vulnerability report content

        Returns:
            VerificationResult indicating missing sections
        """
        _LOG.debug("Verifying required sections in report")

        if not report or report.strip() == "":
            return VerificationResult(
                passed=False,
                errors=["Report is empty"],
                suggestions=["Provide a detailed vulnerability report with all required sections"]
            )

        report_lower = report.lower()
        missing_sections = []

        for section, variants in self.SECTION_VARIANTS.items():
            found = False
            for variant in variants:
                # Check for section header patterns like "## Overview" or "**Overview**" or "Overview:"
                patterns = [
                    f"#{1,6}\\s*{re.escape(variant)}",  # Markdown headers
                    f"\\*\\*{re.escape(variant)}\\*\\*",  # Bold
                    f"^{re.escape(variant)}:",  # Colon suffix
                    f"\\n{re.escape(variant)}\\n",  # Plain text on own line
                ]

                for pattern in patterns:
                    if re.search(pattern, report_lower, re.MULTILINE | re.IGNORECASE):
                        found = True
                        break

                # Also check simple substring match as fallback
                if not found and variant in report_lower:
                    found = True

                if found:
                    break

            if not found:
                missing_sections.append(section)

        if missing_sections:
            return VerificationResult(
                passed=False,
                errors=[f"Missing required section: {section}" for section in missing_sections],
                suggestions=[
                    "Ensure your report includes all required sections:",
                    "- Overview",
                    "- Where it occurs",
                    "- Vulnerability Details",
                    "- Impact",
                    "- Steps to Reproduce / Exploit",
                    "- Remediation"
                ]
            )

        _LOG.debug("All required sections found")
        return VerificationResult(passed=True)

    def _verify_code_snippets(self, report: str, file_path: str) -> VerificationResult:
        """
        Extract code snippets from the report and verify they match actual source.

        Args:
            report: The vulnerability report content
            file_path: Path to the primary vulnerable file

        Returns:
            VerificationResult indicating code snippet accuracy
        """
        _LOG.debug("Verifying code snippets in report")

        # Extract code blocks from markdown: ```language\ncode\n```
        code_block_pattern = r'```(?:(\w+))?\n(.*?)```'
        code_blocks = re.findall(code_block_pattern, report, re.DOTALL)

        if not code_blocks:
            return VerificationResult(
                passed=True,
                warnings=["No code snippets found in the report - consider adding code examples"]
            )

        errors = []
        warnings = []

        for lang, code in code_blocks:
            code = code.strip()

            # Skip very short snippets (likely just examples or commands)
            if len(code) < 20:
                continue

            # Skip shell commands, URLs, and non-code content
            if lang in ['bash', 'sh', 'shell', 'cmd', 'powershell', 'console']:
                continue

            # Try to find this code in the codebase
            found, match_info = self._search_code_in_codebase(code)

            if not found:
                # Try fuzzy matching
                partial_matches = self._fuzzy_search_code(code)
                if partial_matches:
                    warnings.append(
                        f"Code snippet may not exactly match source. Similar code found in: {partial_matches[0]}"
                    )
                else:
                    # Only report as error if it looks like actual source code
                    if self._looks_like_source_code(code, lang):
                        errors.append(f"Code snippet not found in codebase: {code[:80]}...")

        return VerificationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _looks_like_source_code(self, code: str, lang: str) -> bool:
        """Determine if a code snippet looks like actual source code vs. example/pseudocode."""
        # Skip if it looks like pseudocode or placeholder
        placeholder_patterns = [
            r'\.\.\.',
            r'<.*?>',  # Placeholder tags like <your_code_here>
            r'TODO',
            r'FIXME',
            r'example',
            r'placeholder'
        ]

        for pattern in placeholder_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False

        # Check for actual code patterns
        code_indicators = [
            r'def\s+\w+',  # Python function
            r'function\s+\w+',  # JavaScript function
            r'class\s+\w+',  # Class definition
            r'import\s+',  # Import statement
            r'from\s+\w+\s+import',  # Python import
            r'require\s*\(',  # Node.js require
            r'\w+\s*=\s*',  # Assignment
        ]

        for pattern in code_indicators:
            if re.search(pattern, code):
                return True

        return False

    def _search_code_in_codebase(self, code: str) -> Tuple[bool, Optional[str]]:
        """Search for exact code match in the codebase."""
        # Normalize whitespace for comparison
        normalized_code = ' '.join(code.split())

        try:
            for root, dirs, files in os.walk(self.code_root):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', '.git']]

                for file in files:
                    # Only search text files
                    if not self._is_source_file(file):
                        continue

                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            normalized_content = ' '.join(content.split())

                            if normalized_code in normalized_content:
                                rel_path = os.path.relpath(file_path, self.code_root)
                                return True, rel_path
                    except Exception:
                        continue
        except Exception as e:
            _LOG.warning("Error searching codebase: %s", e)

        return False, None

    def _fuzzy_search_code(self, code: str) -> List[str]:
        """Fuzzy search for similar code in the codebase."""
        matches = []

        # Extract key identifiers from the code
        identifiers = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', code))

        if len(identifiers) < 3:
            return []

        try:
            for root, dirs, files in os.walk(self.code_root):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', '.git']]

                for file in files:
                    if not self._is_source_file(file):
                        continue

                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            file_identifiers = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', content))

                            # Calculate overlap
                            overlap = len(identifiers & file_identifiers) / len(identifiers)
                            if overlap > 0.7:
                                rel_path = os.path.relpath(file_path, self.code_root)
                                matches.append((overlap, rel_path))
                    except Exception:
                        continue

            matches.sort(reverse=True, key=lambda x: x[0])
            return [path for _, path in matches[:3]]
        except Exception as e:
            _LOG.warning("Error in fuzzy search: %s", e)
            return []

    def _is_source_file(self, filename: str) -> bool:
        """Check if a file is likely a source code file."""
        source_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
            '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.sql',
            '.html', '.css', '.scss', '.vue', '.svelte', '.yaml', '.yml', '.json',
            '.xml', '.sh', '.bash', '.ps1', '.lua', '.pl', '.r'
        }
        ext = os.path.splitext(filename)[1].lower()
        return ext in source_extensions

    def _llm_review(self, finding: VulnerabilityFinding) -> VerificationResult:
        """
        Use an LLM to review the finding for quality and accuracy.

        Args:
            finding: The vulnerability finding to review

        Returns:
            VerificationResult from the LLM review
        """
        _LOG.debug("Running LLM review for finding: %s", finding.title)

        review_prompt = f"""You are a security vulnerability report reviewer. Review the following finding for quality and accuracy.

Title: {finding.title}
Severity: {finding.severity.value}
Confidence: {finding.confidence.value}
Exploitability: {finding.exploitability.value}
File Path: {finding.file_path}

Report:
{finding.report}

Summary:
{finding.summary}

Evaluate the following criteria:
1. Is the vulnerability description technically accurate and specific?
2. Are the severity, confidence, and exploitability ratings justified by the description?
3. Are the exploitation steps clear and actionable?
4. Is the remediation advice specific and implementable?
5. Are there any factual errors or inconsistencies?

Respond in JSON format:
{{
    "passed": true or false,
    "errors": ["list of critical issues that MUST be fixed before accepting"],
    "warnings": ["list of non-critical issues or improvements"],
    "suggestions": ["list of optional improvements"]
}}

Only set "passed" to false if there are critical errors. Be specific in your feedback."""

        try:
            response = self.verification_model.generate(
                messages=[Message(role=MessageRole.USER, content=review_prompt)],
                response_format={"type": "json_object"}
            )

            if not response.content:
                _LOG.warning("LLM review returned empty response")
                return VerificationResult(
                    passed=True,
                    warnings=["LLM review returned empty response"]
                )

            # Parse JSON response
            try:
                result = json.loads(response.content)
                return VerificationResult(
                    passed=result.get("passed", True),
                    errors=result.get("errors", []),
                    warnings=result.get("warnings", []),
                    suggestions=result.get("suggestions", [])
                )
            except json.JSONDecodeError as e:
                _LOG.warning("Failed to parse LLM review response as JSON: %s", e)
                return VerificationResult(
                    passed=True,
                    warnings=[f"LLM review response was not valid JSON: {response.content[:200]}"]
                )

        except Exception as e:
            _LOG.error("LLM review failed: %s", e)
            return VerificationResult(
                passed=True,
                warnings=[f"LLM review failed: {str(e)}"]
            )

    def _verify_kill_chain(self, report: str) -> VerificationResult:
        """
        Execute the exploit steps to verify the vulnerability works.

        Args:
            report: The vulnerability report containing exploit steps

        Returns:
            VerificationResult from kill chain execution
        """
        _LOG.debug("Running kill chain verification")

        if not self.python_tool and not self.bash_tool:
            return VerificationResult(
                passed=True,
                warnings=["Kill chain validation skipped - no execution tools configured"]
            )

        # Extract exploit code from the Steps to Reproduce section
        exploit_code = self._extract_exploit_code(report)

        if not exploit_code:
            return VerificationResult(
                passed=False,
                errors=["No executable exploit code found in Steps to Reproduce section"],
                suggestions=[
                    "Include a Python or bash code block with exploit code in the Steps to Reproduce section",
                    "Example: ```python\\nimport requests\\n# exploit code here\\n```"
                ]
            )

        # Execute the exploit code
        try:
            lang, code = exploit_code

            if lang == 'python' and self.python_tool:
                _LOG.info("Executing Python exploit code for verification")
                result = self.python_tool.forward(code)
            elif lang in ['bash', 'sh', 'shell'] and self.bash_tool:
                _LOG.info("Executing bash exploit code for verification")
                result = self.bash_tool.forward(code)
            else:
                return VerificationResult(
                    passed=True,
                    warnings=[f"Cannot execute {lang} code - no appropriate tool configured"]
                )

            result_str = str(result).lower()

            # Check for common failure indicators
            failure_indicators = ['error', 'traceback', 'exception', 'failed', 'denied', 'refused']

            for indicator in failure_indicators:
                if indicator in result_str:
                    return VerificationResult(
                        passed=False,
                        errors=[f"Exploit execution encountered issues: {str(result)[:500]}"],
                        suggestions=["Review and fix the exploit code to run without errors"]
                    )

            _LOG.info("Kill chain verification passed")
            return VerificationResult(
                passed=True,
                suggestions=[f"Exploit execution output: {str(result)[:200]}"]
            )

        except Exception as e:
            return VerificationResult(
                passed=False,
                errors=[f"Exploit execution error: {str(e)}"],
                suggestions=["Ensure the exploit code is syntactically correct and can run in the target environment"]
            )

    def _extract_exploit_code(self, report: str) -> Optional[Tuple[str, str]]:
        """
        Extract executable exploit code from the Steps to Reproduce section.

        Returns:
            Tuple of (language, code) or None if no executable code found
        """
        # Find the Steps to Reproduce section
        section_pattern = r'(?:steps?\s+to\s+(?:reproduce|exploit)|reproduction|exploit\s+steps?|poc|proof\s+of\s+concept)'
        section_match = re.search(section_pattern, report, re.IGNORECASE)

        if not section_match:
            return None

        # Get content after the section header until the next section
        start_pos = section_match.end()
        remaining = report[start_pos:]

        # Find code blocks in this section
        code_pattern = r'```(python|bash|sh|shell)\n(.*?)```'
        code_matches = re.findall(code_pattern, remaining, re.DOTALL | re.IGNORECASE)

        if code_matches:
            # Return the first executable code block
            lang, code = code_matches[0]
            return (lang.lower(), code.strip())

        return None
