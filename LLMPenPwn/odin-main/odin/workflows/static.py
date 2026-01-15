import json
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from ..constants import VulnerabilityFinding
from ..model import OpenAIModel
from ..runtime.workflow import Workflow
from ..tools import AddVulnerabilityFindingTool
from ..utils.helpers import get_standard_tools


class StaticWorkflow(Workflow):
    @classmethod
    def add_arguments(cls, group):
        super().add_common_arguments(group)
        group.add_argument(
            "--output-findings",
            help="Path to save findings (default: findings.json)",
            required=False,
            type=str,
        )
        group.add_argument(
            "--enable-verification",
            help="Enable finding verification (default: True)",
            action="store_true",
            default=True,
        )
        group.add_argument(
            "--disable-verification",
            help="Disable finding verification",
            action="store_false",
            dest="enable_verification",
        )
        group.add_argument(
            "--enable-kill-chain",
            help="Enable kill chain validation (executes exploit code in sandbox)",
            action="store_true",
            default=False,
        )
        group.add_argument(
            "--max-verification-retries",
            help="Maximum verification retry attempts before accepting finding (default: 3)",
            type=int,
            default=3,
        )
        group.add_argument(
            "--verification-model",
            help="Model to use for verification LLM review (default: gpt-5-nano)",
            type=str,
            default="gpt-5-nano",
        )

    def run(self, env) -> List[VulnerabilityFinding]:
        self.logger.info("Starting static analysis on %s", self.code_path)

        model = self.create_model()

        # Configure verification
        enable_verification = self.kwargs.get("enable_verification", True)
        enable_kill_chain = self.kwargs.get("enable_kill_chain", False)
        max_verification_retries = self.kwargs.get("max_verification_retries", 3)
        verification_model_id = self.kwargs.get("verification_model", "gpt-5-nano")

        # Create verification model if verification is enabled
        verification_model = None
        if enable_verification:
            try:
                verification_model = OpenAIModel(model_id=verification_model_id)
                self.logger.info("Using verification model: %s", verification_model_id)
            except Exception as e:
                self.logger.warning("Failed to create verification model: %s. Continuing without LLM review.", e)
                verification_model = None

        # Get execution tools for kill chain validation if enabled
        python_tool = None
        bash_tool = None
        if enable_kill_chain and env:
            try:
                python_tool = env.get_python_tool() if hasattr(env, 'get_python_tool') else None
                bash_tool = env.get_bash_tool() if hasattr(env, 'get_bash_tool') else None
                self.logger.info("Kill chain validation enabled with python_tool=%s, bash_tool=%s",
                                 python_tool is not None, bash_tool is not None)
            except Exception as e:
                self.logger.warning("Failed to get execution tools for kill chain: %s", e)

        # Create finding tool with verification
        findings_tool = AddVulnerabilityFindingTool(
            finding_callback=self.kwargs.get("finding_callback", None),
            verification_model=verification_model,
            code_root="/opt/resources",
            enable_verification=enable_verification,
            enable_kill_chain=enable_kill_chain,
            max_verification_retries=max_verification_retries,
            python_tool=python_tool,
            bash_tool=bash_tool,
        )

        self.logger.info("Finding verification: enabled=%s, kill_chain=%s, max_retries=%d",
                         enable_verification, enable_kill_chain, max_verification_retries)

        tools = get_standard_tools(env, include_decompiler=True) + [findings_tool]

        agent = self.create_agent(model, tools, max_steps=self.kwargs.get("max_steps", 100))

        task = textwrap.dedent(f"""
        You are a security expert performing a static vulnerability analysis on this codebase.

        Your goal is to identify security vulnerabilities, especially:
        1. Injection vulnerabilities (SQL, command, etc.)
        2. Authentication and authorization flaws
        3. Sensitive data exposure
        4. XML External Entities (XXE)
        5. Broken access control
        6. Cross-Site Scripting (XSS)
        7. Insecure deserialization
        8. Using components with known vulnerabilities
        9. Insecure cryptographic functions
        
        Process:
        1. Begin by exploring the codebase structure with `ls -la /opt/resources`
        2. Focus on entry points (web routes, sockets, etc.)
        3. Trace through the code, following data flows from inputs to sensitive operations
        4. When you find a vulnerability, use record it using the `add_vulnerability_finding`.
            - IMPORTANT: Ensure that the finding includes a detailed description using **markdown**, and it includes code snippets or file paths where relevant. Include as much detail as possible to help a developer understand and fix the issue. Additionally include high level steps to exploit the vulnerability in order to steal information. Do not include a references section. A good description has about around 600 words with these required sections:
                - Overview
                - Where it occurs
                - Vulnerability Details
                - Impact
                - Steps to Reproduce / Exploit
                - Remediation
            - IMPORTANT: Ensure all code snippets are formatted in markdown using triple backticks and the appropriate language (e.g. ```python).
        5. In the end, return a summary of all findings.

        Be thorough and look for subtle issues. Provide concrete, actionable findings.
        """).strip()

        if self.kwargs.get("additional_task_info"):
            task += "\n\n" + self.kwargs.get("additional_task_info").strip()

        agent_output = agent.run(self.format_task(task, env))
        self.logger.info("Agent analysis complete")

        self.save_trace(agent)

        if self.kwargs.get("output_findings"):
            findings_path = Path(self.kwargs.get("output_findings"))
            findings_path.write_text(json.dumps([asdict(f) for f in findings_tool.findings], indent=2), encoding="utf-8")

        return findings_tool.findings


