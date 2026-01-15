import json
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import List

from ..constants import VulnerabilityFinding
from ..runtime.workflow import Workflow
from ..tools import AddVulnerabilityFindingTool
from ..utils.helpers import get_standard_tools


class CryptoWorkflow(Workflow):
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
            "--add-search-tool",
            help="Add a search tool to the agent",
            action="store_true",
            default=False,
        )

    def run(self, env) -> List[VulnerabilityFinding]:
        self.logger.info("Starting crypto analysis on %s", self.code_path)

        model = self.create_model(
            default_model="gpt-5",
            default_effort="high",
            add_search_tool=self.kwargs.get("add_search_tool", False),
        )

        findings_tool = AddVulnerabilityFindingTool(self.kwargs.get("finding_callback", None))
        tools = get_standard_tools(env, include_decompiler=True) + [findings_tool]

        agent = self.create_agent(model, tools)

        task = textwrap.dedent(f"""
        You are a security expert performing a static vulnerability analysis on this codebase.

        Your goal is to identify cryptography-related security vulnerabilities, especially:
        1. Weak hash functions
        2. Weak/insecure parameter choice
        3. Insecure random
        4. Bruteforceable random/secret values
        5. Use of broken / deprecated ciphers
        6. Nonce reuse/bias/leakage (when applicable)
        7. Hardcoded secret materials
        8. Insufficient key/secret/nonce/etc lengths
        9. Additional information leaks (i.e if it gave more information in a signature than needed that could be used to attack secrets)
        10. Potential use of lattice/LLL attacks on linear equations to find secret values
        11. Any other cryptographic implementation issue. you should identify schemes, learn common issues in implementation, and see if any are applicable. I.e, if there is an RSA signature implementation, search common pitfalls with RSA signatures and see if any apply. 

        For any vulnerability, find a relevant CTF writeup from past competitions.

        TRACK BITSIZES. it is ALWAYS good, given lattices, to see what values are small, what values are large, and if small values are used where large values are expected.
        Track mathematical relationships between values, especially linear relationships. These can often be exploited with lattice attacks.
        Example: If we get parts of an RSA private key, we could use lattices to recover the full key.

        Alot of (but not all) cryptography we are looking for is math heavy, so if your exploit idea requires math, you're probably on the right track.
        When presented with custom math, you should analyze the math, and look for mathematics based attacks ranging from fields like number theory, abstract algebra, etc.
        When theres math, theres often a smarter way than large bruteforce.

        Note: If there is a custom cryptographic scheme, or a custom implementation of a cryptographic scheme, it is guaranteed to be vulnerable in some form, though you must identify how to break it, and in the case of custom implementations, what is different.
        Further note: in the event of custom implementation, you should also make sure to examine potential "mouseslips" in the implementation, that could lead to it working but providing a nonstandard output. Example: Mix_columns being commented in AES.
            - This means you should be analyzing the custom implementation LINE BY LINE, flagging ANY LINE THAT DOESNT MATCH THE SPEC.

        Once you find a vulnerability, you should use your knowledge base to search for CTF writeups close to the vulnerability to provide an idea of efficient exploitation.
        I.e: the lower half of this rsa private key is leaked. searches and finds coppersmith. outlines coppersmith.

        If there is a binary, ghidra is provided and you should decompile it to analyze it.

        You should be analyzing every line that has cryptography related functionality. Including implementations and etc.
        Just because something is called "AES" does not mean it is secure, as implementations can be flawed.
        It is not sufficient to say "custom implementation of X is bad", you must identify the exact issues with the implementation.

        Your flow should at least include the following
            IF you see custom cryptography:
                1. Identify the cryptographic scheme being implemented
                If it is fully custom (i.e not a standard scheme with a custom implementation):
                    a) Analyze the math behind the scheme
                    b) Identify potential weaknesses based on known attacks against similar schemes
                    c) Look for mathematical relationships between variables that could be exploited
                    d) Search for CTF writeups or academic papers that discuss similar custom schemes and their vulnerabilities)
                If it is a standard scheme with a custom implementation:
                    a) Compare the implementation line by line against the official specification
                    b) Identify any deviations from the spec that could introduce vulnerabilities
                    c) Look for common implementation pitfalls associated with the scheme
                    d) Search for CTF writeups or academic papers that discuss similar implementation flaws and their exploitation (example: if it is AES-CBC, you would google and find it is malleable)
            IF you see standard cryptography usage:
                1. Check parameter choices against best practices
                2. Look for hardcoded secrets or keys
                3. Analyze random number generation methods
                4. Search for known vulnerabilities associated with the libraries being used
            If you see random, hashes, etc:
                1. Analyze their usage in the code
                2. Look for weak algorithms or configurations

        For EVERY FINDING, you should have at least 1 relevant CTF writeup acquired via the search tool.

        Furthermore, note WE HAVE 2 MINUTES TO RUN AN EXPLOIT. 48 bit brutes, for example, are real world practical but not acceptable as is here (should still be noted, but further analysis for reductions is critical)          

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
                - CTF writeup(s)
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
