import textwrap
from pathlib import Path

from ..runtime.workflow import Workflow
from ..utils.helpers import get_standard_tools


class OverviewWorkflow(Workflow):
    @classmethod
    def add_arguments(cls, group):
        super().add_common_arguments(group)
        group.add_argument(
            "--output-report",
            help="Path to save the markdown report of the codebase overview",
            default=None,
            required=False,
            type=str,
        )

    def run(self, env) -> str:
        self.logger.info("Starting codebase overview analysis")

        model = self.create_model()
        tools = get_standard_tools(env, include_decompiler=True)
        agent = self.create_agent(model, tools)

        task = textwrap.dedent(f"""
        You are a software expert performing a codebase overview analysis on this codebase.

        Your goal is to identify the main components, architecture, and potential areas of interest for more analysis.

        Process:
        1. Begin by exploring the codebase structure with `ls -la /opt/resources`
        2. Focus on entry points (web routes, sockets, etc.)
        3. Trace through the code, following data flows from inputs to sensitive operations (e.g. database queries, file writes, network calls).
        4. Identify key components, libraries used, and any unusual or complex logic.
        5. All codebases given will fall in the following categories: HTTP/web application or a network service.
            - For HTTP/web applications, identify routes, authentication mechanisms, and data handling. Ensure these are all stated and mapped out in the final report.
            - For network services, identify protocols used, authentication, and data handling (lifecycles). Ensure these are all stated and mapped out in the final report.
        6. All codebases will also have a theme from the following categories: web, cryptography, reversing, pwn. Categories may overlap (e.g. web + cryptography). Ensure your report states the theme(s) and how they relate to the codebase.
            - For web-themed codebases, pay special attention to web routes, input handling, and session management.
            - For cryptography-themed codebases, identify where and how cryptographic functions are used (do not worry about breaking them, just identify usage).
            - For reversing-themed & pwn-themed codebases, identify binaries, their purpose, and how they are used by the overall system. Use the decompilation tool to assist with this. Specifically for pwn-themed codebases, identify memory operations that a user can potentially influence/manipulate.
        7. Document your findings as you go, including file paths, function names, and relevant code snippets.

        The final message you output should be a comprehensive overview of the codebase, in markdown format, include the following sections:
        - Title: A concise title for the codebase overview (e.g., "[Web/Crypto] XYZ Overview")
        - Summary: A brief summary of the codebase's purpose and functionality.
        - Architecture: A description of the overall architecture, including key components and their interactions.
        - Key Components: A detailed list of the main components, libraries used, and their roles.
        - Data Flows: An analysis of data flows, from inputs to sensitive operations.
        - Areas of Interest: Highlight any unusual, complex, or potentially vulnerable areas that may warrant further investigation.
        - Themes: Identify the theme(s) of the codebase (web, cryptography, reversing, pwn) and explain how they relate to the codebase.
        - Steps you took to analyze the codebase, including commands run and files inspected.

        Use markdown formatting for headings, lists, tables, text format (bold, italics), and code blocks to enhance readability.

        Be thorough and methodical in your approach. Your final output should be a comprehensive overview of the codebase, highlighting its structure, key components, and any potential areas of interest for further analysis.
        """).strip()

        agent_output = agent.run(self.format_task(task, env))
        self.logger.info("Agent analysis complete")

        self.save_trace(agent)

        report = agent_output.output
        if self.kwargs.get("output_report"):
            rp = Path(self.kwargs.get("output_report"))
            rp.write_text(report or "")
            self.logger.info("Saved codebase overview report to %s", rp.resolve())

        return report


