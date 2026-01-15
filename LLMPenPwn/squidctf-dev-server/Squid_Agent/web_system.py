from odin.agents import MultiStepAgent
from squidagent.tools.adapters import function_to_odin_tool, make_delegate_tool
from base_ctf_system import BaseCTFSystem
from squidagent.tools.ctf_tools import *
from squidagent.tools.general_tools import *
from squidagent.tools.web_tools import *
from squidagent.tools.rev_tools import *
from squidagent.tools.validation_tools import *
from odin.model import OpenAIResponsesModel, PortKeyModel, DeepSeekAPIModel

"""
Descriptions used in sub agent tool delegation
"""

TRIAGE_AGENT_DESCRIPTION = "Highly experienced web security researcher whose goal is to confirm the exploitability of vulnerabilities in CTF challenges. Validates and verifies exploit chains."

EXPLOIT_DEV_AGENT_DESCRIPTION = "Exploit developer whose goal is to develop a theoretical exploit chain for a CTF challenge. Creates exploit strategies and methodologies."

CWE_ANALYSIS_AGENT_DESCRIPTION = "Web security researcher whose goal is to analyze a CTF challenge's functionality and provide a broad list of vulnerabilities that it could potentially be vulnerable to."

VULN_RESEARCHER_AGENT_DESCRIPTION = "Web security researcher whose goal is to analyze a CTF challenge for security vulnerabilities and report them to a triager agent for verification."

SCRIPT_DEV_AGENT_DESCRIPTION = "Develops and validates exploit scripts and challenge solutions based on analysis findings. Creates exploit code, tests it locally and remotely, captures flags, and iterates based on validation feedback."


class WebSystem(BaseCTFSystem):
    """
    Agent Hierarchy Tree:
    web_manager_agent
        - script_development_agent
        - vuln_researcher_agent
            - triage_agent
        - cwe_analysis_agent
        - exploit_dev_agent
    """
    @property
    def category(self) -> str:
        return "web"
    @staticmethod
    def TOOL(func):
        return function_to_odin_tool(func)
    
    def setup_agents(self):
        TOOL = self.TOOL
        """
        Agent: triage_agent
        Description: Highly experienced web security researcher whose goal is to confirm the exploitability
        of vulnerabilities in CTF challenges. Validates and verifies exploit chains.
        """
        triage_agent = MultiStepAgent(
            model=self.model,
            tools=[
                TOOL(get_challenge_info),
                TOOL(list_ctf_files),
                TOOL(list_workspace_files),
                TOOL(run_container_command),
                TOOL(run_exploit_script),
                TOOL(list_directory),
                TOOL(read_file),
                TOOL(write_file),
                TOOL(make_web_request),
                TOOL(submit_flag),
                TOOL(run_solve_script),
                TOOL(read_files_in_directory),
                # Validation tools
                TOOL(run_exploit_and_capture_flag),
                TOOL(test_exploit_locally),
            ],
            max_steps=20,
            system_prompt=self.get_prompt("triage_agent_prompt"),
            verbose=True,
        )


        """
        Agent: exploit_dev_agent
        Description: Exploit developer whose goal is to develop a theoretical exploit chain for a CTF challenge.
        Creates exploit strategies and methodologies.
        """
        exploit_dev_agent = MultiStepAgent(
            model=self.model,
            tools=[
                TOOL(get_challenge_info),
                TOOL(list_ctf_files),
                TOOL(list_workspace_files),
                TOOL(run_container_command),
                TOOL(run_exploit_script),
                TOOL(list_directory),
                TOOL(read_file),
                TOOL(make_web_request),
                TOOL(run_solve_script),
                TOOL(read_files_in_directory),
            ],
            max_steps=12,
            system_prompt=self.get_prompt("exploit_dev_prompt"),
            verbose=True,
        )


        """
        Agent: cwe_analysis_agent
        Description: Web security researcher whose goal is to analyze a CTF challenge's functionality and
        provide a broad list of vulnerabilities that it could potentially be vulnerable to.
        """
        cwe_analysis_agent = MultiStepAgent(
           model=self.model,
            tools=[
                TOOL(list_ctf_files),
                TOOL(list_workspace_files),
                TOOL(list_directory),
                TOOL(read_file),
                TOOL(unarchive_file),
                TOOL(read_files_in_directory),
            ],
            max_steps=10,
            system_prompt=self.get_prompt("cwe_analysis_prompt"),
            verbose=True,
        )


        """
        Agent: vuln_researcher_agent
        Description: Web security researcher whose goal is to analyze a CTF challenge for security
        vulnerabilities and report them to a triager agent for verification.
        """
        vuln_researcher_agent = MultiStepAgent(
            model=self.model,
            tools=[
                TOOL(get_challenge_info),
                TOOL(list_ctf_files),
                TOOL(list_workspace_files),
                TOOL(run_container_command),
                TOOL(run_exploit_script),
                TOOL(list_directory),
                TOOL(read_file),
                TOOL(make_web_request),
                TOOL(submit_flag),
                TOOL(run_solve_script),
                TOOL(read_files_in_directory),
                # Validation tools
                TOOL(run_exploit_and_capture_flag),
                TOOL(get_remote_info),
                # sub agents:
                make_delegate_tool("triage_agent", triage_agent, TRIAGE_AGENT_DESCRIPTION),
            ],
            max_steps=20,
            system_prompt=self.get_prompt("vuln_researcher_prompt"),
            verbose=True,
        )


        """
        Agent: script_development_agent
        Description: Develops and validates exploit scripts and challenge solutions based on analysis findings.
        Creates exploit code, tests it locally and remotely, captures flags, and iterates based on
        validation feedback.
        """
        script_development_agent = MultiStepAgent(
            model=self.model,
            tools=[
                # Challenge information
                TOOL(get_challenge_info),
                TOOL(get_remote_info),
                
                # File discovery
                TOOL(list_ctf_files),
                TOOL(list_workspace_files),
                TOOL(list_directory),
                
                # Smart file reading (NEW - token efficient)
                TOOL(read_file_lines),
                TOOL(read_file_ranges),
                TOOL(get_file_outline),
                TOOL(search_in_file),
                TOOL(get_file_stats),
                TOOL(get_file_size),
                
                # Legacy file reading (kept for compatibility)
                TOOL(read_file),
                TOOL(read_files_in_directory),
                
                # Smart file writing (NEW - safe and incremental)
                TOOL(create_file_with_content),
                TOOL(patch_file_lines),
                TOOL(insert_lines),
                TOOL(delete_lines),
                TOOL(get_file_hash),
                # Command execution
                TOOL(run_container_command),
                TOOL(run_solve_script),
                TOOL(run_exploit_script),
                
                # Network operations
                TOOL(make_web_request),
                
                # Webhook tools
                TOOL(webhook_clear),
                TOOL(webhook_create),
                TOOL(webhook_view),
                TOOL(webhook_update_config),
                TOOL(webhook_get_config),
                
                # Validation and testing
                TOOL(run_exploit_and_capture_flag),
                TOOL(submit_flag),
            ],
            max_steps=20,
            system_prompt=self.get_prompt("script_development_prompt"),
            verbose=True,
        )
        """
        Agent: web_manager_agent
        Description: Top-level orchestrator for web challenges. 
        Coordinates vulnerability research, exploit development, and script development agents, manages
        information flow through RAG database, performs initial triage and file analysis, and validates
        final solutions. Responsible for overall challenge strategy and ensuring all components work
        together to solve the challenge.
        """
        self.agents['manager_agent'] = MultiStepAgent(
            model=self.model,
            tools=[
                TOOL(get_challenge_info),
                TOOL(list_ctf_files),
                TOOL(list_workspace_files),
                TOOL(list_directory),
                TOOL(read_file),
                TOOL(submit_flag),
                TOOL(unarchive_file),
                TOOL(read_files_in_directory),
                #TOOL(get_remote_info),
                # Validation tools
                TOOL(run_exploit_and_capture_flag),
                # Validation and testing
                TOOL(submit_flag),
                # sub agents:
                 make_delegate_tool("script_development_agent", script_development_agent, SCRIPT_DEV_AGENT_DESCRIPTION),
                 make_delegate_tool("vuln_researcher_agent", vuln_researcher_agent, VULN_RESEARCHER_AGENT_DESCRIPTION),
                 make_delegate_tool("cwe_analysis_agent", cwe_analysis_agent, CWE_ANALYSIS_AGENT_DESCRIPTION),
                 make_delegate_tool("exploit_dev_agent", exploit_dev_agent, EXPLOIT_DEV_AGENT_DESCRIPTION),
           ],
            max_steps=20,
            system_prompt=self.get_prompt("web_manager_prompt"),
            verbose=True,
        )
            


