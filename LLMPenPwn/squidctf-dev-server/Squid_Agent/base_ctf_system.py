#!/usr/bin/env python3

import functools
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

"""
Base system for category-specific Odin agents.
"""

from squid_logging import run_log

os.makedirs(os.path.join(os.getcwd(), 'logs_squidagent'), exist_ok=True)
logging.basicConfig(filename=os.path.join(os.getcwd(), 'logs_squidagent', 'base_ctf_system.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseCTFSystem(ABC):
    """Base class for CTF category-specific systems."""
    
    def __init__(self, model, api_key: str = None, anthropic_api_key: str = None, use_portkey: bool = True):
        self.use_portkey = use_portkey
        
        # Portkey configuration
        if use_portkey:
            self.portkey_base_url = "https://localhost:8080/v1"
            self.portkey_api_key = "IHjviXWEzquA5MQPwTCloki8ViqR"
        
        self.model = model
        self.api_key = api_key
        self.anthropic_api_key = anthropic_api_key
        self.agents = {}
        self.setup_agents()

        # Per-challenge cost tracking (in USD)
        self._costs: dict[str, float] = {}
        self._current_challenge: str | None = None

        # Attach step-completed callback to all agents so we can aggregate model costs
        for a in self.agents.values():
            try:
                a.on_step_completed_callback = self._on_step_completed
            except Exception:
                # Some agent instances may not accept callbacks; ignore safely
                pass

        # Rate limiting configuration
        self.min_request_interval = 4.0  # Minimum seconds between requests
        self.last_request_time = None
        self.request_lock = threading.Lock()

    def create_model(self, model_id: str):
        """
        Create a LiteLLM model instance with optional Portkey routing.
        
        Args:
            model_id: The model identifier (e.g., "gpt-4", "claude-sonnet-4-5")
        
        Returns:
            LiteLLMModel configured with or without Portkey
        """
        from smolagents import LiteLLMModel
        
        if self.use_portkey:
            logger.info(f"Creating model {model_id} with Portkey routing")
            return LiteLLMModel(
                model_id=model_id,
                api_base=self.portkey_base_url,
                api_key=self.portkey_api_key,
            )
        else:
            logger.info(f"Creating model {model_id} without Portkey")
            return LiteLLMModel(
                model_id=model_id,
                api_key=self.api_key
            )

    def wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        with self.request_lock:
            if self.last_request_time:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_request_interval:
                    sleep_time = self.min_request_interval - elapsed
                    logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
            self.last_request_time = time.time()
            
    def get_prompt(self, prompt_name: str) -> str:
        """Load prompt template from file."""
        prompt_path = Path(__file__).parent / ".." / "squidagent" / "prompts" / self.category / prompt_name
        with open(prompt_path, 'r') as f:
            template_content = f.read()
        return template_content
    
    @abstractmethod
    def setup_agents(self):
        """Setup agents specific to this CTF category."""
        pass
    
    def get_manager_agent(self):
        """Get the main manager agent for this category."""
        return self.agents.get('manager_agent')
    
    def get_all_agents(self) -> list:
        """Get all agents in this system."""
        return list(self.agents.values())
    
    def _on_step_completed(self, trace: dict):
        """
        Callback called by agents after each step completes.
        `trace` is the dict produced by asdict(self.get_trace()) from an Agent.
        This extracts token usage costs (if present) and accumulates by current challenge.
        """
        try:
            steps = trace.get("steps", [])
            if not steps:
                return

            last = steps[-1]
            msg = last.get("message") or {}
            tu = msg.get("token_usage") or {}

            input_cost = float(tu.get("input_cost", 0.0) or 0.0)
            output_cost = float(tu.get("output_cost", 0.0) or 0.0)
            step_cost = input_cost + output_cost

            key = self._current_challenge or "unknown"
            total = self._costs.get(key, 0.0) + step_cost
            self._costs[key] = total

            logger.info("Cost update for challenge '%s': step cost=$%.6f, total so far=$%.6f", key, step_cost, total)
            # Print a short, user-friendly cost update to stdout
            print(f"[COST] Challenge '{key}': spent ${total:.6f} (+${step_cost:.6f})")
        except Exception as e:
            logger.debug("Failed to process cost update: %s", e)
    
    def solve_challenge(self, challenge, environment, title_for_logging: str = None) -> dict:
        """Solve a challenge using this system's agents."""
        results = {
            "challenge_name": challenge.name,
            "category": challenge.category,
            "solved": False,
            "flag_submitted": None,
            "error": None,
            "agent_output": None,
            "cost": 0.0,
        }
        
        challenge_key = title_for_logging or challenge.name
        self._current_challenge = challenge_key
        initial_cost = self._costs.get(challenge_key, 0.0)
        
        self.wait_for_rate_limit()
        manager_agent = self.get_manager_agent()
        if not manager_agent:
            results["error"] = "No manager agent found for this system"
            return results
        
        prompt = self._create_challenge_prompt(challenge)
        logger.info(f"Using {getattr(manager_agent, 'name', 'manager_agent')} for challenge: {challenge.name}")
        logger.info(f"Challenge prompt: {prompt}")
        
        try:
            agent_result = manager_agent.run(prompt)
            try:
                results["agent_output"] = getattr(agent_result, "output", str(agent_result))
                if hasattr(agent_result, "error") and agent_result.error:
                    error_str = str(agent_result.error)
                    logger.warning(f"Agent returned with error: {error_str}")
                    print(f"[ERROR] Agent returned with error: {error_str}")
                    results["error"] = error_str
                    if "maximum steps" in error_str.lower() or "max steps" in error_str.lower():
                        logger.warning("Agent hit max_steps limit - may need to increase max_steps or fix loop")
                        print(f"[WARNING] Agent hit max_steps limit - may need to increase max_steps or fix loop")
                if hasattr(agent_result, "success") and not agent_result.success:
                    logger.warning(f"Agent run was not successful")
            except Exception as e:
                logger.error(f"Error extracting agent output: {e}")
                results["agent_output"] = str(agent_result)
        except Exception as e:
            error_msg = f"Agent run failed with exception: {e}"
            logger.error(error_msg, exc_info=True)
            results["error"] = error_msg
            results["agent_output"] = f"Agent execution failed: {e}"

        challenge_info = environment.get_challenge_info()
        results["solved"] = challenge_info.get("solved")
        results["flag_submitted"] = challenge_info.get("flag_submitted")

        # Autonomous follow-ups: keep going until solved or iteration cap
        # Track tool call sequences across runs to detect loops and inject warnings
        iteration_cap = 5
        n = 0
        tool_call_sequences = []  # Track sequences from each run
        loop_warning_injected = False
        
        # Check initial run for problematic patterns
        try:
            agent_trace = manager_agent.get_trace()
            recent_steps = agent_trace.get("steps", [])[-10:] if hasattr(manager_agent, 'get_trace') else []
            initial_run_tools = []
            for step in recent_steps:
                tool_calls = step.get("tool_calls", [])
                for tc in tool_calls:
                    tool_name = tc.get("name", "")
                    if tool_name:
                        initial_run_tools.append(tool_name)
            if initial_run_tools:
                tool_call_sequences.append(tuple(initial_run_tools))
        except Exception as e:
            logger.debug(f"Could not check initial agent trace: {e}")
        
        while not results["solved"] and n < iteration_cap:
            n += 1
            logger.info(f"Follow-up iteration {n}/{iteration_cap}")
            
            # Check for loops BEFORE sending follow-up to inject warning if needed
            loop_warning = None
            try:
                agent_trace = manager_agent.get_trace()
                recent_steps = agent_trace.get("steps", [])[-10:] if hasattr(manager_agent, 'get_trace') else []
                current_run_tools = []
                for step in recent_steps:
                    tool_calls = step.get("tool_calls", [])
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        if tool_name:
                            current_run_tools.append(tool_name)
                
                # Only track non-empty sequences
                if current_run_tools:
                    tool_call_sequences.append(tuple(current_run_tools))
                    
                    # Detect if we're repeating the same sequence (loop detection)
                    if len(tool_call_sequences) >= 2:
                        last_sequence = tool_call_sequences[-1]
                        # Check if this sequence appeared in the last 2 runs
                        if tool_call_sequences.count(last_sequence) >= 2:
                            logger.warning(f"Loop detected: agent repeating tool call sequence {list(last_sequence)}")
                            loop_warning = (
                                f"CRITICAL: You are stuck in a loop repeating the same tool calls: {list(last_sequence)}. "
                                f"This pattern has been repeated {tool_call_sequences.count(last_sequence)} times. "
                                f"You MUST break this loop immediately by:\n"
                                f"1. If file_info is failing, skip it and proceed directly to calling binary_analysis_agent\n"
                                f"2. Do NOT call get_challenge_info, list_ctf_files, and file_info again\n"
                                f"3. Instead, immediately call binary_analysis_agent with the information you already have\n"
                                f"4. If you need file information, use checksec or strings_analysis instead, or proceed without it"
                            )
                    
                    # Also check for the specific problematic pattern: get_challenge_info, list_ctf_files, file_info
                    if len(current_run_tools) >= 3:
                        pattern = tuple(current_run_tools[:3])
                        if pattern == ("get_challenge_info", "list_ctf_files", "file_info"):
                            pattern_count = sum(1 for seq in tool_call_sequences if len(seq) >= 3 and seq[:3] == pattern)
                            if pattern_count >= 2:
                                logger.warning(f"Loop detected: agent stuck in initial reconnaissance pattern {list(pattern)}")
                                loop_warning = (
                                    f"CRITICAL LOOP DETECTED: You have called get_challenge_info, list_ctf_files, and file_info "
                                    f"in sequence {pattern_count} times. The file_info tool may be failing or hanging. "
                                    f"You MUST:\n"
                                    f"1. STOP calling file_info - it appears to be failing\n"
                                    f"2. Proceed immediately to call binary_analysis_agent with what you know\n"
                                    f"3. You already have the challenge info and file list - that's enough to start analysis\n"
                                    f"4. If you need file type info, try checksec or proceed without it - binary_analysis_agent can handle it"
                                )
            except Exception as e:
                logger.debug(f"Could not check agent trace for loop detection: {e}")
            
            # Build follow-up message with loop warning if detected
            if loop_warning and not loop_warning_injected:
                followup = loop_warning + "\n\n" + (
                    "Continue solving autonomously. Use available tools to probe and exploit. "
                    "If you created any JSON briefs, immediately call the appropriate delegate tool "
                    "(e.g., crypto_vulnerability_agent, code_review_agent, crypto_script_agent) instead of stopping. "
                    "Do not request user input."
                )
                loop_warning_injected = True
                logger.info("Injecting loop warning into agent context")
            else:
                followup = (
                    "Continue solving autonomously. Use available tools to probe and exploit. "
                    "If you created any JSON briefs, immediately call the appropriate delegate tool "
                    "(e.g., crypto_vulnerability_agent, code_review_agent, crypto_script_agent) instead of stopping. "
                    "Do not request user input."
                )
            
            # Add a small delay to ensure previous tool calls have completed
            time.sleep(0.5)
            
            try:
                agent_result = manager_agent.run(followup)
                try:
                    results["agent_output"] = getattr(agent_result, "output", str(agent_result))
                    if hasattr(agent_result, "error") and agent_result.error:
                        error_str = str(agent_result.error)
                        logger.warning(f"Agent returned with error in follow-up {n}: {error_str}")
                        print(f"[ERROR] Agent returned with error in follow-up {n}: {error_str}")
                        if not results.get("error"):
                            results["error"] = error_str
                        if "maximum steps" in error_str.lower() or "max steps" in error_str.lower():
                            logger.warning(f"Agent hit max_steps limit in follow-up {n} - breaking loop")
                            print(f"[WARNING] Agent hit max_steps limit in follow-up {n} - breaking loop")
                            break
                    if hasattr(agent_result, "success") and not agent_result.success:
                        logger.warning(f"Agent follow-up {n} was not successful")
                except Exception as e:
                    logger.error(f"Error extracting agent output in follow-up {n}: {e}")
                    results["agent_output"] = str(agent_result)
            except Exception as e:
                error_msg = f"Agent follow-up {n} failed with exception: {e}"
                logger.error(error_msg, exc_info=True)
                if not results.get("error"):
                    results["error"] = error_msg
                results["agent_output"] = f"Agent execution failed in follow-up {n}: {e}"
                break

            challenge_info = environment.get_challenge_info()
            results["solved"] = challenge_info.get("solved")
            results["flag_submitted"] = challenge_info.get("flag_submitted")
            
            # If challenge is solved, break immediately
            if results["solved"]:
                break

        # Log why we're exiting the loop
        if not results["solved"]:
            if n >= iteration_cap:
                logger.warning(f"Challenge '{challenge_key}' failed: Exhausted all {iteration_cap} follow-up iterations without solving")
                print(f"[WARNING] Challenge '{challenge_key}' failed: Exhausted all {iteration_cap} follow-up iterations without solving")
                if not results.get("error"):
                    results["error"] = f"Exhausted all {iteration_cap} follow-up iterations without solving the challenge"
            else:
                logger.info(f"Challenge '{challenge_key}' exited follow-up loop after {n} iterations (solved={results['solved']})")
                print(f"[INFO] Challenge '{challenge_key}' exited follow-up loop after {n} iterations (solved={results['solved']})")

        final_cost = self._costs.get(challenge_key, 0.0)
        results["cost"] = final_cost - initial_cost
        logger.info(f"Challenge '{challenge_key}' total cost: ${results['cost']:.6f}")
        
        return results
    
    def _create_challenge_prompt(self, challenge) -> str:
        """Create an initial prompt for the manager agent."""
        prompt_file = f"{self.category}_manager_prompt"
        return self.get_prompt(prompt_file)
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Return the CTF category this system handles."""
        pass