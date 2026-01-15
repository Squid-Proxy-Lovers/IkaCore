#!/usr/bin/env python3
"""
Manager classes for coordinating multi-agent penetration testing.
"""

import logging
import threading
import time
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum

from base import BaseSystem, Target, AgentType, AgentStage

# Import local memory module explicitly to avoid conflict with IkaMem.memory
_local_dir = Path(__file__).parent
_memory_spec = importlib.util.spec_from_file_location("local_memory", _local_dir / "memory.py")
_local_memory = importlib.util.module_from_spec(_memory_spec)
_memory_spec.loader.exec_module(_local_memory)
SharedMemory = _local_memory.SharedMemory

logger = logging.getLogger(__name__)


class ReconAgent:
    """Recon agent for network discovery and initial scanning."""
    
    def __init__(self, system: BaseSystem, memory: SharedMemory):
        self.system = system
        self.memory = memory
    
    def run_recon(self, subnet: str) -> Dict[str, Target]:
        """Run reconnaissance on a subnet."""
        logger.info(f"Starting recon on subnet: {subnet}")
        
        # TODO: Use system to run nmap/echo scan
        # This is a placeholder - the actual implementation would use
        # the system's tools (BashTool) to run nmap commands
        
        
        task = Task(
            target_path=subnet,
            goal=f"Perform network reconnaissance on {subnet}. Use nmap to discover hosts and open ports. Create a network summary.",
            notes="Use nmap to scan the subnet. Identify all live hosts, open ports, and services."
        )
        
        result = self.system.run_task(task)
        
        # Parse results and store in memory
        # For now, this is a placeholder - you'd parse the agent's output
        # and create Target objects
        
        return {}
    
    def generate_network_summary(self) -> str:
        """Generate network diagram/summary."""
        targets = self.memory.get_all_targets()
        summary_json = self.memory.get_network_summary_json()
        
        # Generate a simple text summary (could be mermaid diagram)
        summary = "Network Summary:\n"
        for ip, data in summary_json.items():
            summary += f"\n{ip} ({data.get('hostname', 'unknown')}):\n"
            summary += f"  Ports: {data.get('open_ports', [])}\n"
            summary += f"  Services: {data.get('services', {})}\n"
            summary += f"  Stage: {data.get('current_stage', 'unknown')}\n"
        
        self.memory.set_network_diagram(summary)
        return summary


class TargetManager:
    """Manages multiple agents for a single IP target."""
    
    def __init__(self, target_ip: str, memory: SharedMemory, 
                 agent_systems: Dict[AgentType, BaseSystem]):
        self.target_ip = target_ip
        self.memory = memory
        self.agent_systems = agent_systems  # AgentType -> BaseSystem
        self.running = False
        self.lock = threading.Lock()
    
    def run(self):
        """Run all agents for this target in stages."""
        self.running = True
        target = self.memory.get_target(self.target_ip)
        
        if not target:
            logger.error(f"Target {self.target_ip} not found in memory")
            return
        
        # Run through stages
        stages = [
            AgentStage.SERVICE_DISCOVERY,
            AgentStage.CREDENTIAL_TESTING,
            AgentStage.CONFIG_ISSUES,
            AgentStage.PRIVILEGE_ESCALATION,
            AgentStage.POST_EXPLOITATION,
        ]
        
        for stage in stages:
            if not self.running:
                break
            
            logger.info(f"Running stage {stage.value} for {self.target_ip}")
            self.memory.update_target_stage(self.target_ip, stage)
            self._run_stage(stage, target)
            
            # Check for updates from other agents
            self._check_for_updates()
        
        # Enter sleeping stage
        self.memory.update_target_stage(self.target_ip, AgentStage.SLEEPING)
        logger.info(f"Target {self.target_ip} entering sleep stage")
    
    def _run_stage(self, stage: AgentStage, target: Target):
        """Run agents for a specific stage."""
        # Determine which agents should run at this stage
        agents_to_run = self._get_agents_for_stage(stage, target)
        
        # Run agents (could be parallel in future)
        for agent_type, system in agents_to_run.items():
            if agent_type in self.agent_systems:
                try:
                    self._run_agent(agent_type, system, stage, target)
                except Exception as e:
                    logger.error(f"Error running {agent_type.value} agent: {e}")
    
    def _get_agents_for_stage(self, stage: AgentStage, target: Target) -> Dict[AgentType, BaseSystem]:
        """Determine which agents should run at this stage based on discovered services."""
        agents = {}
        
        # Service discovery: all agents do initial discovery
        if stage == AgentStage.SERVICE_DISCOVERY:
            return self.agent_systems
        
        # Credential testing: agents relevant to discovered services
        if stage == AgentStage.CREDENTIAL_TESTING:
            services = target.services.values()
            if any('http' in s.lower() or 'https' in s.lower() for s in services):
                if AgentType.WEB in self.agent_systems:
                    agents[AgentType.WEB] = self.agent_systems[AgentType.WEB]
            if any('smb' in s.lower() or 'rdp' in s.lower() or 'win' in s.lower() for s in services):
                if AgentType.WINDOWS in self.agent_systems:
                    agents[AgentType.WINDOWS] = self.agent_systems[AgentType.WINDOWS]
            if AgentType.MISC in self.agent_systems:
                agents[AgentType.MISC] = self.agent_systems[AgentType.MISC]
        
        # Config issues, priv esc, post-exploit: similar logic
        if stage in [AgentStage.CONFIG_ISSUES, AgentStage.PRIVILEGE_ESCALATION, AgentStage.POST_EXPLOITATION]:
            return self.agent_systems  # All agents for now
        
        return agents
    
    def _run_agent(self, agent_type: AgentType, system: BaseSystem, 
                   stage: AgentStage, target: Target):
        """Run a specific agent for a stage."""
        goal = self._get_goal_for_stage(agent_type, stage, target)
        
        task = Task(
            target_path=self.target_ip,
            goal=goal,
            notes=f"Stage: {stage.value}, Target: {self.target_ip}"
        )
        
        result = system.run_task(task)
        logger.info(f"{agent_type.value} agent completed for {self.target_ip} at stage {stage.value}")
    
    def _get_goal_for_stage(self, agent_type: AgentType, stage: AgentStage, target: Target) -> str:
        """Generate goal description for agent at stage."""
        base_goal = f"Perform {agent_type.value} testing on {self.target_ip}"
        
        if stage == AgentStage.SERVICE_DISCOVERY:
            return f"{base_goal}. Discover services and enumerate."
        elif stage == AgentStage.CREDENTIAL_TESTING:
            return f"{base_goal}. Test for default/known credentials."
        elif stage == AgentStage.CONFIG_ISSUES:
            return f"{base_goal}. Look for configuration issues."
        elif stage == AgentStage.PRIVILEGE_ESCALATION:
            return f"{base_goal}. Look for privilege escalation opportunities."
        elif stage == AgentStage.POST_EXPLOITATION:
            return f"{base_goal}. Perform post-exploitation analysis."
        else:
            return base_goal
    
    def _check_for_updates(self):
        """Check memory for updates from other agents."""
        target = self.memory.get_target(self.target_ip)
        if target:
            # Refresh target data
            pass
    
    def stop(self):
        """Stop the manager."""
        self.running = False


class GlobalManager:
    """Manages all targets and coordinates the overall pentest."""
    
    def __init__(self, memory: SharedMemory, agent_systems: Dict[AgentType, BaseSystem]):
        self.memory = memory
        self.agent_systems = agent_systems
        self.target_managers: Dict[str, TargetManager] = {}
        self.recon_agent: Optional[ReconAgent] = None
        self.running = False
    
    def start_recon(self, subnet: str, recon_system: BaseSystem):
        """Start reconnaissance phase."""
        logger.info(f"Starting global recon for subnet: {subnet}")
        self.recon_agent = ReconAgent(recon_system, self.memory)
        targets = self.recon_agent.run_recon(subnet)
        
        # Store discovered targets
        for target in targets.values():
            self.memory.add_target(target)
        
        # Generate network summary
        summary = self.recon_agent.generate_network_summary()
        logger.info(f"Recon complete. Discovered {len(targets)} targets.")
        logger.info(f"Network summary:\n{summary}")
    
    def start_target_managers(self):
        """Start managers for all discovered targets."""
        targets = self.memory.get_all_targets()
        
        for target in targets:
            if target.ip not in self.target_managers:
                manager = TargetManager(target.ip, self.memory, self.agent_systems)
                self.target_managers[target.ip] = manager
                
                # Start manager in a thread
                thread = threading.Thread(target=manager.run, daemon=True)
                thread.start()
                logger.info(f"Started manager for target: {target.ip}")
    
    def wait(self):
        """Wait for all managers to complete (or sleep)."""
        # Wait for all target managers
        while self.running:
            time.sleep(5)
            # Check status, ping for updates, etc.
    
    def stop(self):
        """Stop all managers."""
        self.running = False
        for manager in self.target_managers.values():
            manager.stop()

