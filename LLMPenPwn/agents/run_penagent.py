#!/usr/bin/env python3
"""
Multi-agent penetration testing system entry point.
"""

import argparse
import logging
import sys
import importlib.util
from pathlib import Path

# Import local memory module before adding IkaCore to path
_local_dir = Path(__file__).parent
_memory_spec = importlib.util.spec_from_file_location("local_memory", _local_dir / "memory.py")
_local_memory = importlib.util.module_from_spec(_memory_spec)
_memory_spec.loader.exec_module(_local_memory)
SharedMemory = _local_memory.SharedMemory

# Add Para-Core src to path for IkaCore
_para_core_src = Path(__file__).parent.parent.parent / "src"
if str(_para_core_src) not in sys.path:
    sys.path.insert(0, str(_para_core_src))

from base import BaseSystem, AgentType
from managers import GlobalManager
from env import HostEnvironment
from tools import get_recon_toolset, get_manager_toolset, get_targeted_toolset, get_web_toolset, get_windows_toolset, get_misc_toolset, get_code_review_toolset

BASE_MODEL_ID = "claude-3-5-sonnet-20241022"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_api_keys(key_file: str) -> dict:
    """Read API keys from config file."""
    api_keys = {}
    try:
        with open(key_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    api_keys[key.strip()] = value.strip()
    except Exception as e:
        logger.error(f"Error reading API keys from {key_file}: {e}")
    return api_keys


def create_agent_systems(model_id: str, api_key: str, env, tools_map: dict = None):
    """Create BaseSystem instances for each agent type."""
    if tools_map is None:
        tools_map = {
            AgentType.WEB: get_web_toolset(),
            AgentType.WINDOWS: get_windows_toolset(),
            AgentType.MISC: get_misc_toolset(),
            AgentType.CODE_REVIEW: get_code_review_toolset(),
        }
    
    agent_systems = {}
    for agent_type, tools in tools_map.items():
        agent_systems[agent_type] = BaseSystem.from_model_and_tools(
            model_id=model_id,
            api_key=api_key,
            tools=tools,
            env=env,
            max_steps=100,
            name=f"{agent_type.value}_agent",
            description=f"{agent_type.value} penetration testing agent",
            role=agent_type.value,
        )
    
    return agent_systems


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Penetration Testing System")
    parser.add_argument("--subnet", required=True, help="Subnet to scan (e.g., 192.168.1.0/24)")
    parser.add_argument("--keys", default="keys.cfg", help="Path to API keys config file")
    parser.add_argument("--db", default="pentest_memory.db", help="Path to SQLite database")
    
    args = parser.parse_args()
    
    # Load API keys
    api_keys = read_api_keys(args.keys)
    api_key = api_keys.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in keys file")
        return 1
    
    # Initialize shared memory
    logger.info(f"Initializing shared memory database: {args.db}")
    memory = SharedMemory(db_path=args.db)
    
    # Create host environment (no containers, runs directly on host)
    logger.info("Creating host execution environment...")
    env = HostEnvironment()
    
    # Create agent systems
    logger.info("Creating agent systems...")
    agent_systems = create_agent_systems(BASE_MODEL_ID, api_key, env)
    
    # Create recon system
    recon_tools = get_recon_toolset()
    recon_system = BaseSystem.from_model_and_tools(
        model_id=BASE_MODEL_ID,
        api_key=api_key,
        tools=recon_tools,
        env=env,
        max_steps=100,
        name="recon_agent",
        description="Network reconnaissance agent",
        role="recon",
    )
    
    # Start the multi-agent system
    logger.info("Starting multi-agent penetration testing system...")
    manager = GlobalManager(memory, agent_systems)
    manager.start_recon(args.subnet, recon_system)
    manager.start_target_managers()
    
    logger.info("System running. Waiting for completion...")
    manager.wait()
    
    logger.info("Penetration testing complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
