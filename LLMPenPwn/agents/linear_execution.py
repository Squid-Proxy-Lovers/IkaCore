#!/usr/bin/env python3
"""
Linear execution system for penetration testing agents.
Starts agents in sequence, beginning with the recon agent.
"""

import argparse
import json
import logging
import sqlite3
import threading
import time
import sys
from pathlib import Path
from datetime import datetime

# Add Para-Core src to path for IkaCore
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import yaml  # type: ignore
except ImportError:
    try:
        import ruamel.yaml as yaml  # type: ignore
    except ImportError:
        print("Error: yaml or ruamel.yaml required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

from base import BaseSystem
from env import HostEnvironment
from tools import (
    get_recon_toolset,
    get_manager_toolset,
    get_targeted_toolset,
    get_web_toolset,
    get_windows_toolset,
    get_misc_toolset,
    get_code_review_toolset,
    poll_notifications,
)
from logging_config import setup_logging, AgentLoggerAdapter

BASE_MODEL_ID = "claude-sonnet-4-5"  # Default, can be overridden with --model
NETWORK_JSON_FILE = "network_scan.json"
_prompts_dir = Path(__file__).parent / "prompts"
PENTESTING_MANAGER_PROMPT = str(_prompts_dir / "pentesting_manager.yaml")
TARGETED_AGENT_PROMPT = str(_prompts_dir / "targeted_agent.yaml")
WEB_AGENT_PROMPT = str(_prompts_dir / "web_agent.yaml")
WINDOWS_AGENT_PROMPT = str(_prompts_dir / "windows_agent.yaml")
MISC_AGENT_PROMPT = str(_prompts_dir / "misc_agent.yaml")
CODE_REVIEW_AGENT_PROMPT = str(_prompts_dir / "code_review_agent.yaml")

setup_logging(
    log_dir="logs",
    log_level="INFO",
    enable_file_logging=True,
    enable_json_logging=True,
)
logger = logging.getLogger(__name__)

DB_PATH_DEFAULT = "pentest_memory.db"


def load_prompt(prompt_file: str) -> str:
    """Load system prompt from YAML file."""
    try:
        with open(prompt_file, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('system_prompt', '')
    except Exception as e:
        logger.error(f"Error loading prompt from {prompt_file}: {e}")
        return ""


def create_recon_agent(model_id: str, api_key: str, env, prompt_file: str) -> BaseSystem:
    """Create the recon agent system."""
    tools = get_recon_toolset()
    system_prompt = load_prompt(prompt_file)
    
    return BaseSystem.from_model_and_tools(
        model_id=model_id,
        api_key=api_key,
        tools=tools,
        env=env,
        max_steps=200,
        system_prompt=system_prompt,
        enable_summarization=True,
        name="recon_agent",
        description="Network reconnaissance agent",
        role="recon",
    )


def run_recon_agent(system: BaseSystem, subnet: str, json_file: str) -> None:
    """Run the recon agent to analyze the subnet."""
    agent_logger = AgentLoggerAdapter(logger, "RECON")
    agent_logger.info(f"Starting recon agent for subnet: {subnet}")
    
    task_prompt = f"""Perform comprehensive network reconnaissance on subnet {subnet}.

Your objectives:
1. Discover all live hosts in the subnet
2. Identify open ports and services on each host
3. Determine operating systems where possible
4. Analyze each service in detail
5. Document everything in the JSON file: {json_file}

Use the add_target and add_port tools to build the network JSON file.
Start with a quick nmap scan to discover hosts, then perform detailed scans on each host.

At the end, provide a text-based network diagram summarizing the entire network topology.
"""
    
    agent_logger.info("Recon agent starting execution...")
    result = system.run(task_prompt)
    
    agent_logger.info("Recon agent completed")
    output = result.get("output", "")
    success = result.get("success", True)
    
    if output:
        agent_logger.info(f"Final output length: {len(output)} characters")
        agent_logger.info(f"Final output preview: {output[:500]}..." if len(output) > 500 else f"Final output: {output}")
        print(f"\n{'='*60}")
        print("RECON AGENT OUTPUT:")
        print(f"{'='*60}")
        print(output[:2000] if len(output) > 2000 else output)
        if len(output) > 2000:
            print(f"\n... (truncated, full output in logs) ...")
        print(f"{'='*60}\n")
    else:
        agent_logger.warning(f"Recon agent completed with no output. Success: {success}")
        print(f"\n{'='*60}")
        print("WARNING: Recon agent returned no output")
        print(f"Success: {success}")
        print(f"{'='*60}\n")
    
    # Also check for summary
    summary = result.get("summary", "")
    if summary:
        print(f"\nSUMMARY: {summary[:500]}\n")
    
    return result


def create_pentesting_manager_agent(model_id: str, api_key: str, env, prompt_file: str) -> BaseSystem:
    tools = get_manager_toolset()
    system_prompt = load_prompt(prompt_file)
    return BaseSystem.from_model_and_tools(
        model_id=model_id,
        api_key=api_key,
        tools=tools,
        env=env,
        max_steps=999999,
        system_prompt=system_prompt,
        enable_summarization=True,
        name="pentesting_manager",
        description="Pentesting manager agent",
        role="manager",
    )


def create_subagent_systems(model_id: str, api_key: str, env) -> dict:
    """Create BaseSystem instances for specialized agents."""
    prompt_map = {
        "WEB": WEB_AGENT_PROMPT,
        "WINDOWS": WINDOWS_AGENT_PROMPT,
        "MISC": MISC_AGENT_PROMPT,
        "CODE_REVIEW": CODE_REVIEW_AGENT_PROMPT,
    }
    toolset_map = {
        "WEB": get_web_toolset,
        "WINDOWS": get_windows_toolset,
        "MISC": get_misc_toolset,
        "CODE_REVIEW": get_code_review_toolset,
    }
    systems = {}
    for name, prompt_path in prompt_map.items():
        system_prompt = load_prompt(prompt_path)
        systems[name] = BaseSystem.from_model_and_tools(
            model_id=model_id,
            api_key=api_key,
            tools=toolset_map[name](),
            env=env,
            max_steps=999999,
            system_prompt=system_prompt,
            enable_summarization=True,
            name=f"{name.lower()}_agent",
            description=f"{name.lower()} specialized agent",
            role=name.lower(),
        )
    return systems


def run_pentesting_manager(system: BaseSystem, json_file: str):
    task_prompt = f"""You are the pentesting manager overseeing the entire network.

Network JSON path: {json_file}

Goals:
1) Review all discovered targets and ports
2) Propose breakthroughs and high-value avenues across the network
3) Update target summaries and ports when you find improvements
4) Produce a concise network-level report of progress

Use the JSON editing tools (add_target, add_port, update_target_summary) to record findings."""
    agent_logger = AgentLoggerAdapter(logger, "MANAGER")
    agent_logger.info("Pentesting manager starting...")
    result = system.run(task_prompt)
    agent_logger.info("Pentesting manager completed")
    return result


def run_pentesting_manager_loop(system: BaseSystem, json_file: str):
    """Run pentesting manager in a loop until cancelled."""
    agent_logger = AgentLoggerAdapter(logger, "MANAGER")
    agent_logger.info("Pentesting manager starting (infinite loop mode)...")
    
    initial_task = f"""You are the pentesting manager overseeing the entire network.

Network JSON path: {json_file}

You run continuously, coordinating the penetration test. You do NOT stop unless explicitly cancelled.
After completing each analysis cycle, continue monitoring and coordinating.

Goals:
1) Review all discovered targets and ports
2) Propose breakthroughs and high-value avenues across the network
3) Delegate work to specialized sub-agents
4) Monitor for new findings and coordinate responses
5) Update target summaries and ports when you find improvements
6) Produce periodic network-level reports

Use the JSON editing tools (add_target, add_port, update_target_summary) to record findings.
Use delegate_to_subagent to assign work to specialized agents."""
    
    try:
        while True:
            agent_logger.info("Manager starting new coordination cycle...")
            result = system.run(initial_task)
            agent_logger.info(f"Manager cycle completed. Output: {result.get('output', '')[:200] if result.get('output') else 'None'}...")
            
            if not result.get('success', True):
                agent_logger.warning(f"Manager cycle had error: {result.get('error', 'Unknown error')}")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        agent_logger.info("Manager stopped by user")
    except Exception as e:
        agent_logger.error(f"Manager error: {e}")


def create_targeted_agent(model_id: str, api_key: str, env, prompt_file: str) -> BaseSystem:
    tools = get_targeted_toolset()
    system_prompt = load_prompt(prompt_file)
    return BaseSystem.from_model_and_tools(
        model_id=model_id,
        api_key=api_key,
        tools=tools,
        env=env,
        max_steps=999999,
        system_prompt=system_prompt,
        enable_summarization=True,
        name="targeted_agent",
        description="Targeted penetration testing agent",
        role="targeted",
    )


def run_targeted_agent(system: BaseSystem, target_id: str, target_data: dict, json_file: str, db_path: str = "pentest_memory.db"):
    ip = target_data.get("ip", "")
    ports = target_data.get("ports", {})
    summary = target_data.get("summary", "")
    
    agent_logger = AgentLoggerAdapter(logger, f"TARGETED-{target_id}", target_ip=ip)
    agent_logger.info(f"Targeted agent starting for {target_id} ({ip})")
    
    # Use IP for notifications if available, otherwise use target_id
    notification_target = ip if ip else target_id
    
    initial_task = f"""You are a targeted agent for {target_id} ({ip}).

Current data:
- Summary: {summary}
- Ports: {ports}

Follow the pentest stages in order:
1) service_discovery
2) credential_testing
3) config_issues
4) privilege_escalation
5) post_exploitation

IMPORTANT: When you have exhausted all available options and cannot proceed further, 
call enter_wait_mode with a clear reason. You will be notified when new data becomes available.

Before starting each stage, check for notifications using poll_notifications.
Use add_port and update_target_summary to record progress in {json_file}.
Document findings clearly with write_vulnerability and store_interesting_data."""
    
    try:
        while True:
            # Poll for notifications directly (tool is available to agent, but we check here too)
            # Use IP for notifications if available, otherwise use target_id
            try:
                notifications_json = poll_notifications(notification_target, limit=5, db_path=db_path)
                notifications = json.loads(notifications_json) if notifications_json else []
            except (json.JSONDecodeError, Exception) as e:
                agent_logger.debug(f"Error polling notifications: {e}")
                notifications = []
            
            if notifications and len(notifications) > 0:
                agent_logger.info(f"New notifications received for {target_id}: {len(notifications)} notifications")
                notifications_text = json.dumps(notifications, indent=2)
                task_prompt = f"""{initial_task}

NEW NOTIFICATIONS:
{notifications_text}

Review these notifications and see if you can use the new information to make progress.
If you can proceed, continue with your current stage. If not, enter wait mode again."""
            else:
                task_prompt = initial_task
            
            result = system.run(task_prompt)
            
            output = result.get('output', '')
            if result.get('success', True) and ("wait mode" in output.lower() or "entered wait mode" in output.lower()):
                agent_logger.info(f"Targeted agent {target_id} entered wait mode")
                time.sleep(10)
                continue
            elif not result.get('success', True):
                agent_logger.warning(f"Targeted agent {target_id} encountered error: {result.get('error', 'Unknown error')}")
                time.sleep(5)
                continue
            else:
                agent_logger.info(f"Targeted agent {target_id} completed cycle")
                time.sleep(5)
                
    except KeyboardInterrupt:
        agent_logger.info(f"Targeted agent {target_id} stopped by user")
    except Exception as e:
        agent_logger.error(f"Targeted agent {target_id} error: {e}")


def _ensure_agent_tasks_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_type TEXT,
            target_id TEXT,
            task_json TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()


def _fetch_next_task(conn: sqlite3.Connection, agent_type: str) -> dict | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT task_id, agent_type, target_id, task_json FROM agent_tasks WHERE agent_type = ? AND status = 'PENDING' ORDER BY task_id ASC LIMIT 1",
        (agent_type,),
    )
    row = cur.fetchone()
    if not row:
        return None
    task = {"task_id": row[0], "agent_type": row[1], "target_id": row[2], "task_json": row[3]}
    cur.execute(
        "UPDATE agent_tasks SET status = 'IN_PROGRESS', updated_at = ? WHERE task_id = ?",
        (datetime.utcnow().isoformat(), int(task["task_id"])),
    )
    conn.commit()
    return task


def _complete_task(conn: sqlite3.Connection, task_id: int, status: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE agent_tasks SET status = ?, updated_at = ? WHERE task_id = ?",
        (status, datetime.utcnow().isoformat(), int(task_id)),
    )
    conn.commit()


def run_subagent_worker(agent_type: str, system: BaseSystem, json_file: str, db_path: str) -> None:
    logger.info(f"Subagent worker started: {agent_type}")
    while True:
        try:
            conn = sqlite3.connect(db_path)
            _ensure_agent_tasks_table(conn)
            task = _fetch_next_task(conn, agent_type)
            if not task:
                conn.close()
                time.sleep(2)
                continue
            task_id = int(task["task_id"])
            target_id = str(task["target_id"])
            task_json = str(task["task_json"])
            conn.close()

            prompt = (
                f"You are a specialized agent ({agent_type}).\n\n"
                f"Network JSON path: {json_file}\n"
                f"Target: {target_id}\n\n"
                f"Task JSON:\n{task_json}\n\n"
                "Execute the task safely. Record any findings using write_vulnerability/store_interesting_data and update JSON summaries as needed."
            )
            result = system.run(prompt)

            conn2 = sqlite3.connect(db_path)
            _ensure_agent_tasks_table(conn2)
            _complete_task(conn2, task_id, "COMPLETED")
            conn2.close()
            logger.info(f"Subagent task completed: {agent_type} task_id={task_id}")
            _ = result
        except Exception as e:
            logger.error(f"Subagent worker error ({agent_type}): {e}")
            try:
                conn3 = sqlite3.connect(db_path)
                _ensure_agent_tasks_table(conn3)
                if "task_id" in locals():
                    _complete_task(conn3, int(task_id), "FAILED")
                conn3.close()
            except Exception:
                pass
            time.sleep(2)


def get_api_key_for_model(model_id: str, api_keys: dict) -> str:
    """Get the appropriate API key for the given model ID."""
    model_id_lower = model_id.lower()
    
    if "claude" in model_id_lower or "anthropic" in model_id_lower:
        key = api_keys.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not found in keys file")
        return key
    elif "gpt" in model_id_lower or "openai" in model_id_lower:
        key = api_keys.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in keys file")
        return key
    elif "deepseek" in model_id_lower:
        key = api_keys.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY not found in keys file")
        return key
    else:
        # Default to Anthropic if unknown
        key = api_keys.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not found in keys file (default)")
        return key


def main():
    parser = argparse.ArgumentParser(description="Linear Execution System for Penetration Testing")
    parser.add_argument("--subnet", required=True, help="Subnet to scan (e.g., 192.168.1.0/24)")
    parser.add_argument("--keys", default="keys.cfg", help="Path to API keys config file")
    parser.add_argument("--model", default=BASE_MODEL_ID, help=f"Model ID to use (default: {BASE_MODEL_ID})")
    parser.add_argument("--json-output", default=NETWORK_JSON_FILE, help="Path to network JSON output file")
    parser.add_argument("--prompt-file", default=str(Path(__file__).parent / "prompts" / "recon_agent.yaml"), help="Path to recon agent prompt file")
    parser.add_argument("--manager-prompt", default=PENTESTING_MANAGER_PROMPT, help="Path to pentesting manager prompt file")
    parser.add_argument("--targeted-prompt", default=TARGETED_AGENT_PROMPT, help="Path to targeted agent prompt file")
    parser.add_argument("--db", default=DB_PATH_DEFAULT, help="Path to SQLite database (shared)")
    
    args = parser.parse_args()
    
    # Read API keys
    api_keys = {}
    try:
        with open(args.keys, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    api_keys[key.strip()] = value.strip()
    except Exception as e:
        logger.error(f"Error reading API keys from {args.keys}: {e}")
        return 1
    
    # Get API key for the selected model
    try:
        api_key = get_api_key_for_model(args.model, api_keys)
        logger.info(f"Using model: {args.model}")
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    model_id = args.model
    
    # Create environment
    logger.info("Creating host execution environment...")
    env = HostEnvironment()
    
    # Create recon agent
    logger.info("Creating recon agent...")
    recon_system = create_recon_agent(model_id, api_key, env, args.prompt_file)
    
    # Run recon agent
    logger.info("=" * 60)
    logger.info("PHASE 1: RECONNAISSANCE")
    logger.info("=" * 60)
    run_recon_agent(recon_system, args.subnet, args.json_output)
    
    # Check if JSON file was created
    json_path = Path(args.json_output)
    if json_path.exists():
        logger.info(f"Network JSON file created: {args.json_output}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            logger.info(f"Discovered {len(data)} targets")
    else:
        logger.warning(f"Network JSON file not found at {args.json_output}")
        data = {}

    # Phase 2: Pentesting manager (network-wide) - runs indefinitely
    logger.info("=" * 60)
    logger.info("PHASE 2: PENTESTING MANAGER (Running indefinitely until cancelled)")
    logger.info("=" * 60)
    manager_system = create_pentesting_manager_agent(model_id, api_key, env, args.manager_prompt)
    manager_thread = threading.Thread(
        target=run_pentesting_manager_loop,
        args=(manager_system, args.json_output),
        daemon=True,
    )
    manager_thread.start()

    # Phase 2b: Specialized subagent workers (drain delegated tasks)
    subagents = create_subagent_systems(model_id, api_key, env)
    subagent_threads = []
    for agent_type, sys_obj in subagents.items():
        t = threading.Thread(
            target=run_subagent_worker,
            args=(agent_type, sys_obj, args.json_output, args.db),
            daemon=True,
        )
        t.start()
        subagent_threads.append(t)

    # Phase 3: Targeted agents (one per target)
    logger.info("=" * 60)
    logger.info("PHASE 3: TARGETED AGENTS")
    logger.info("=" * 60)
    target_threads = []
    for target_id, target_data in data.items():
        targeted_system_instance = create_targeted_agent(model_id, api_key, env, args.targeted_prompt)
        t = threading.Thread(
            target=run_targeted_agent,
            args=(targeted_system_instance, target_id, target_data, args.json_output, args.db),
            daemon=True,
        )
        t.start()
        target_threads.append(t)

    logger.info("=" * 60)
    logger.info("All agents started. Manager runs indefinitely.")
    logger.info("Press Ctrl+C to stop all agents.")
    logger.info("=" * 60)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down all agents...")
        return 0


if __name__ == "__main__":
    sys.exit(main())
