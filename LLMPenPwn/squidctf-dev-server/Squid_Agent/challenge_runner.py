#!/usr/bin/env python3

import subprocess
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

os.makedirs(os.path.join(os.getcwd(), 'logs_squidagent'), exist_ok=True)
logging.basicConfig(filename=os.path.join(os.getcwd(), 'logs_squidagent', 'challenge_runner.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from tmux_manager import TmuxChallengeManager

def run_single_challenge(challenge_id, dataset, base_path, flag_format, status_dir):
    manager = TmuxChallengeManager()
    
    # Preserve restart_count if status already exists
    existing_status = manager.read_status(challenge_id)
    restart_count = existing_status.get('restart_count', 0) if existing_status else 0
    
    status = {
        'challenge_id': challenge_id,
        'state': 'starting',
        'step': 0,
        'start_time': time.time(),
        'result': 'pending',
        'tmux_session': f"squid_{challenge_id}",
        'restart_count': restart_count
    }
    manager.update_status(challenge_id, status)
    
    status['state'] = 'running'
    manager.update_status(challenge_id, status)
    
    script_path = Path(__file__).parent / "run_squidagent.py"
    python_path = sys.executable
    
    cmd = [
        python_path,
        str(script_path),
        "--dataset", dataset,
        "--challenge", challenge_id,
        "--base-path", base_path if base_path else "."
    ]
    
    if flag_format:
        cmd.extend(["--flag-format", flag_format])
    
    log_file = Path(status_dir) / f"{challenge_id}.log"
    
    with open(log_file, 'w') as log:
        log.write(f"Starting challenge: {challenge_id}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write(f"Started at: {datetime.now()}\n")
        log.write("-" * 80 + "\n\n")
        log.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        stopped = False
        while process.poll() is None:
            # Check if we should stop
            current_status = manager.read_status(challenge_id)
            if current_status and current_status.get('state') == 'stopping':
                logger.info(f"Stop signal received for {challenge_id}, terminating process")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                stopped = True
                break
            
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        step_count = content.count('Step ')
                        status['step'] = step_count
                        manager.update_status(challenge_id, status)
                except:
                    pass
            time.sleep(2)
        
        exit_code = process.returncode if not stopped else -1
    
    if stopped:
        status['state'] = 'stopped'
        status['end_time'] = time.time()
        status['exit_code'] = -1
        status['result'] = 'stopped'
    else:
        status['state'] = 'completed'
        status['end_time'] = time.time()
        status['exit_code'] = exit_code
        
        if exit_code == 0:
            status['result'] = 'success'
        else:
            status['result'] = 'failed'
    
    cost = 0.0
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            if '🦑🦑🦑🦑' in content or 'Correct flag submitted:' in content:
                status['result'] = 'success'
            elif 'Error:' in content or 'Failed' in content:
                status['result'] = 'failed'
            
            import re
            cost_pattern = r'\[COST\] Challenge \'[^\']+\': spent \$([\d.]+)'
            cost_matches = re.findall(cost_pattern, content)
            if cost_matches:
                cost = float(cost_matches[-1])
            else:
                cost_pattern2 = r'Challenge \'[^\']+\' total cost: \$([\d.]+)'
                cost_matches2 = re.findall(cost_pattern2, content)
                if cost_matches2:
                    cost = float(cost_matches2[-1])
    except Exception as e:
        logger.debug(f"Error extracting cost from log: {e}")
    
    status['cost'] = cost
    status['duration'] = status.get('end_time', time.time()) - status.get('start_time', time.time())
    
    manager.update_status(challenge_id, status)
    
    return exit_code

if __name__ == "__main__":
    if len(sys.argv) < 5:
        logger.error("Usage: challenge_runner.py <challenge_id> <dataset> <base_path> <flag_format> <status_dir>")
        sys.exit(1)
    
    challenge_id = sys.argv[1]
    dataset = sys.argv[2]
    base_path = sys.argv[3]
    flag_format = sys.argv[4] if sys.argv[4] != "None" else None
    status_dir = sys.argv[5]
    
    exit_code = run_single_challenge(challenge_id, dataset, base_path, flag_format, status_dir)
    sys.exit(exit_code)

