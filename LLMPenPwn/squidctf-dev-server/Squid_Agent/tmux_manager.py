#!/usr/bin/env python3

import subprocess
import json
import logging
import os
import time
import sys
from pathlib import Path
from datetime import datetime

os.makedirs(os.path.join(os.getcwd(), 'logs_squidagent'), exist_ok=True)
logging.basicConfig(filename=os.path.join(os.getcwd(), 'logs_squidagent', 'tmux_manager.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TmuxChallengeManager:
    def __init__(self, session_prefix="squid"):
        self.session_prefix = session_prefix
        self.status_dir = Path("./tmux_status")
        self.status_dir.mkdir(exist_ok=True)
        self.master_session = f"{session_prefix}_master"
        
    def create_session(self, session_name, command):
        subprocess.run([
            "tmux", "new-session", "-d", "-s", session_name, command
        ], check=False)
        
    def session_exists(self, session_name):
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True
        )
        return result.returncode == 0
    
    def kill_session(self, session_name):
        if self.session_exists(session_name):
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=False)
    
    def list_windows(self, session_name):
        if not self.session_exists(session_name):
            return []
        result = subprocess.run(
            ["tmux", "list-windows", "-t", session_name, "-F", "#{window_index}:#{window_name}"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    
    def get_status_file(self, challenge_id):
        return self.status_dir / f"{challenge_id}.json"
    
    def update_status(self, challenge_id, status_data):
        status_file = self.get_status_file(challenge_id)
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    
    def read_status(self, challenge_id):
        status_file = self.get_status_file(challenge_id)
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def get_all_statuses(self):
        statuses = []
        for status_file in self.status_dir.glob("*.json"):
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
                    statuses.append(data)
            except:
                pass
        return statuses
    
    def render_master_display(self):
        statuses = self.get_all_statuses()
        statuses.sort(key=lambda x: x.get('start_time', 0))
        
        lines = []
        lines.append("=" * 100)
        lines.append("SQUID CTF CHALLENGE TRACKER")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"Total Challenges: {len(statuses)}")
        lines.append(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("-" * 100)
        
        header = f"{'Challenge ID':<30} {'Status':<15} {'Step':<8} {'Cost ($)':<12} {'Result':<10} {'Restarts':<10}"
        lines.append(header)
        lines.append("-" * 100)
        
        for status in statuses:
            challenge_id = status.get('challenge_id', 'unknown')[:29]
            state = status.get('state', 'unknown')[:14]
            step = str(status.get('step', 0))[:7]
            cost = status.get('cost', 0.0)
            cost_str = f"${cost:.6f}" if cost > 0 else "N/A"
            cost_str = cost_str[:11]
            result = status.get('result', 'pending')[:9]
            restarts = str(status.get('restart_count', 0))[:9]
            
            line = f"{challenge_id:<30} {state:<15} {step:<8} {cost_str:<12} {result:<10} {restarts:<10}"
            lines.append(line)
        
        lines.append("-" * 100)
        lines.append("")
        
        completed = sum(1 for s in statuses if s.get('state') == 'completed')
        failed = sum(1 for s in statuses if s.get('result') == 'failed')
        running = sum(1 for s in statuses if s.get('state') == 'running')
        total_restarts = sum(s.get('restart_count', 0) for s in statuses)
        total_cost = sum(s.get('cost', 0.0) for s in statuses)
        
        lines.append(f"Running: {running} | Completed: {completed} | Failed: {failed} | Total Restarts: {total_restarts} | Total Cost: ${total_cost:.6f}")
        lines.append("")
        lines.append("Press Ctrl+C to exit | Refresh: 2s")
        lines.append("=" * 100)
        
        return '\n'.join(lines)
    
    def start_master_display(self):
        if self.session_exists(self.master_session):
            self.kill_session(self.master_session)
        
        display_script = self.status_dir / "display.sh"
        python_path = sys.executable
        script_path = Path(__file__).absolute()
        
        with open(display_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("while true; do\n")
            f.write("  clear\n")
            f.write(f"  {python_path} {script_path} --display-only\n")
            f.write("  sleep 2\n")
            f.write("done\n")
        
        os.chmod(display_script, 0o755)
        
        self.create_session(self.master_session, f"bash {display_script}")
        logger.info(f"Master display started in tmux session: {self.master_session}")
        logger.info(f"Attach with: tmux attach -t {self.master_session}")

def run_display_only():
    manager = TmuxChallengeManager()
    print(manager.render_master_display())

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--display-only":
        run_display_only()
    else:
        manager = TmuxChallengeManager()
        manager.start_master_display()

