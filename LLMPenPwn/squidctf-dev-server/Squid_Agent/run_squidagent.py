#!/usr/bin/env python3
import argparse
import functools
import json
import os
import sys
import time
import shutil
from pathlib import Path
import logging 
from odin.model import OpenAIResponsesModel, PortKeyModel, DeepSeekAPIModel
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

# Import the updated logger
from squid_logging import run_log
import litellm

BASE_MODEL_ID = "gpt-5"
#BASE_MODEL_ID = "deepseek-reasoner"
sys.path.append(str(Path(__file__).parent / ".." / "squidagent"))
litellm._turn_on_debug()
from ctf_environment import *
from squidagent.tools.ctf_tools import set_ctf_environment
from base_ctf_system import BaseCTFSystem
from rev_system import RevSystem
from pwn_system import PwnSystem
from crypto_system import CryptoSystem
from web_system import WebSystem
from forensics_system import ForensicsSystem
from misc_system import MiscSystem

os.makedirs(os.path.join(os.getcwd(), 'logs_squidagent'), exist_ok=True)
logging.basicConfig(filename=os.path.join(os.getcwd(), 'logs_squidagent', 'run_squidagent.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SquidAgentSolver:
    def __init__(self, api_key: str = None, anthropic_api_key: str = None, openrouter_api_key: str = None, deepseek_api_key: str = None, model_id: str = BASE_MODEL_ID):
        if model_id.startswith("deepseek"):
            self.model = DeepSeekAPIModel(model_id=model_id, api_key=deepseek_api_key or api_key)
        else:
            self.model = OpenAIResponsesModel(model_id=model_id, api_key=api_key)
        self.api_key = api_key
        self.anthropic_api_key = anthropic_api_key
        self.openrouter_api_key = openrouter_api_key
        self.deepseek_api_key = deepseek_api_key
        print("API Key: ", self.api_key)
        # Create specialized subsystem classes
        self.systems = {
            'rev': RevSystem(self.model, self.api_key, self.anthropic_api_key),
            'pwn': PwnSystem(self.model, self.api_key, self.anthropic_api_key),
            'crypto': CryptoSystem(self.model, self.api_key, self.anthropic_api_key),
            'web': WebSystem(self.model, self.api_key, self.anthropic_api_key),
            'forensics': ForensicsSystem(self.model, self.api_key, self.anthropic_api_key),
            'misc': MiscSystem(self.model, self.api_key, self.anthropic_api_key),
        }

        # challenge state
        self.environment = None
        self.current_challenge = None
        self.dataset = None
        self.dataset_base_path = None

    def cleanup_all_non_ida_containers(self):
        """Clean up all containers that don't have 'idadocker' in their image name."""
        try:
            logger.info("Cleaning up all non-IDA containers at startup...")
            
            result = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.ID}} {{.Image}} {{.Names}}"], 
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            container_id = parts[0]
                            image = parts[1]
                            container_name = parts[2] if len(parts) >= 3 else "unknown"
                            
                            if "idadocker" not in image.lower():
                                logger.info(f"Removing non-IDA container: {container_id[:12]} ({container_name}, image: {image})")
                                subprocess.run(["docker", "stop", container_id], capture_output=True, text=True)
                                subprocess.run(["docker", "rm", container_id], capture_output=True, text=True)
            
            logger.info("Non-IDA containers cleanup complete")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup non-IDA containers: {e}")

    def load_dataset(self, dataset_file: str) -> bool:
        """
        Use challenges.json file to load the dataset
        Args: dataset_file (str): Path to challenges.json file
        Returns: bool: True if dataset loaded successfully
        """
        try:
            dataset_path = Path(dataset_file)
            self.dataset_base_path = dataset_path.parent

            with open(dataset_file, 'r') as f:
                self.dataset = json.load(f)
            logger.info(f"Loaded dataset with {len(self.dataset)} challenges")
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False

    def get_challenge_list(self, category: str = None) -> list:
        if not self.dataset:
            return []
        challenge_ids = list(self.dataset.keys())
        
        if category:
            target_category = self.normalize_category(category)
            
            # Filter challenges by category
            filtered = []
            for challenge_id in challenge_ids:
                challenge_category = self.dataset[challenge_id].get('category', '')
                normalized_challenge_category = self.normalize_category(challenge_category)
                if normalized_challenge_category == target_category:
                    filtered.append(challenge_id)
            return filtered
        
        return challenge_ids

    def get_prompt(self, prompt_name):
        prompt_path = Path(__file__).parent.parent / "squidagent" / "prompts" / prompt_name
        with open(prompt_path, 'r') as f:
            template_content = f.read()
        return template_content

    @staticmethod
    def reset_rag_database():
        """Reset the RAG database by deleting all collections and data."""
        try:
            rag_db_dir = Path("./rag_db")
            if rag_db_dir.exists():
                logger.info(f"Resetting RAG database at {rag_db_dir}")
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=str(rag_db_dir))
                    collections = client.list_collections()
                    for collection in collections:
                        logger.info(f"Deleting RAG collection: {collection.name}")
                        client.delete_collection(collection.name)
                    logger.info("RAG database reset complete")
                except ImportError:
                    logger.info("chromadb not installed, deleting RAG DB directory instead")
                    shutil.rmtree(rag_db_dir)
                    logger.info(f"Deleted RAG DB directory: {rag_db_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean RAG DB via chromadb: {e}")
                    logger.info("Attempting to delete RAG DB directory instead")
                    shutil.rmtree(rag_db_dir)
                    logger.info(f"Deleted RAG DB directory: {rag_db_dir}")
            else:
                logger.info("RAG DB directory does not exist, nothing to reset")
        except Exception as e:
            logger.error(f"Error resetting RAG database: {e}")


    def create_challenge_from_dataset(self, challenge_id: str, base_path: str = ".",flag_format: str = None) -> CTFChallenge:
        """
        Args:
            challenge_id (str): Challenge ID from the dataset
            base_path (str): Base path for resolving challenge paths
                - if none uses dataset base path
            flag_format (str): Flag format for the challenge
        Returns: CTFChallenge: Challenge object ready for solving
        """

        if not self.dataset or challenge_id not in self.dataset:
            raise ValueError(f"Challenge '{challenge_id}' not found in dataset")
        

        # default to using "." if base_path is not provided and not set by default
        if self.dataset_base_path is not None:
            base_path = self.dataset_base_path


        dataset_entry = self.dataset[challenge_id]
        challenge_path = Path(base_path) / dataset_entry["path"]
        challenge_json_path = challenge_path / "challenge.json"

        if not challenge_json_path.exists():
            raise FileNotFoundError(f"challenge.json not found at {challenge_json_path}")

        with open(challenge_json_path, 'r') as f:
            challenge_config = json.load(f)

        # get list of files in the challenge directory if not provided in config
        files = challenge_config.get("files")
        if not isinstance(files, list):
            try:
                files = [f.name for f in challenge_path.iterdir() if f.is_file() and f.name != "challenge.json"]
            except Exception:
                files = []

        # get service host and port
        service_host = challenge_config.get('box', 'localhost')
        service_port = challenge_config.get('internal_port') or challenge_config.get('port')
        description = challenge_config.get("description", "")
        if service_host != None and service_port != None:
            if "{box}" in description:
                description = description.replace("{box}", str(service_host))
                description = description.replace("{port}", str(service_port))
            else:
                description = description + f" The service is hosted on {service_host}:{service_port}."
        if flag_format is None:
            description = description + ' Unless specified already, the flag format for this challenge is flag{}'
        else:
            description = description + f" The flag format for this challenge is {flag_format}"
    
        # create CTFChallenge object
        challenge = CTFChallenge(
            name=challenge_config.get("name", challenge_id),
            category=challenge_config.get("category", dataset_entry.get("category", "misc")),
            description=description,
            files=files,
            challenge_dir=challenge_path,
            flag=challenge_config.get("flag")
        )

        return challenge

    @staticmethod
    def normalize_category(category: str) -> str:
        """Normalize category name to handle synonyms and variations."""
        if not category:
            return ""
        normalized = category.strip().lower()
        mapping = {
            'reverse_engineering': 'rev',
            'reverse': 'rev',
            'rev': 'rev',
            'pwn': 'pwn',
            'crypto': 'crypto',
            'cryptography': 'crypto',
            'web': 'web',
            'webex': 'web',
            'forensics': 'forensics',
            'misc': 'misc'
        }
        return mapping.get(normalized, normalized)

    def get_system_for_category(self, category: str) -> BaseCTFSystem:
        # Accept common synonyms and normalized keys
        key = self.normalize_category(category)
        return self.systems.get(key)

    def solve_challenge_from_dataset(self, challenge_id: str, base_path: str = ".",flag_format: str = 'flag{}') -> dict:
        """
        Args:
            challenge_id (str): Challenge ID from the dataset
            base_path (str): Base path for resolving challenge paths
                - if none uses dataset base path
            flag_format (str): Flag format for the challenge
        Returns:
            dict: Results of the challenge solving attempt
        """

        if not self.dataset:
            return {"error": "No dataset loaded. Use load_dataset() first."}

        try:
            challenge = self.create_challenge_from_dataset(challenge_id, base_path,flag_format)
            return self._solve_challenge(challenge, title_for_logging=challenge_id, challenge_id=challenge_id)

        except Exception as e:
            return {"error": f"Failed to create challenge from dataset: {e}"}

    def solve_challenge_from_config(self, config_file: str) -> dict:
        """
        Args: config_file (str): Path to JSON configuration file
        Returns: dict: Results of the challenge solving attempt
        """
        try:
            challenge_path = Path(config_file).parent

            with open(config_file, 'r') as f:
                challenge_config = json.load(f)

            # get list of files in the challenge directory
            files = []
            if challenge_path.exists():
                files = [f.name for f in challenge_path.iterdir()
                        if f.is_file() and f.name != "challenge.json"]
            
            service_host = challenge_config.get('box')
            service_port = challenge_config.get('internal_port') or challenge_config.get('port')
            description = challenge_config.get("description", "")
            if service_host is not None and "{box}" in description:
                description = description.replace("{box}", str(service_host))
            if service_port is not None and "{port}" in description:
                description = description.replace("{port}", str(service_port))

            #parse json and create CTFChallenge object
            challenge = CTFChallenge(
                name=challenge_config.get("name", challenge_path.name),
                category=challenge_config.get("category", "misc"),
                description=challenge_config.get("description", "") + " Unless specified already, the flag format for this challenge is flag{{}}",
                files=challenge_config.get("files", files),
                challenge_dir=challenge_path,
                flag=challenge_config.get("flag")
            )

            return self._solve_challenge(challenge)

        except Exception as e:
            return {"error": f"Failed to load challenge config: {e}"}

    def _solve_challenge(self, challenge, title_for_logging=None, challenge_id=None) -> dict:
        """
        Internal method to solve a challenge with the environment setup
        Args:
            challenge (CTFChallenge): The challenge to solve
            title_for_logging (str): The title for logging
            challenge_id (str): The challenge ID for container naming
        Returns: dict: Results of the challenge solving attempt
        """

        results = {
            "challenge_name": challenge.name,
            "category": challenge.category,
            "solved": False,
            "flag_submitted": None,
            "error": None,
            "agent_output": None,
            "cost": 0.0,
        }

        container_suffix = challenge_id if challenge_id else challenge.name
        with SmolagentsCTFEnvironment(challenge, container_suffix=container_suffix).managed_environment() as env:
            self.environment = env
            self.current_challenge = challenge
            set_ctf_environment(env)

            system = self.get_system_for_category(challenge.category)
            if not system:
                results["error"] = f"No system found for category: {challenge.category}"
                return results

            logger.info(f"Using {system.category} system for challenge: {challenge.name}")
            results = system.solve_challenge(challenge, env, title_for_logging)
            
            if "cost" in results:
                logger.info(f"Challenge '{title_for_logging or challenge.name}' total cost: ${results['cost']:.6f}")
                print(f"[COST] Challenge '{title_for_logging or challenge.name}': spent ${results['cost']:.6f}")

        return results

def read_api_keys(key_file: str) -> dict:
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

def solve_challenges(solver, challenge_ids, base_path, flag_format):
    """
    Solves a list of challenges in serial mode (single run).
    Logs execution to logs_squidagent/single_runs with timestamps and live saving.
    """
    # Define a specific directory for single runs
    single_run_log_dir = os.path.join(os.getcwd(), 'logs_squidagent', 'single_runs')
    os.makedirs(single_run_log_dir, exist_ok=True)

    for challenge_id in challenge_ids:
        logger.info(f"Starting single run for challenge: {challenge_id}")
        
        # Enable stdout capture to catch the agent's print statements
        with run_log(challenge_id, base_dir=single_run_log_dir, timestamp=True, capture_stdout=True) as run_logger:
            try:
                run_logger.log_event(
                    role="system", 
                    content=f"Starting single run solve for {challenge_id}"
                )
                
                # Execute the solver
                # The output from this call (print/logging) will now be caught by run_log
                results = solver.solve_challenge_from_dataset(challenge_id, base_path, flag_format)
                
                # Log completion
                run_logger.log_event(
                    role="system",
                    content="Solver finished execution",
                    meta={"full_results": results}
                )

                # Set success/fail in log
                is_solved = results.get('solved', False)
                error_msg = results.get('error')
                
                run_logger.set_outcome(
                    success=is_solved,
                    exit_reason="solved" if is_solved else "failed",
                    error=error_msg
                )

            except Exception as e:
                logger.error(f"Exception during single run solve for {challenge_id}: {e}")
                run_logger.set_outcome(success=False, exit_reason="exception", error=str(e))
                # We re-raise so the user sees the crash in console too
                raise e

def run_tmux_mode(challenge_ids, dataset, base_path, flag_format, restart_fail=False):
    from tmux_manager import TmuxChallengeManager
    import subprocess
    
    manager = TmuxChallengeManager()
    
    for existing_file in manager.status_dir.glob("*.json"):
        existing_file.unlink()
    
    for existing_file in manager.status_dir.glob("*.log"):
        existing_file.unlink()
    
    def start_challenge(challenge_id):
        session_name = f"squid-{challenge_id}"
        
        if manager.session_exists(session_name):
            manager.kill_session(session_name)
        
        flag_format_arg = flag_format if flag_format else "None"
        
        cmd = f"{sys.executable} {runner_script} {challenge_id} {dataset} {base_path or '.'} {flag_format_arg} {manager.status_dir}"
        
        manager.create_session(session_name, cmd)
        logger.info(f"Started tmux session: {session_name} for challenge: {challenge_id}")
    
    runner_script = Path(__file__).parent / "challenge_runner.py"
    python_path = sys.executable
    
    logger.info(f"Starting {len(challenge_ids)} challenges in tmux sessions...")
    
    for challenge_id in challenge_ids:
        start_challenge(challenge_id)
        time.sleep(0.5)
    
    logger.info("\nAll challenge sessions started.")
    manager.start_master_display()
    
    logger.info(f"\nTo view individual challenge logs:")
    for challenge_id in challenge_ids:
        logger.info(f"  tmux attach -t squid-{challenge_id}")
    
    logger.info(f"\nTo view master display:")
    logger.info(f"  tmux attach -t {manager.master_session}")
    
    if not restart_fail:
        return 0
    
    logger.info("\nRestart-fail mode enabled. Monitoring and restarting failed challenges...")
    
    while True:
        time.sleep(5)
        
        for challenge_id in challenge_ids:
            status_data = manager.read_status(challenge_id)
            
            if status_data and status_data.get('state') == 'completed':
                if status_data.get('result') == 'failed':
                    restart_count = status_data.get('restart_count', 0) + 1
                    status_data['restart_count'] = restart_count
                    manager.update_status(challenge_id, status_data)
                    logger.info(f"Challenge {challenge_id} failed. Restarting (attempt {restart_count})...")
                    start_challenge(challenge_id)
                    time.sleep(0.5)
                elif status_data.get('result') == 'success':
                    logger.info(f"Challenge {challenge_id} succeeded!")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="SquidAgent CTF Solver")
    parser.add_argument("--emc", action="store_true", help="Enable multi-challenge solving with tmux sessions")
    parser.add_argument("--dataset", default=None, help="Path to challenges.json dataset file")
    parser.add_argument("--challenge", default=None, help="Challenge ID(s) from the dataset (use comma to separate multiple challenges for multi-challenge mode)")
    parser.add_argument("--category", default=None, help="Filter challenges by category (e.g., pwn, crypto, rev, web, forensics, misc). Only works with --emc.")
    parser.add_argument("--base-path", default="../.", help="Base path for resolving challenge paths (default: current directory)")
    parser.add_argument("--list", action="store_true", help="List available challenges in the dataset")
    parser.add_argument("--flag-format", default=None, help="Set the flag format for this challenge")
    parser.add_argument("--restart-fail", action="store_true", help="Clear all status files in the status directory")
    parser.add_argument("--no-rag-reset", action="store_true", help="Skip RAG database reset on startup")

    args = parser.parse_args()
    
    if not args.no_rag_reset:
        logger.info("Resetting RAG database on startup...")
        SquidAgentSolver.reset_rag_database()
    else:
        logger.info("Skipping RAG database reset (--no-rag-reset flag set)")
    
    keys_file = Path(__file__).parent.parent / "keys.cfg"
    api_keys = read_api_keys(str(keys_file))
    args.api_key = api_keys.get("OPENAI_API_KEY")
    args.anthropic_api_key = api_keys.get("ANTHROPIC_API_KEY")
    args.openrouter_api_key = api_keys.get("OPENROUTER_API_KEY")
    args.deepseek_api_key = api_keys.get("DEEPSEEK_API_KEY")

    solver = SquidAgentSolver(api_key=args.api_key, anthropic_api_key=args.anthropic_api_key, openrouter_api_key=args.openrouter_api_key, deepseek_api_key=args.deepseek_api_key)

    if not args.dataset:
        logger.error("Dataset is required")
        return 1

    if not solver.load_dataset(args.dataset):
        logger.error("Failed to load dataset")
        return 1

    if args.category and not args.emc:
        logger.error("--category option can only be used with --emc")
        return 1

    if args.list:
        challenges = solver.get_challenge_list()
        logger.info("Available challenges:")
        for challenge_id in challenges:
            dataset_entry = solver.dataset[challenge_id]
            logger.info(f"  {challenge_id}: {dataset_entry.get('category', 'unknown')} - {dataset_entry.get('path', 'unknown path')}")
        return 0

    if args.emc:
        if args.challenge is None:
            challenge_ids = solver.get_challenge_list(category=args.category)
        else:
            challenge_ids = [c.strip() for c in args.challenge.split(',')]

            # If category is specified, filter the provided challenge IDs
            if args.category:
                target_category = SquidAgentSolver.normalize_category(args.category)
                filtered = []
                for challenge_id in challenge_ids:
                    if challenge_id in solver.dataset:
                        challenge_category = solver.dataset[challenge_id].get('category', '')
                        normalized_challenge_category = SquidAgentSolver.normalize_category(challenge_category)
                        if normalized_challenge_category == target_category:
                            filtered.append(challenge_id)
                challenge_ids = filtered
        
        if not challenge_ids:
            logger.error(f"No challenges found for category: {args.category}" if args.category else "No challenges found")
            return 1
        
        return run_tmux_mode(challenge_ids, args.dataset, args.base_path, args.flag_format, args.restart_fail)
    
    if not args.challenge:
        logger.error("Challenge is required unless using --list or --emc")
        return 1
    
    challenge_ids = [args.challenge]
    
    logger.info(f"Solving CTF challenges: {', '.join(challenge_ids)}")
    base_path = args.base_path if hasattr(args, 'base_path') and args.base_path != "." else None
    
    solve_challenges(solver, challenge_ids, base_path, args.flag_format)

    return 0

if __name__ == "__main__":
    sys.exit(main())
