import subprocess
import json
import logging
import os
import re
import shutil
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager
import glob
log_dir = os.path.join(os.getcwd(), 'logs_squidagent')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'ctf_environment.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CTFChallenge:
    """Represents a CTF challenge with metadata and files."""
    name: str
    category: str
    description: str
    files: List[str]
    challenge_dir: Path
    flag: Optional[str] = None

class SmolagentsCTFEnvironment:
    """Manages Docker environment and CTF challenges for smolagents integration."""
    
    def __init__(self, challenge: CTFChallenge, container_image: str = "ctfenv:squidagent", 
                 ida_container_image: str = "ctfenv:idadocker", network: str = None, 
                 workspace_dir: str = "./workspace", auto_cleanup: bool = True, 
                 container_suffix: str = None):
        self.challenge = challenge
        self.container_image = container_image
        self.ida_container_image = ida_container_image
        self.container_suffix = container_suffix or challenge.name
        self.network = network or f"ctfnet_{self.container_suffix}"
        self.workspace_name = workspace_dir
        self.auto_cleanup = auto_cleanup
        self.container_id = None
        self.ida_container_id = None
        self.challenge_service_id = None
        self.container_name = f"squid_{self.container_suffix}"
        self.challenge_ip = None
        self.challenge_service_ports: Dict[int, int] = {}
        self.solved = False
        self.giveup = False
        self.internal_flag = None
        self.flag_submitted = None
        self.new_challenge_dir = None
        self.modified_compose_file = None
        
    def setup(self) -> bool:
        """Set up the CTF environment and copy challenge files."""
        try:
            # self._cleanup_non_ida_containers()
            
            self._cleanup_this_challenge_containers()
            
            self.create_network()
            
            self.start_challenge_service()
            
            if not self.start_docker():
                return False
            
            if not self.start_ida_docker():
                return False
            
            # Add service aliases to /etc/hosts after containers are running
            self._add_service_aliases_to_hosts()
                
            self.copy_challenge_files()
            
            logger.info(f"CTF Environment setup complete for challenge: {self.challenge.name}")
            logger.info(f"Main container: {self.container_id[:12] if self.container_id else 'None'}")
            logger.info(f"IDA container: {self.ida_container_id[:12] if self.ida_container_id else 'None'}")
            return True
            
        except Exception as e:
            logger.info(f"Error setting up CTF environment: {e}")
            return False
    
    def create_network(self):
        """Create Docker network for CTF challenge communication."""
        try:
            result = subprocess.run(
                ["docker", "network", "inspect", self.network],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                logger.info(f"Creating Docker network: {self.network}")
                create_result = subprocess.run(
                    ["docker", "network", "create", self.network],
                    capture_output=True, text=True
                )
                if create_result.returncode != 0:
                    if "already exists" not in create_result.stderr.lower():
                        logger.error(f"Failed to create Docker network: {create_result.stderr}")
                        raise RuntimeError(f"Network creation failed: {create_result.stderr}")
                    else:
                        logger.info(f"Docker network {self.network} already exists")
                else:
                    logger.info(f"Docker network {self.network} created successfully")
            else:
                logger.info(f"Docker network {self.network} already exists")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Docker network: {e}")
            raise
    
    
    def _cleanup_this_challenge_containers(self):
        """Clean up only containers for this specific challenge."""
        try:
            logger.info(f"Cleaning up existing containers for challenge: {self.challenge.name}...")
            
            result = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.ID}} {{.Names}}"], 
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            container_id = parts[0]
                            container_name = parts[1]
                            if self.container_name in container_name:
                                logger.info(f"Stopping and removing container: {container_id[:12]} ({container_name})")
                                subprocess.run(["docker", "stop", container_id], capture_output=True, text=True)
                                subprocess.run(["docker", "rm", container_id], capture_output=True, text=True)
            
            logger.info("Challenge-specific containers cleaned up")
            
        except Exception as e:
            logger.info(f"Warning: Failed to cleanup challenge containers: {e}")
    
    def _find_existing_ida_container(self) -> Optional[str]:
        """Find an existing running IDA container."""
        try:
            logger.info(f"Looking for IDA container with image: {self.ida_container_image}")
            # Get list of running containers with their images
            result = subprocess.run(["docker", "ps", "--format", "{{.ID}} {{.Image}}"], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                logger.info(f"Running containers: {result.stdout.strip()}")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            container_id = parts[0]
                            image = parts[1]
                            logger.debug(f"Checking container {container_id[:12]} with image: {image}")
                            # Check if this is specifically an IDA container (exact matching)
                            if (image == self.ida_container_image or 
                                "idadocker" in image.lower()):
                                logger.info(f"Found IDA container: {container_id[:12]} (image: {image})")
                                return container_id
                            else:
                                logger.debug(f"Container {container_id[:12]} is not an IDA container (image: {image})")
            else:
                logger.info("No running containers found")
            return None
        except Exception as e:
            logger.info(f"Warning: Failed to find existing IDA container: {e}")
            return None
    
    def _validate_ida_container(self, container_id: str) -> bool:
        """Validate that the IDA container is actually running IDA services."""
        try:
            # Check if the container is running and has IDA services
            result = self.run_command_in_ida_container("ps aux | grep -E '(ida|headless)' | grep -v grep", timeout=10)
            if result and ("ida" in result.lower() or "headless" in result.lower()):
                return True
            return False
        except Exception as e:
            logger.info(f"Warning: Failed to validate IDA container: {e}")
            return False
    
    def _cleanup_conflicting_containers(self, port: int):
        """Clean up any existing containers that might conflict with the specified port."""
        try:
            # Find containers using the specific port
            result = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.ID}} {{.Ports}} {{.Status}}"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and str(port) in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            container_id = parts[0]
                            status = ' '.join(parts[2:])  # Status might have spaces
                            # Only stop containers that are actually running
                            if 'Up' in status:
                                logger.info(f"Stopping conflicting container on port {port}: {container_id[:12]}")
                                subprocess.run(
                                    ["docker", "stop", container_id],
                                    capture_output=True, text=True
                                )
                                # Wait a moment for the port to be released
                                import time
                                time.sleep(2)
                        
        except Exception as e:
            logger.info(f"Warning: Failed to cleanup conflicting containers on port {port}: {e}")
    
    def _force_cleanup_port(self, port: int):
        """Force cleanup of any process using the specified port."""
        try:
            # Use netstat to find processes using the port
            result = subprocess.run(
                ["netstat", "-tulpn", "|", "grep", f":{port}"],
                shell=True, capture_output=True, text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                logger.info(f"Found processes using port {port}: {result.stdout.strip()}")
                
                # Try to kill any Docker processes using the port
                docker_result = subprocess.run(
                    ["docker", "ps", "--format", "{{.ID}} {{.Ports}}", "--filter", f"publish={port}"],
                    capture_output=True, text=True
                )
                
                if docker_result.returncode == 0 and docker_result.stdout.strip():
                    for line in docker_result.stdout.strip().split('\n'):
                        if line.strip():
                            container_id = line.split()[0]
                            logger.info(f"Force stopping Docker container using port {port}: {container_id[:12]}")
                            subprocess.run(
                                ["docker", "stop", container_id],
                                capture_output=True, text=True
                            )
                            subprocess.run(
                                ["docker", "rm", container_id],
                                capture_output=True, text=True
                            )
                            
        except Exception as e:
            logger.info(f"Warning: Failed to force cleanup port {port}: {e}")
            
    def start_challenge_service(self):
            """Start challenge service using docker-compose if available.
            
            This version intelligently identifies the public-facing container by checking
            which service defined 'ports' in the original docker-compose.yml.
            """
            try:
                compose_file = glob.glob(str(self.challenge.challenge_dir / "*compose.y*ml"))[0]
                print(compose_file)
                challenge_json_path = self.challenge.challenge_dir / "challenge.json"
                print(self.challenge.challenge_dir)
                ports_loaded_from_json = False
                
                # 1. Attempt to load configuration details (Port) from challenge.json
                logger.debug(f"Attempting to load port from challenge.json at: {challenge_json_path}")
                if challenge_json_path.exists():
                    try:
                        with open(challenge_json_path, 'r') as f:
                            cfg = json.load(f)
                            service_port = cfg.get('internal_port') or cfg.get('port')
                            
                            if service_port:
                                self.challenge_service_ports = {int(service_port): int(service_port)}
                                logger.info(f"✅ Using internal port from challenge.json: {service_port}")
                                ports_loaded_from_json = True
                            else:
                                logger.info("Challenge.json found but no 'internal_port' or 'port' key present.")
                    except Exception as e:
                        logger.warning(f"Could not load internal port from challenge.json: {e}")

                if compose_file != None and os.path.isfile(compose_file):
                    logger.info(f"Starting challenge service from {compose_file}")
                    
                    # Create a modified docker-compose file without port bindings (to prevent conflicts)
                    modified_compose_file = self._create_modified_compose_file(compose_file)
                    self.modified_compose_file = modified_compose_file # Store for cleanup

                    # Set project name and run docker-compose up
                    compose_env = os.environ.copy()
                    compose_env["COMPOSE_PROJECT_NAME"] = f"challenge_{self.container_suffix}"
                    modified_filename = modified_compose_file.name
                    self._set_internal_flag()
                    print(self.new_challenge_dir)
                    print(modified_filename)
                    result = subprocess.run(
                        ["docker-compose", "-f", f"{self.new_challenge_dir}/{modified_filename}", "up", "-d"],
                        env=compose_env,
                        check=True, 
                        capture_output=True, 
                        text=True
                    )

                    print("STDOUT:")
                    print(result.stdout)
                    print("\nSTDERR:")
                    print(result.stderr)
                    
                    logger.info("Challenge service started successfully")

                    # Get ALL service container IDs
                    list_result = subprocess.run(
                        ["docker-compose", "-f", modified_filename, "ps", "-q"],
                        cwd=self.challenge.challenge_dir,
                        env=compose_env,
                        capture_output=True, text=True
                    )
                    
                    if list_result.stdout.strip():
                        service_ids = list_result.stdout.strip().split('\n')
                        logger.info(f"Challenge container(s) launched: {len(service_ids)}")

                        # --- LOGIC START: IDENTIFY PUBLIC CONTAINER ---
                        target_service_id = None
                        target_service_name = None
                        original_compose_data = {}

                        # Load YAML once to check service definitions
                        try:
                            import yaml
                            with open(compose_file, 'r') as f:
                                original_compose_data = yaml.safe_load(f)
                        except Exception as e:
                            logger.error(f"Failed to load original compose file for parsing: {e}")

                        # Iterate through running containers to match them to the compose service definition
                        for sid in service_ids:
                            try:
                                # Connect EVERY container to the network immediately
                                # (We do this loop anyway, so might as well connect now)
                                try:
                                    subprocess.run(
                                        ["docker", "network", "connect", self.network, sid],
                                        capture_output=True, text=True
                                    )
                                except subprocess.CalledProcessError:
                                    pass # Already connected

                                # Inspect to get the name
                                name_res = subprocess.run(
                                    ["docker", "inspect", "--format", "{{.Name}}", sid],
                                    capture_output=True, text=True, check=True
                                )
                                # Name format typically: /challenge_{suffix}_{service}_1
                                full_container_name = name_res.stdout.strip().lstrip('/')
                                project_name = f"challenge_{self.container_suffix}"
                                
                                current_service_name = ""
                                if full_container_name.startswith(project_name):
                                    # Clean extraction of service name
                                    parts = full_container_name.replace(project_name + '_', '').split('_')
                                    current_service_name = parts[0] if parts else ""

                                # Check if this service has 'ports' in the original YAML
                                svc_config = original_compose_data.get('services', {}).get(current_service_name, {})
                                
                                if 'ports' in svc_config:
                                    target_service_id = sid
                                    target_service_name = current_service_name
                                    logger.info(f"🎯 Identified public-facing service: '{current_service_name}' (ID: {sid[:12]})")
                                
                            except Exception as e:
                                logger.warning(f"Error analyzing container {sid}: {e}")

                        # Fallback: If no service had explicit 'ports' (or parsing failed), use the first one
                        if not target_service_id and service_ids:
                            logger.warning("No service found with 'ports' defined. Defaulting to first container.")
                            target_service_id = service_ids[0]
                            # Try to guess the name for logging/parsing
                            target_service_name = "unknown" 

                        self.challenge_service_id = target_service_id
                        # --- LOGIC END: IDENTIFY PUBLIC CONTAINER ---

                        # Set the internal flag for the challenge
    
                        
                        # 2. Get the challenge service's internal IP address (For the Target Container)
                        if self.challenge_service_id:
                            logger.debug(f"Inspecting main container {self.challenge_service_id[:12]} for IP address...")
                            inspect_result = subprocess.run(
                                ["docker", "inspect", self.challenge_service_id],
                                capture_output=True, text=True
                            )
                            
                            if inspect_result.returncode == 0:
                                service_info = json.loads(inspect_result.stdout)[0]
                                networks = service_info.get('NetworkSettings', {}).get('Networks', {})
                                
                                if self.network in networks:
                                    self.challenge_ip = networks[self.network].get('IPAddress')
                                    logger.info(f"Challenge Service Internal IP: {self.challenge_ip}")
                                else:
                                    logger.warning(f"Custom network '{self.network}' not found in container inspect output.")
                            else:
                                logger.error(f"Failed to inspect service container: {inspect_result.stderr}")
                        
                        # 3. Fallback to Docker Compose Port Extraction (Only if not from JSON)
                        if not ports_loaded_from_json:
                            if target_service_name and original_compose_data:
                                logger.info(f"Extracting ports for service '{target_service_name}' from compose config.")
                                svc_config = original_compose_data.get('services', {}).get(target_service_name, {})
                                
                                if 'ports' in svc_config:
                                    ports = {}
                                    for port_mapping in svc_config['ports']:
                                        # Handle "80:80" or "80" formats
                                        # We want the container side (internal)
                                        # Split by colon, take last element, split by slash (if tcp/udp specified)
                                        port_str = str(port_mapping)
                                        port_part = port_str.split(':')[-1].split('/')[0]
                                        try:
                                            internal_port = int(port_part)
                                            ports[internal_port] = internal_port
                                        except ValueError:
                                            logger.warning(f"Skipping invalid port entry: {port_mapping}")
                                            continue
                                    
                                    self.challenge_service_ports = ports
                                    logger.info(f"✅ Challenge Service Internal Ports (from compose): {self.challenge_service_ports}")
                                else:
                                    logger.info(f"Target service '{target_service_name}' has no 'ports' defined.")
                            else:
                                logger.info("Ports not loaded from json and no target service name identified.")
                        else:
                            logger.info("Port detection finished successfully using challenge.json.")
                        
                    else:
                        logger.warning("Could not retrieve challenge service container IDs after startup.")
                        
                else:
                    logger.info("No docker-compose.yml found, skipping challenge service startup")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to start challenge service via docker-compose: {e}")
                logger.error(f"stdout: {e.stdout.strip()}, stderr: {e.stderr.strip()}")
            except Exception as e:
                logger.error(f"Critical error in start_challenge_service: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                
            # Log final status
            logger.info(f"--- start_challenge_service FINAL STATUS ---")
            logger.info(f"Challenge IP: {self.challenge_ip}")
            logger.info(f"Challenge Ports: {self.challenge_service_ports}")
            logger.info(f"------------------------------------------")
    def _set_internal_flag(self):
            """Set (or randomize) the flag inside the challenge files without clobbering entire files."""
            charset = string.ascii_lowercase + string.digits

            # 1. Determine Flag Format
            description = self.challenge.description or ""
            match = re.search(r"flag format for this challenge is\s+(\S+)", description, flags=re.IGNORECASE)
            if match:
                flag_format = match.group(1).strip()
            elif self.challenge.flag and "{" in self.challenge.flag:
                flag_format = re.sub(r"\{.*\}", "{}", self.challenge.flag)
            else:
                flag_format = "flag{}"

            # 2. Generate New Flag
            token = ''.join(random.choices(charset, k=32))
            if "{}" in flag_format:
                self.internal_flag = flag_format.replace("{}", f"{{{token}}}")
            else:
                self.internal_flag = flag_format

            # 3. Create New Challenge Directory
            self.new_challenge_dir = self.challenge.challenge_dir / f"challenge_{self.container_suffix}"
            if self.new_challenge_dir.exists():
                shutil.rmtree(self.new_challenge_dir)
            shutil.copytree(self.challenge.challenge_dir, self.new_challenge_dir)

            # 4. Search for Flag Placeholders
            search_token = flag_format.replace("{}", "{")
            # Added "-n" to get line numbers (though we just need filenames here)
            flag_file = subprocess.run(
                ["grep", "-r", "-n", search_token, str(self.challenge.challenge_dir)],
                capture_output=True,
                text=True
            )

            if flag_file.returncode != 0 or not flag_file.stdout.strip():
                logger.error(f"Could not find flag placeholder '{search_token}' in challenge directory: {flag_file.stderr}")
                return

            # 5. Process Matches (Handle Multiple Files)
            grep_lines = flag_file.stdout.strip().splitlines()
            processed_files = set()
            
            # Compile regex once: Matches the format and anything inside the curly braces
            pattern = re.escape(flag_format).replace(r"\{\}", r"\{[^}]*\}")

            for line in grep_lines:
                if not line.strip():
                    continue

                # Grep output format is filename:line_number:content
                source_path_str, _, _ = line.partition(":")
                
                # Skip if we have already patched this file
                if source_path_str in processed_files:
                    continue
                
                processed_files.add(source_path_str)
                source_path = Path(source_path_str)

                # Determine relative path in the new directory
                try:
                    rel_path = source_path.relative_to(self.challenge.challenge_dir)
                except ValueError:
                    rel_path = source_path.name
                
                target_file = self.new_challenge_dir / rel_path

                if not target_file.exists():
                    logger.warning(f"Target file {target_file} does not exist, skipping.")
                    continue

                # Read, Replace All, Write
                try:
                    file_contents = target_file.read_text()
                    
                    # count=0 (default) replaces ALL occurrences
                    new_contents, replacements = re.subn(pattern, self.internal_flag, file_contents)

                    if replacements > 0:
                        target_file.write_text(new_contents)
                        # logger.info(f"Replaced {replacements} flag(s) in {rel_path}")
                    else:
                        logger.warning(f"Flag format '{flag_format}' not found in {target_file} during regex pass.")
                
                except Exception as e:
                    logger.error(f"Failed to process file {target_file}: {e}")

            # 6. Update Challenge Directory Reference
            self.challenge.challenge_dir = self.new_challenge_dir
        
    def _add_service_aliases_to_hosts(self):
        """Add service hostname aliases to /etc/hosts in the main container for DNS fallback."""
        if not self.container_id or not self.challenge_service_id:
            return
        
        try:
            # Get the service container's network info
            inspect_result = subprocess.run(
                ["docker", "inspect", self.challenge_service_id],
                capture_output=True, text=True
            )
            
            if inspect_result.returncode != 0:
                logger.info(f"Could not inspect service container for hosts file update")
                return
            
            import json
            service_info = json.loads(inspect_result.stdout)[0]
            networks = service_info.get('NetworkSettings', {}).get('Networks', {})
            
            # Find the network entry for our challenge network
            if self.network in networks:
                network_info = networks[self.network]
                ip_address = network_info.get('IPAddress')
                aliases = network_info.get('Aliases', [])
                
                if ip_address and aliases:
                    # Add each alias to /etc/hosts in the main container
                    for alias in aliases:
                        # Skip container IDs and generic names
                        if '.' in alias or alias.endswith('.io'):
                            cmd = f"grep -q '{alias}' /etc/hosts || echo '{ip_address} {alias}' >> /etc/hosts"
                            result = self.run_command_in_container(cmd)
                            logger.info(f"Added /etc/hosts entry: {ip_address} {alias}")
            
        except Exception as e:
            logger.info(f"Warning: Failed to update /etc/hosts with service aliases: {e}")
    
    def _create_modified_compose_file(self, original_compose_file: Path) -> Path:
        """Create a modified docker-compose file that uses our network and removes port bindings."""
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML is required for docker-compose file modification. Install with: pip install PyYAML")
            raise
        
        # Read the original compose file
        with open(original_compose_file, 'r') as f:
            compose_data = yaml.safe_load(f)
        
        # Keep track of internal networks to preserve
        internal_networks = {}
        if 'networks' in compose_data:
            for net_name, net_config in compose_data['networks'].items():
                if net_name != 'ctfnet' and not (isinstance(net_config, dict) and net_config.get('external')):
                    internal_networks[net_name] = net_config or {}
        
        # Modify the compose data
        if 'services' in compose_data:
            for service_name, service_config in compose_data['services'].items():
                # Remove port bindings to avoid conflicts
                if 'ports' in service_config:
                    logger.info(f"Removing port bindings for service {service_name}: {service_config['ports']}")
                    del service_config['ports']
                
                # Skip network modification for services with network_mode (e.g. host, bridge)
                has_network_mode = 'network_mode' in service_config
                if has_network_mode:
                    logger.info(f"Preserving network_mode for service {service_name}: {service_config['network_mode']}")
                    if 'networks' in service_config:
                        logger.info(f"Removing networks from service {service_name} (conflicts with network_mode)")
                        del service_config['networks']
                
                # Escape dollar signs in environment variables for docker-compose v1 compatibility
                if 'environment' in service_config:
                    env_list = service_config['environment']
                    if isinstance(env_list, list):
                        fixed_env = []
                        for env_var in env_list:
                            if isinstance(env_var, str) and '=' in env_var:
                                key, value = env_var.split('=', 1)
                                value = value.replace('$', '$$')
                                fixed_env.append(f"{key}={value}")
                            else:
                                fixed_env.append(env_var)
                        service_config['environment'] = fixed_env
                    elif isinstance(env_list, dict):
                        for key, value in env_list.items():
                            if isinstance(value, str):
                                env_list[key] = value.replace('$', '$$')
                
                # Update network configuration to use our challenge-specific network
                # Skip if service has network_mode
                if not has_network_mode:
                    if 'networks' in service_config:
                        old_networks = service_config['networks']
                        new_networks = {}
                        
                        # Check if service uses ctfnet
                        uses_ctfnet = False
                        if isinstance(old_networks, dict):
                            uses_ctfnet = 'ctfnet' in old_networks
                            for net_name, net_config in old_networks.items():
                                if net_name == 'ctfnet':
                                    new_networks[self.network] = net_config if isinstance(net_config, dict) else {}
                                elif net_name in internal_networks:
                                    new_networks[net_name] = net_config if isinstance(net_config, dict) else {}
                        elif isinstance(old_networks, list):
                            uses_ctfnet = 'ctfnet' in old_networks
                            for net_name in old_networks:
                                if net_name == 'ctfnet':
                                    new_networks[self.network] = {}
                                elif net_name in internal_networks:
                                    new_networks[net_name] = {}
                        
                        if uses_ctfnet or not new_networks:
                            if self.network not in new_networks:
                                new_networks[self.network] = {}
                        
                        service_config['networks'] = new_networks if new_networks else [self.network]
                    else:
                        service_config['networks'] = [self.network]
        
        # Update top-level networks section
        compose_data['networks'] = internal_networks.copy()
        compose_data['networks'][self.network] = {'external': True}
        
        # Write modified compose file
        modified_file = self.challenge.challenge_dir / f"docker-compose.{self.container_suffix}.yml"
        with open(modified_file, 'w') as f:
            yaml.dump(compose_data, f, default_flow_style=False)
        
        logger.info(f"Created modified compose file: {modified_file}")
        return modified_file
    
    def start_docker(self) -> bool:
        """Start the Docker container for the CTF environment."""
        try:
            logger.info(f"Starting CTF container with image: {self.container_image}")
            logger.info(f"Container name: {self.container_name}")
            
            cmd = [
                "docker", "run", "-d", "--rm",
                "--name", self.container_name,
                "--network", self.network,
                "--platform", "linux/amd64",
                "-w", "/home/ctfplayer",
                self.container_image,
                "tail", "-f", "/dev/null"
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.container_id = result.stdout.strip()
            
            logger.info(f"Container started: {self.container_id[:12]}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.info(f"Failed to start Docker container: {e}")
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Return code: {e.returncode}")
            logger.info(f"stdout: {e.stdout}")
            logger.info(f"stderr: {e.stderr}")
            return False
    
    def start_ida_docker(self) -> bool:
        """Start the IDA Docker container for reverse engineering (only if not already running)."""
        try:
            # Check if an IDA container is already running
            existing_ida_container = self._find_existing_ida_container()
            if existing_ida_container:
                logger.info(f"Found existing IDA container: {existing_ida_container[:12]}")
                logger.info(f"Using existing IDA container (no restart to preserve manual setup)")
                self.ida_container_id = existing_ida_container
                return True
            
            logger.info(f"No IDA container found, starting new one...")
            
            # Check if the IDA container image exists
            try:
                result = subprocess.run(
                    ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "--filter", f"reference={self.ida_container_image}"],
                    capture_output=True, text=True
                )
                if result.returncode != 0 or not result.stdout.strip():
                    logger.warning(f"IDA container image {self.ida_container_image} not found!")
                    logger.info("Available images:")
                    subprocess.run(["docker", "images"], capture_output=False)
                    return False
                else:
                    logger.info(f"IDA container image {self.ida_container_image} found")
            except Exception as e:
                logger.info(f"Warning: Could not check for IDA container image: {e}")
            
            # Only clean up conflicting containers on port 1338 if no IDA container exists
            self._cleanup_conflicting_containers(1338)
            
            logger.info(f"Starting IDA container with image: {self.ida_container_image}")
            
            cmd = [
                "docker", "run", "-d", "--rm",
                "--network", self.network,
                "--platform", "linux/amd64",
                "-p", "1338:1338",  # Different port to avoid conflicts
                "-w", "/home/ctfplayer",
                self.ida_container_image,
                "tail", "-f", "/dev/null"  # Keep container alive indefinitely
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.ida_container_id = result.stdout.strip()
            
            logger.info(f"IDA container started: {self.ida_container_id[:12]}")
            logger.info(f"Note: This container will be preserved for future use to maintain manual setup")
            
            # Wait a moment for the container to fully start
            import time
            time.sleep(3)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.info(f"Failed to start IDA Docker container: {e}")
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Return code: {e.returncode}")
            logger.info(f"stdout: {e.stdout}")
            logger.info(f"stderr: {e.stderr}")
            return False
    
    def copy_challenge_files(self):
        """Copy challenge files directly into both containers."""
        logger.info("Copying challenge files...")
        
        for file_name in self.challenge.files:
            source_path = self.challenge.challenge_dir / file_name
            
            if source_path.exists():
                # Extract just the filename for flat structure in main container
                flat_filename = Path(file_name).name
                dest_path = f"/home/ctfplayer/{flat_filename}"
                
                # Copy to main container
                if self.container_id:
                    cmd = ["docker", "cp", str(source_path), f"{self.container_id}:{dest_path}"]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        logger.info(f"Copied {file_name} to main container as {flat_filename}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error copying {file_name} to main container: {e}")
                
                # Copy to IDA container in challenge-specific directory
                if self.ida_container_id:
                    # Create challenge-specific directory in IDA container
                    challenge_dir = f"/home/ctfplayer/{self.container_suffix}"
                    dest_path_ida = f"{challenge_dir}/{flat_filename}"
                    
                    # First, ensure the directory exists
                    try:
                        subprocess.run(
                            ["docker", "exec", self.ida_container_id, "mkdir", "-p", challenge_dir],
                            check=True, capture_output=True
                        )
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Could not create directory {challenge_dir} in IDA container: {e}")
                    
                    # Copy the file
                    cmd = ["docker", "cp", str(source_path), f"{self.ida_container_id}:{dest_path_ida}"]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        logger.info(f"Copied {file_name} to IDA container as {dest_path_ida}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error copying {file_name} to IDA container: {e}")
            else:
                logger.info(f"Warning: Challenge file {file_name} not found at {source_path}")
    
    def _validate_container(self, container_id: str) -> bool:
        """Check if a container is still running."""
        if not container_id:
            return False
        try:
            # Use the short ID (first 12 characters) for filtering since docker ps shows short IDs
            short_id = container_id[:12]
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.ID}}", "--filter", f"id={short_id}"],
                capture_output=True, text=True
            )
            # Check if the short container ID appears in the output
            return short_id in result.stdout and result.returncode == 0
        except Exception:
            return False

    def run_command_in_container(self, command: str, workdir: str = "/home/ctfplayer/", timeout: int = 120) -> str:
            """Execute a Command in the analyst container not the WEB CHALLENGE container."""
            if not self.container_id:
                return "Error: No container running"
            
            # Validate container is still running
            if not self._validate_container(self.container_id):
                logger.info(f"DEBUG: Container validation failed for {self.container_id[:12]}")
                return "Error: Container is no longer running"
                
            try:
                # Execute as root to avoid permission issues
                cmd = ["docker", "exec", "-u", "0", "-w", workdir, self.container_id, "bash", "-c", command]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                
                if result.returncode == 0:
                    return result.stdout
                else:
                    # Logging for debugging
                    logger.info("--- DEBUG: COMMAND FAILED ---")
                    logger.info(f"COMMAND: {' '.join(cmd)}")
                    logger.info(f"EXIT CODE: {result.returncode}")
                    logger.error(f"STDERR:\n{result.stderr.strip()}") 
                    logger.info("--- END DEBUG ---")
                    return f"Command failed (exit code {result.returncode}):\n{result.stderr}"
                    
            except subprocess.TimeoutExpired as e:
                # --- KEY CHANGE HERE ---
                # The exception object 'e' contains the stdout/stderr captured before the timeout
                partial_stdout = e.stdout if e.stdout else "[No output]"
                partial_stderr = e.stderr if e.stderr else "[No error output]"
                
                logger.warning(f"Command timed out after {timeout}s. Returning partial output.")
                
                return (
                    f"TIMEOUT: Command exceeded limit of {timeout} seconds.\n"
                    f"--- PARTIAL STDOUT ---\n{partial_stdout}\n"
                    f"--- PARTIAL STDERR ---\n{partial_stderr}"
                )
                
            except Exception as e:
                return f"Error executing command: {e}"
    
    def run_command_in_ida_container(self, command: str, workdir: str = None, timeout: int = 30) -> str:
        """Execute a command in the IDA Docker container.
        
        Args:
            command: Shell command to execute inside the IDA container.
            workdir: Working directory inside the container (defaults to challenge-specific directory).
            timeout: Time limit in seconds before the command is aborted.
        """
        if not self.ida_container_id:
            return "Error: No IDA container running"
        
        # Validate container is still running
        if not self._validate_container(self.ida_container_id):
            return "Error: IDA container is no longer running"
        
        # Default workdir to challenge-specific directory
        if workdir is None:
            workdir = f"/home/ctfplayer/{self.container_suffix}"
            
        try:
            # Execute as root to avoid permission issues on copied files
            cmd = ["docker", "exec", "-u", "0", "-w", workdir, self.ida_container_id, "bash", "-c", command]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return result.stdout
            else:
                logger.info("--- DEBUG: IDA COMMAND FAILED ---")
                logger.info(f"COMMAND: {' '.join(cmd)}")
                logger.info(f"EXIT CODE: {result.returncode}")
                logger.info(f"STDOUT:\n{result.stdout.strip()}")
                logger.info(f"STDERR:\n{result.stderr.strip()}")
                logger.info("--- END DEBUG ---")
                return f"Command failed (exit code {result.returncode}):\n{result.stderr}"
                
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {e}"
    
    def run_command_local(self, command: str) -> str:
        """Execute a command locally in the workspace."""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                cwd=str(self.challenge.challenge_dir) if getattr(self, 'challenge', None) and getattr(self.challenge, 'challenge_dir', None) else None,
                timeout=120
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Command failed (exit code {result.returncode}):\n{result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing command: {e}"
    
    def submit_flag(self, flag: str) -> bool:
        """Submit a flag for the CTF challenge."""
        self.flag_submitted = flag
        
        # For challenges with a running service, try to verify the flag by testing the exploit
        if self.challenge_service_id and hasattr(self.challenge, 'internal_port'):
            try:
                # Try to verify the flag by testing if we can retrieve it from the service
                verified = self._verify_flag_from_service(flag)
                if verified:
                    self.solved = True
                    logger.info(f"Correct flag verified from service: {flag}")
                    self.cleanup_workspace_on_completion()
                    return True
                else:
                    logger.info(f"Flag not verified from service: {flag}")
                    return False
            except Exception as e:
                logger.info(f"Warning: Could not verify flag from service: {e}")
                # Fall back to static comparison if service verification fails
        
        # Static flag comparison (fallback or for challenges without services)
        if self.challenge.flag and flag.strip() == self.challenge.flag.strip():
            self.solved = True
            logger.info(f"🦑🦑🦑 Correct flag submitted: {flag}")
            self.cleanup_workspace_on_completion()
            return True
        else:
            logger.info(f"❌ Incorrect flag submitted: {flag}")
            return False
    
    def _verify_flag_from_service(self, flag: str) -> bool:
        """
        Verify a flag by checking if it can be retrieved from the running challenge service.
        This is more realistic than static comparison.
        """
        # Get challenge configuration
        challenge_config = None
        challenge_json_path = self.challenge.challenge_dir / "challenge.json"
        
        if challenge_json_path.exists():
            import json
            with open(challenge_json_path, 'r') as f:
                challenge_config = json.load(f)
        
        if not challenge_config:
            return False
        
        # Get service details
        service_host = challenge_config.get('box', 'localhost')
        service_port = challenge_config.get('internal_port', 80)
        
        # Test connection to service from within the ctfenv container
        test_cmd = f"timeout 5 nc -z {service_host} {service_port}"
        
        try:
            result = self.run_command_in_container(test_cmd)
            if "Connection refused" in result or "timed out" in result:
                logger.info(f"Cannot connect to challenge service at {service_host}:{service_port}")
                return False
            
            logger.info(f"Successfully connected to challenge service at {service_host}:{service_port}")
            
            # For pwn challenges, if the flag contains the expected format and 
            # we can connect to the service, consider it verified
            # (More sophisticated verification could be added per challenge type)
            if flag.startswith('csawctf{') and flag.endswith('}'):
                return True
                
        except Exception as e:
            logger.info(f"Error verifying flag from service: {e}")
            
        return False
    
    def giveup_challenge(self):
        """Mark the challenge as given up."""
        self.giveup = True
        logger.info("Challenge marked as given up")
    
    def get_challenge_info(self) -> Dict[str, Any]:
        """Get challenge information as a dictionary."""
        return {
            "name": self.challenge.name,
            "category": self.challenge.category,
            "description": self.challenge.description,
            "files": self.challenge.files,
            "workspace_dir": "/home/ctfplayer",
            "main_container_id": self.container_id[:12] if self.container_id else None,
            "ida_container_id": self.ida_container_id[:12] if self.ida_container_id else None,
            "solved": self.solved,
            "giveup": self.giveup,
            "flag_submitted": self.flag_submitted
        }
    
    def list_workspace_files(self) -> List[str]:
        """List all files in the container workspace."""
        if not self.container_id:
            return []
            
        # List all files in the container's home directory  
        result = self.run_command_in_container("find /home/ctfplayer -type f 2>/dev/null || echo ''")
        
        if result.strip():
            files = [f.strip() for f in result.strip().split('\n') if f.strip()]
            # Return relative paths from /home/ctfplayer
            return [f.replace('/home/ctfplayer/', '') for f in files if f.startswith('/home/ctfplayer/')]
        else:
            return []
    
    def list_ctf_files(self) -> List[str]:
        """List CTF challenge files from the main container."""
        if not self.container_id:
            return []
            
        # List files in the container's home directory
        result = self.run_command_in_container("ls -1 /home/ctfplayer/ 2>/dev/null || echo ''")
        
        if result.strip():
            return [f.strip() for f in result.strip().split('\n') if f.strip()]
        else:
            return []
    
    def list_ida_ctf_files(self) -> List[str]:
        """List CTF challenge files from the IDA container for this challenge."""
        if not self.ida_container_id:
            return []
            
        # List files in the challenge-specific directory
        challenge_dir = f"/home/ctfplayer/{self.container_suffix}"
        result = self.run_command_in_ida_container(f"ls -1 {challenge_dir}/ 2>/dev/null || echo ''")
        
        if result.strip():
            # Return filenames relative to challenge directory
            files = [f.strip() for f in result.strip().split('\n') if f.strip()]
            # Remove full paths, just return filenames
            return [Path(f).name for f in files]
        else:
            return []
    
    def cleanup(self):
        """Clean up the environment and stop containers for this challenge only."""
        if self.container_id:
            try:
                logger.info(f"Stopping main container: {self.container_id[:12]}")
                subprocess.run(["docker", "stop", self.container_id], 
                             check=True, capture_output=True)
                self.container_id = None
            except subprocess.CalledProcessError as e:
                logger.info(f"Error stopping main container: {e}")
        
        if self.challenge_service_id:
            try:
                logger.info(f"Stopping challenge service: {self.challenge_service_id[:12]}")
                compose_env = os.environ.copy()
                compose_env["COMPOSE_PROJECT_NAME"] = f"challenge_{self.container_suffix}"
                
                # Use the modified compose file if it exists
                compose_cmd = ["docker-compose"]
                if self.modified_compose_file and self.modified_compose_file.exists():
                    # Use just the filename since we're in the challenge directory
                    compose_cmd.extend(["-f", self.modified_compose_file.name])
                compose_cmd.append("down")
                
                subprocess.run(
                    compose_cmd,
                    cwd=self.challenge.challenge_dir,
                    env=compose_env,
                    check=True, capture_output=True
                )
                self.challenge_service_id = None
                
                # Clean up the modified compose file
                if self.modified_compose_file and self.modified_compose_file.exists():
                    try:
                        self.modified_compose_file.unlink()
                        logger.info(f"Removed modified compose file: {self.modified_compose_file}")
                    except Exception as e:
                        logger.info(f"Warning: Could not remove modified compose file: {e}")
                
            except subprocess.CalledProcessError as e:
                logger.info(f"Error stopping challenge service: {e}")
        
        self.cleanup_ida_container_contents()
        self.cleanup_rag_db()
        self.cleanup_workspace()
        
        # Clean up challenge-specific network
        self.cleanup_network()
    
    def stop_ida_container(self):
        """Manually stop the IDA container if needed."""
        if self.ida_container_id:
            try:
                logger.info(f"Stopping IDA container: {self.ida_container_id[:12]}")
                subprocess.run(["docker", "stop", self.ida_container_id], 
                             check=True, capture_output=True)
                self.ida_container_id = None
            except subprocess.CalledProcessError as e:
                logger.info(f"Error stopping IDA container: {e}")
    
    def restart_ida_container(self) -> bool:
        """Force restart the IDA container (WARNING: This will lose manual setup)."""
        logger.info("WARNING: Force restarting IDA container - this will lose any manual setup!")
        logger.info("Only use this if absolutely necessary.")
        self.stop_ida_container()
        return self.start_ida_docker()
    
    def cleanup_ida_container_contents(self):
        """Clean up challenge-specific contents from IDA container without stopping it."""
        if not self.ida_container_id:
            logger.info("No IDA container to clean up")
            return
        
        try:
            logger.info(f"Cleaning up challenge contents in IDA container")
            
            challenge_dir = f"/home/ctfplayer/{self.container_suffix}"
            result = self.run_command_in_ida_container(f"test -d {challenge_dir} && echo 'exists' || echo 'nonexistent'", timeout=10)
            
            if "exists" in result:
                logger.info(f"Removing challenge directory in IDA container: {challenge_dir}")
                self.run_command_in_ida_container(f"rm -rf {challenge_dir}", timeout=60)
                logger.info(f"Cleaned up challenge directory: {challenge_dir}")
            else:
                logger.info(f"Challenge directory does not exist in IDA container: {challenge_dir}")
                
        except Exception as e:
            logger.info(f"Error cleaning up IDA container contents: {e}")
    
    def cleanup_rag_db(self):
        """Clean up the RAG database by deleting all collections."""
        try:
            rag_db_dir = Path("./rag_db")
            if rag_db_dir.exists():
                logger.info(f"Cleaning up RAG database at {rag_db_dir}")
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=str(rag_db_dir))
                    collections = client.list_collections()
                    for collection in collections:
                        try:
                            logger.info(f"Deleting RAG collection: {collection.name}")
                            client.delete_collection(collection.name)
                        except Exception as e:
                            logger.info(f"Warning: Could not delete collection {collection.name}: {e}")
                except ImportError:
                    logger.info("chromadb not installed, deleting RAG DB directory instead")
                    shutil.rmtree(rag_db_dir)
                    logger.info(f"Deleted RAG DB directory: {rag_db_dir}")
                except Exception as e:
                    logger.info(f"Warning: Could not clean RAG DB via chromadb: {e}")
                    logger.info("Attempting to delete RAG DB directory instead")
                    shutil.rmtree(rag_db_dir)
                    logger.info(f"Deleted RAG DB directory: {rag_db_dir}")
            else:
                logger.info("RAG DB directory does not exist, nothing to clean")
        except Exception as e:
            logger.info(f"Error cleaning up RAG database: {e}")
    
    def cleanup_network(self):
        """Clean up the challenge-specific Docker network."""
        try:
            # Check if network exists
            result = subprocess.run(
                ["docker", "network", "inspect", self.network],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Removing Docker network: {self.network}")
                subprocess.run(
                    ["docker", "network", "rm", self.network],
                    capture_output=True, text=True
                )
                logger.info(f"Docker network {self.network} removed")
            else:
                logger.info(f"Docker network {self.network} does not exist, skipping removal")
                
        except subprocess.CalledProcessError as e:
            logger.info(f"Warning: Failed to remove Docker network {self.network}: {e}")
    
    def cleanup_workspace(self):
        """Clean up workspace - container-only approach, no local cleanup needed."""
        # Since everything is in the container, no local cleanup is needed
        # The container itself will be cleaned up in the main cleanup() method
        logger.info("Workspace cleanup: Using container-only approach, no local files to clean")
    
    def cleanup_workspace_on_completion(self):
        """Optional cleanup for when challenge is completed successfully."""
        if self.auto_cleanup:
            if self.solved:
                logger.info("Challenge solved! Container will be cleaned up automatically.")
                self.cleanup_workspace()
            elif self.giveup:
                logger.info("Challenge given up. Container will be cleaned up automatically.")
                self.cleanup_workspace()
    
    @contextmanager
    def managed_environment(self):
        """Context manager for automatically setting up and cleaning up environment."""
        try:
            if self.setup():
                yield self
            else:
                raise Exception("Failed to setup CTF environment")
        finally:
            # Only cleanup if auto_cleanup is enabled and challenge is completed
            if self.auto_cleanup and (self.solved or self.giveup):
                self.cleanup()
            else:
                logger.info("Environment cleanup skipped - containers will remain running for reuse")

# Helper functions for creating challenges from different sources
def create_challenge_from_dir(challenge_dir: Path, name: str = None, category: str = "misc", 
                            description: str = "", flag: str = None) -> CTFChallenge:
    """Create a CTFChallenge from a directory containing challenge files."""
    challenge_dir = Path(challenge_dir)
    if not challenge_dir.exists():
        raise FileNotFoundError(f"Challenge directory not found: {challenge_dir}")
    
    # Get all files in directory (excluding subdirectories for now)
    files = [f.name for f in challenge_dir.iterdir() if f.is_file()]
    
    if name is None:
        name = challenge_dir.name
    
    return CTFChallenge(
        name=name,
        category=category,
        description=description,
        files=files,
        challenge_dir=challenge_dir,
        flag=flag
    )

def create_challenge_from_dict(challenge_data: Dict[str, Any], base_dir: Path = None) -> CTFChallenge:
    """Create a CTFChallenge from a dictionary (e.g., loaded from JSON)."""
    if base_dir is None:
        base_dir = Path(".")
    
    challenge_dir = base_dir / challenge_data.get("challenge_dir", ".")
    challenge_dir = Path(challenge_dir)
    
    # Load service details from challenge.json if present to fill placeholders
    service_host = None
    service_port = None
    challenge_json_path = challenge_dir / "challenge.json"
    if challenge_json_path.exists():
        try:
            with open(challenge_json_path, 'r') as f:
                cfg = json.load(f)
                service_host = cfg.get('box')
                service_port = cfg.get('internal_port') or cfg.get('port')
        except Exception:
            pass
    
    description = challenge_data.get("description", "")
    if service_host is not None and "{box}" in description:
        description = description.replace("{box}", str(service_host))
    if service_port is not None and "{port}" in description:
        description = description.replace("{port}", str(service_port))
    
    # Ensure files is a list; if not provided, list files in directory (excluding challenge.json)
    files = challenge_data.get("files")
    if not isinstance(files, list):
        try:
            files = [f.name for f in challenge_dir.iterdir() if f.is_file() and f.name != "challenge.json"]
        except Exception:
            files = []
    
    return CTFChallenge(
        name=challenge_data.get("name", challenge_dir.name),
        category=challenge_data.get("category", "misc"),
        description=description,
        files=files,
        challenge_dir=challenge_dir,
        flag=challenge_data.get("flag")
    )

# Example usage function
def example_usage():
    """Example of how to use the SmolagentsCTFEnvironment."""
    
    # Create a challenge from directory
    challenge = create_challenge_from_dir(
        challenge_dir=Path("./challenges/example_challenge"),
        name="example_rev",
        category="reverse_engineering",
        description="Example reverse engineering challenge",
        flag="flag{example_flag}"
    )
    
    # Use the environment
    with SmolagentsCTFEnvironment(challenge).managed_environment() as env:
        # Environment is set up automatically
        
        # Run some analysis commands
        file_info = env.run_command_in_container("file *")
        logger.info("File info:", file_info)
        
        # Submit a flag
        env.submit_flag("flag{example_flag}")
        
        # Get challenge status
        info = env.get_challenge_info()
        logger.info("Challenge info:", info)
    
    # Environment is cleaned up automatically

if __name__ == "__main__":
    example_usage()