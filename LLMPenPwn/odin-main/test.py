import json
import multiprocessing
import os
import pathlib
from dataclasses import asdict

from odin.agents import MultiStepAgent
from odin.constants import VulnerabilityFinding
from odin.model import (Message, MessageRole, OpenAIModel,
                        OpenAIResponsesModel, OpenRouterAPIModel)
from odin.studies.exploit import ExploitAnalysisStudy
from odin.tools import BashTool

model = OpenAIResponsesModel(
    model_id="gpt-5-nano-2025-08-07",
    api_key=os.getenv("OPENAI_API_KEY"),
    # base_url=os.getenv("OPENAI_API_BASE_URL"),
    requests_per_minute=60,
    reasoning={"summary": "detailed"}
)

# model = OpenRouterAPIModel(
#     model_id="x-ai/grok-4-fast:free",
#     api_key=os.getenv("OPENROUTER_API_KEY"),
#     requests_per_minute=60,
#     # reasoning_effort="high"
#     extra_body={"reasoning": {"effort": "high", "enabled": True}}
# )

# messages = [
#     Message(role=MessageRole.SYSTEM, content="""You are **Odin**, an expert security engineer, penetration tester and reverse engineer. You will be given repository, codebase, or a website to analyze, and a task to solve.

# All tasks are CTF or security-related challenges, not real-life scenarios or applications. They are safe to explore and analyze, and do not involve any real user data or production systems. The tasks can range from a CTF-style challenge, a complex source code analysis with targeted exploitation or just a live HTTP/network target to investigate. To do so, you have been given access to a list of tools (including a python interpreter & a bash shell) all inside an isolated environment.

# # How you work

# ## Personality

# Your default personality and tone is concise, direct, and friendly. You communicate efficiently, always keeping the user clearly informed about ongoing actions without unnecessary detail. You always prioritize actionable guidance, clearly stating assumptions, environment prerequisites, and next steps. Unless explicitly asked, you avoid excessively verbose explanations about your work.

# ## Environment

# You have access to `shell` and `python`, via tool calls, that are connected to a Ubuntu 22.04 container. You have access to the internet to download any files, install new linux packages, pip install new python packages, read and write files to your working directory, and if provided, read source code of challenges in a read-only directory.

# ## Sharing progress updates

# Since most of your tasks are longer and require critical thinking (i.e. requiring many tool calls, or a plan with multiple steps), you should provide progress updates back to the user at reasonable intervals. These updates should be structured as a concise sentence or two (no more than 8-10 words long) recapping progress so far in plain language: this update demonstrates your understanding of what needs to be done, progress so far (i.e. files explores, subtasks complete), and where you're going next.

# Before doing large chunks of work that may incur latency as experienced by the user (i.e. writing a new file), you should send a concise message to the user with an update indicating what you're about to do to ensure they know what you're spending time on.

# The messages you send before tool calls should describe what is immediately about to be done next in very concise language. If there was previous work done, this preamble message should also include a note about the work done so far to bring the user along.

# ## Task execution

# You are a coding agent. Please keep going until the query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. Autonomously resolve the query to the best of your ability, using the tools available to you, before coming back to the user. If you are not sure about file content or codebase structure pertaining to the task, use your tools to read files and gather the relevant information. Do NOT guess or make up an answer. Do NOT give give suggestions or improvements to the application unless asked.

# # Tool Guidelines

# ## Shell commands

# When using the shell, you must adhere to the following guidelines:

# - When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)
# - Read files in chunks with a max chunk size of 250 lines. Do not use python scripts to attempt to output larger chunks of a file. Command line output will be truncated after 10 kilobytes or 256 lines of output, regardless of the command used.

# ## Python tool

# This is the most powerful tool have access to, please use it often in the following senarios:

# - Developing a proof-of-concept exploit. Ensure to test exploits against a network targets if provided. Favor using python requests package for HTTP/HTTPS targets and pwntools for other socket connections
# - Testing and validating attacks. You have access already to standard libraries such as `numpy`, `pycryptodome`, and `scikit-learn` to help you iterate faster.
# """),
#     Message(role=MessageRole.USER, content="Hello, how are you? respond to me and also call bash tool with /bin/ls command in / directory and call bash tool with /bin/cat command in /etc/hosts in parallel. Then write a poem about world war two in 25 words respond in text."),
# ]

# print(model.generate(messages=messages, tools=[BashTool()]))
# import logging
# logging.basicConfig(level=logging.DEBUG)
# agent = MultiStepAgent(
#     model=model,
#     max_steps=5,
#     tools=[BashTool()]
# )

# output = agent.run(
#     "Hello, how are you? respond and call bash_tool with /bin/ls command in / directory and call bash_tool with /bin/cat command in /etc/hosts in parallel. Then write a poem about world war two in 25 words respond in text."
# )

# with open("response.json", "w") as f:
#     json.dump(asdict(agent.get_trace()), f, indent=2)

# import logging
# logging.basicConfig(level=logging.DEBUG)

# response = agent.run(
#     task="Hello, how are you? respond and call bash_tool with /bin/ls command in / directory and call bash_tool with /bin/cat command in /etc/hosts in parallel",
# )

# def process_finding(finding):
#     """Worker function to process a single finding"""
#     title = finding['title'].lower().replace(" ", "-").replace("/", "-")
    
#     try:
#         with ExploitAnalysisStudy(
#             code_path=pathlib.Path("services/ICC2023-AD-CTF/services/SeaOfHackerz/"),
#             vuln=finding['report'],
#             deploy_service=True,
#         ) as study:
#             result = study.run()

#         with open(f"exploit-report-{title}.md", "w") as f:
#             f.write(result)
            
#         return f"Successfully processed: {title}"
#     except Exception as e:
#         return f"Error processing {title}: {str(e)}"


# if __name__ == "__main__":
#     findings = json.load(open("findings.json"))
    
#     # Determine number of processes (use CPU count or limit to reasonable number)
#     num_processes = min(len(findings), multiprocessing.cpu_count(), 4)
    
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         results = pool.map(process_finding, findings)
    
#     # Print results
#     for result in results:
#         print(result)

finding = {
    "title": "Predictable ID/secret generation via insecure srand()/rand() seeding",
    "report": "## Overview\n\nThe binary generates both the public log ID (directory name) and the per-log secret using the C standard library PRNG (srand()/rand()) and seeds it with predictable values (current time, process id and an internal counter). This produces non-cryptographically-random identifiers and secrets that can be predicted or brute-forced by an attacker. Secrets used to authorize unredaction are therefore not securely generated.\n\n## Where it occurs\n\nFunction `generate_id` in the `logger` binary (decompiled) seeds the PRNG and formats the output into a string buffer:\n\n```c\nvoid generate_id(char *param_1,size_t param_2)\n{\n  time_t tVar3 = time(NULL);\n  uint uVar1 = getpid();\n  uVar1 = counter ^ (uint)tVar3 ^ uVar1;\n  counter = counter + 1;\n  srand(uVar1);\n  int iVar2 = rand();\n  snprintf(param_1, param_2, \"%012ld\", (long)iVar2 % 1000000000000);\n}\n```\n\nThis function is called twice in `save_log()` to create a public directory ID and the corresponding secret value written to the log file first line and shown to the user.\n\n## Vulnerability Details\n\n- The implementation uses the non-cryptographic PRNG `rand()` seeded via `srand()`.\n- The seed is computed as a simple XOR of `time(NULL)`, `getpid()`, and an in-memory `counter` value. These values are either predictable (current time) or have a small entropy space (PID, counter).\n- Using this predictable seeding, an attacker who knows or can guess the approximate time a log was created (or can observe a related event) can re-seed an offline PRNG to reproduce the same sequence of rand() outputs and thus determine both the public ID and the secret.\n\nBecause the secret is used as the authentication token for viewing the unredacted log, a predicted secret allows bypassing the authentication check and reconstructing sensitive log contents.\n\n## Impact\n\n- An attacker that can guess or determine the approximate generation time (which is often trivial for temporally-correlated actions) can predict the secret and hence perform `view_unredacted` to recover the original unredacted log content.\n- If an attacker can predict or enumerate directory IDs, they may be able to access or enumerate stored logs.\n- Secrets created with this weak randomness are also vulnerable to offline brute-force where the entropy is low (rand() typically returns 31 bits or less).\n\nThis substantially weakens the intended secrecy of logs and allows unauthorized access to sensitive stored data.\n\n## Steps to Reproduce / Exploit\n\n1. Observe or estimate the time (wall-clock time) when logs are created. This could be done by creating your own logs and comparing times or by making repeated requests until the service responds.\n2. Reconstruct the seed computation in a local program using candidate times, known PIDs (musl often sets predictable PIDs in containers), and counter guesses (0,1,...).\n3. Seed a local PRNG using srand(seed) and call rand() to compute the expected outputs; format them with the same snprintf rules.\n4. If the predicted secret matches the secret stored in a target's log's first line, request `view_unredacted` supplying the guessed secret and the public ID to retrieve full unredacted contents.\n\nA simple brute-force script can iterate over a small time window and counter values to recover secrets quickly.\n\n## Remediation\n\n- Do not use `srand()`/`rand()` for generating secrets, IDs, or tokens. Use a cryptographically secure random source such as /dev/urandom, getrandom(), or platform CSPRNG APIs (e.g., arc4random, getrandom, OpenSSL RAND_bytes).\n\nExample (POSIX) using getrandom()/getentropy():\n\n```c\nunsigned char buf[16];\nif (getrandom(buf, sizeof(buf), 0) == sizeof(buf)) {\n    // convert to hex or base64 for ID/secret\n}\n```\n\n- Increase entropy (longer secrets) and do not expose raw secrets unnecessarily.\n- If secrets must be human-readable, use a proper token generation library (cryptographic GUIDs or HMAC-based tokens) and enforce sufficient length (128-bit+).\n- Log the use of random tokens only in secure locations and avoid printing them unnecessarily.\n\nFixing the RNG usage will prevent offline prediction and brute-force of authentication tokens and protect log confidentiality.\n",
    "summary": "generate_id() uses srand()/rand() seeded with time, PID, and a simple counter; this produces predictable IDs and secrets that can be guessed or brute-forced.",
    "file_path": "logger",
    "confidence": "high",
    "severity": "medium",
    "exploitability": "high"
}
vuln = finding.report if isinstance(finding, VulnerabilityFinding) else finding.get('report')

from test_gp import get_latest_nop_flagids
flag_regex, service_ip, flagid = get_latest_nop_flagids("logger")

with ExploitAnalysisStudy(
    code_path=pathlib.Path("services/logger/"),
    model_cfg={
        "model_class": "OpenAIResponsesModel",
        "arguments": {
            "model_id": "gpt-5",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "requests_per_minute": 60,
            "reasoning": {"summary": "detailed", "effort": "high"}
        },
    },
    vuln=vuln,
    deploy_service=True,
    max_steps=100,
    flag_id=flagid,
    flag_regex=flag_regex,
    service_ip=service_ip,
) as study:
    result = study.run()

fn = f"exploit-report-{finding['title'].lower().replace(' ', '-')}.md"

with open(fn, "w") as f:
    f.write(result)