import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class Vulnerability:
    name: str
    description: str
    path: str
    line_start: int
    line_end: int

@dataclass
class Service:
    name: str
    service_path: str
    language: List[str]
    vulnerabilities: List[Vulnerability] = None

@dataclass
class ServiceRepo:
    name: str
    url: str
    author: str
    # tags: List[str] TODO: maybe add later?
    services: List[Service] = None

repos = [
    # TODO add more tests cases?
    # "C4T-BuT-S4D/blitz-15-03-2020",
    # "C4T-BuT-S4D/innoctf-final-10-05-2020",
    # "C4T-BuT-S4D/training-15-09-19",
    # "C4T-BuT-S4D/training-05-10-19",
    # "C4T-BuT-S4D/innoctf-teazer-01-03-2020",
    # "C4T-BuT-S4D/training-17-11-19",
    # "C4T-BuT-S4D/training-27-10-19",
    # "C4T-BuT-S4D/training-02-02-20",
    # "C4T-BuT-S4D/blitz-14-06-2020",
    # "C4T-BuT-S4D/training-18-10-20",
    # "C4T-BuT-S4D/ctfcup-2022-stage2",
    # "C4T-BuT-S4D/stay-home-ctf-2022",
    # "C4T-BuT-S4D/stay-home-ctf-2020",
    # "C4T-BuT-S4D/ctfcup-2022-stage3-part1",
    # "C4T-BuT-S4D/bricsctf-2023-stage2",
    # "C4T-BuT-S4D/ctfcup-2023-ad",
    # "C4T-BuT-S4D/goldctf-2024",
    # "C4T-BuT-S4D/bricsctf-2024-finals",

    ServiceRepo(
        name="ctfcup-2024-ad",
        url="https://github.com/C4T-BuT-S4D/ctfcup-2024-ad",
        author="C4T-BuT-S4D",
        services=[
            Service(
                name="crypter",
                service_path="services/crypter",
                language=["C++"],
                vulnerabilities=[
                    Vulnerability(
                        name="Brute-forceable Seed for Cryptographic Function",
                        description="The seed for the cryptographic function is a uint8_t value that is brute-forceable.",
                        path="services/crypter/src/crypto.cc",
                        line_start=9,
                        line_end=11,
                    )
                ],
            ),
            Service(
                name="docs",
                service_path="services/docs",
                language=["Python", "Go"],
                vulnerabilities=[
                    Vulnerability(
                        name='Parameter Injection in Search',
                        description='The `search_docs` function at `/api/documents` endpoint is vulnerable to parameter injection.',
                        path='services/docs/api/src/app/api.py',
                        line_start=169,
                        line_end=183
                    ),
                ],
            ),
            Service(
                name="ark",
                service_path="services/ark",
                language=["Rust"],
                vulnerabilities=[
                    Vulnerability(
                        name="Arbitrary File Write via Path Traversal in copy_file",
                        description=(
                                        "The `copy_file` function's check using `canonicalize(new_path).is_ok()` can be bypassed with malformed "
                                        "paths that cause `canonicalize` to return `Err`, allowing `tokio::fs::copy` to write the source file to an arbitrary location."
                                    ),
                        path="services/ark/src/main.rs",
                        line_start=212,
                        line_end=260
                    ),
                ],
            ),
        ],
    ),

    ServiceRepo(
        name="CybersecNatLab/ICC2023-AD-CTF",
        url="https://github.com/CybersecNatLab/ICC2023-AD-CTF",
        author="CybersecNatLab",
        services=[
            Service(
                name="SeaOfHackerz",
                service_path="services/SeaOfHackerz",
                language=["Python"],
                vulnerabilities=[
                    Vulnerability(
                        name="SQL injection in /login endpoint",
                        description=(
                            "The `login` endpoint uses a query which is not sanitized"
                        ),
                        path="services/SeaOfHackerz/backend/app.py",
                        line_start=792,
                        line_end=793
                    )
                ]
            ),
            Service(
                name="SeaOfHackerz",
                service_path="services/SeaOfHackerz",
                language=["Python"],
                vulnerabilities=[
                    Vulnerability(
                        name="Cookie forgery in /login endpoint",
                        description=(
                            "The `login` endpoint uses a random.seed on the userid to generate a cookie, which is predictable. "
                        ),
                        path="services/SeaOfHackerz/backend/app.py",
                        line_start=799,
                        line_end=800
                    )
                ]
            ),
            Service(
                name="SeaOfHackerz",
                service_path="services/SeaOfHackerz",
                language=["Python"],
                vulnerabilities=[
                    Vulnerability(
                        name="The attack endpoint uses an LCG PRNG",
                        description=(
                            "The `cryptographically_secure_prng` is not cryptographically secure."
                        ),
                        path="services/SeaOfHackerz/backend/app.py",
                        line_start=141,
                        line_end=154
                    )
                ]
            )
        ]
    )
]

def main(args):
    for repo in repos:
        repo_name = repo.name

        if os.path.exists(repo_name):
            subprocess.run(["rm", "-rf", repo_name])
            print(f"Deleted directory: {repo_name}")

        subprocess.run(["git", "clone", repo.url, "--depth", "1"])
        print(f"Cloned repository: {repo.url}")

    for repo in repos:
        for service in repo.services:
            service.service_path = os.path.join(repo.name, service.service_path)

            for vulnerability in service.vulnerabilities:
                vulnerability.path = os.path.join(repo.name, vulnerability.path)

    serializable_data = {
        repo.name: [asdict(service) for service in repo.services] if repo.services else None
        for repo in repos
    }

    with open("services.json", "w") as f:
        json.dump(serializable_data, f, indent=4)
        print(f"Dumped services to services.json")

if __name__ == "__main__":
    main(sys.argv)