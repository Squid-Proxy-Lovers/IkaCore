import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from queueing import JobContext

from odin.constants import SeverityLevel

SEVERITY_LABEL_DEFINITIONS: Dict[SeverityLevel, Dict[str, str]] = {
    SeverityLevel.INFO:     {"name": "Severity: Info",     "color": "6e6e6e", "description": "Informational issue reported by Odin static analysis."},
    SeverityLevel.LOW:      {"name": "Severity: Low",      "color": "0e8a16", "description": "Low severity issue reported by Odin static analysis."},
    SeverityLevel.MEDIUM:   {"name": "Severity: Medium",   "color": "fbca04", "description": "Medium severity issue reported by Odin static analysis."},
    SeverityLevel.HIGH:     {"name": "Severity: High",     "color": "d93f0b", "description": "High severity issue reported by Odin static analysis."},
    SeverityLevel.CRITICAL: {"name": "Severity: Critical", "color": "b60205", "description": "Critical severity issue reported by Odin static analysis."},
}

BASE_URL: str = os.getenv("GITEA_URL", "https://git.uscg.win")
CLONE_ROOT = Path("/tmp/odin")


def _request(method: str, path: str, *, error: str, **kwargs) -> Optional[requests.Response]:
    url = urljoin(BASE_URL, path)
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        print(f"{error}: {exc}")
        return None

def search_repositories(query: str, limit: int = 10) -> List[Dict]:
    params = {"q": query, "limit": limit, "sort": "updated", "order": "desc"}
    response = _request("GET", "api/v1/repos/search", params=params, error="Error searching repositories")
    if not response:
        return []
    return response.json().get("data", [])

def list_labels(owner: str, repo: str) -> List[Dict]:
    response = _request(
        "GET",
        f"api/v1/repos/{owner}/{repo}/labels",
        error=f"Error fetching labels for {owner}/{repo}",
    )
    return response.json() if response else []


def _labels_by_name(owner: str, repo: str) -> Dict[str, Dict]:
    return {label.get("name"): label for label in list_labels(owner, repo)}


def ensure_label(
    owner: str,
    repo: str,
    *,
    name: str,
    color: str,
    description: str,
    existing: Optional[Dict[str, Dict]] = None,
) -> Optional[int]:
    color = color.lstrip("#")
    existing = existing if existing is not None else _labels_by_name(owner, repo)
    label = existing.get(name)
    if label:
        label_id = label.get("id")
        needs_update = (label.get("color", "").lower() != color.lower()) or (label.get("description") != description)
        if needs_update and label_id is not None:
            response = _request(
                "PATCH",
                f"api/v1/repos/{owner}/{repo}/labels/{label_id}",
                json={"color": color, "description": description},
                error=f"Warning: unable to update label {name}",
            )
            if response:
                label = response.json()
                existing[name] = label
                label_id = label.get("id")
        return label_id

    response = _request(
        "POST",
        f"api/v1/repos/{owner}/{repo}/labels",
        json={"name": name, "color": color, "description": description},
        error=f"Error creating label {name} for {owner}/{repo}",
    )
    if not response:
        return None
    label = response.json()
    existing[name] = label
    return label.get("id")

def ensure_severity_labels(owner: str, repo: str) -> Dict[SeverityLevel, int]:
    existing = _labels_by_name(owner, repo)
    ids: Dict[SeverityLevel, int] = {}
    for severity, definition in SEVERITY_LABEL_DEFINITIONS.items():
        label_id = ensure_label(owner, repo, existing=existing, **definition)
        if label_id is not None:
            ids[severity] = label_id
    return ids

def ensure_odin_label(owner: str, repo: str) -> Optional[int]:
    existing = _labels_by_name(owner, repo)
    return ensure_label(
        owner,
        repo,
        name="Odin",
        color="5319e7",
        description="Issues created by Odin static analysis tool",
        existing=existing,
    )

def ensure_exploit_label(owner: str, repo: str) -> Optional[int]:
    existing = _labels_by_name(owner, repo)
    return ensure_label(
        owner,
        repo,
        name="Exploit",
        color="e11d21",
        description="Indicates that a exploit is created for this issue",
        existing=existing,
    )

def ensure_overview_label(owner: str, repo: str) -> Optional[int]:
    existing = _labels_by_name(owner, repo)
    return ensure_label(
        owner,
        repo,
        name="Odin Overview",
        color="0e8a16",
        description="Issues created by Odin codebase overview analysis",
        existing=existing,
    )

def ensure_clientlib_label(owner: str, repo: str) -> Optional[int]:
    existing = _labels_by_name(owner, repo)
    return ensure_label(
        owner,
        repo,
        name="Odin ClientLib",
        color="0e8a16",
        description="Issues created by Odin client library analysis",
        existing=existing,
    )

def create_issue(owner: str, repo: str, title: str, body: str, label_ids: Optional[List[int]] = None) -> Dict:
    payload: Dict[str, object] = {"title": title, "body": body}
    if label_ids:
        payload["labels"] = [lid for lid in label_ids if lid is not None]
    response = _request(
        "POST",
        f"api/v1/repos/{owner}/{repo}/issues",
        json=payload,
        error=f"Error creating issue for {owner}/{repo}",
    )
    return response.json() if response else {}


def respond_to_issue(owner: str, repo: str, issue_id: int, body: str) -> Dict:
    response = _request(
        "POST",
        f"api/v1/repos/{owner}/{repo}/issues/{issue_id}/comments",
        json={"body": body},
        error=f"Error responding to issue {issue_id} for {owner}/{repo}",
    )
    return response.json() if response else {}


def update_issue_labels(
    owner: str,
    repo: str,
    issue_id: int,
    label_ids: Optional[List[int]] = None,
    *,
    replace: bool = False,
) -> Dict:
    labels = [lid for lid in label_ids or [] if lid is not None]
    response = _request(
        "PUT" if replace else "POST",
        f"api/v1/repos/{owner}/{repo}/issues/{issue_id}/labels",
        json={"labels": labels},
        error=f"Error updating labels for issue {issue_id} in {owner}/{repo}",
    )
    return response.json() if response else {}


def clone_to(owner: str, repo: str, destination: str) -> bool:
    subprocess.run(
        ["git", "clone", "--depth", "1", f"{BASE_URL}/{owner}/{repo}.git", destination], check=True
    )


def repo_clone_path(owner: str, repo: str) -> Path:
    return CLONE_ROOT / repo / owner


def clone_repository_job(ctx: JobContext) -> None:
    owner = ctx.payload.get("owner")
    repo = ctx.payload.get("repo")

    if not owner or not repo:
        raise ValueError("Clone jobs require 'owner' and 'repo'")

    destination = repo_clone_path(owner, repo)
    git_dir = destination / ".git"

    if git_dir.is_dir():
        print(f"Repository {owner}/{repo} already cloned at {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        shutil.rmtree(destination)

    try:
        clone_to(owner, repo, str(destination))
    except Exception:
        if destination.exists() and not git_dir.is_dir():
            shutil.rmtree(destination, ignore_errors=True)
        raise

def create_repo_from_exploit_template(
    name: str,
    dest_owner: Optional[str] = None,
    description: Optional[str] = None,
    include_all_branches: bool = False,
    default_branch: Optional[str] = None,
    topics: Optional[List[str]] = None,
) -> Dict:
    payload: Dict[str, object] = {
        "name": name,
        "git_content": True,
        "include_all_branches": include_all_branches,
        "private": True,
    }
    if dest_owner is not None:
        payload["owner"] = dest_owner
    if description is not None:
        payload["description"] = description
    if default_branch is not None:
        payload["default_branch"] = default_branch
    if topics:
        payload["topics"] = topics
    response = _request(
        "POST",
        "api/v1/repos/templates/exploit/generate",
        json=payload,
        error="Error generating repo from template templates/exploit",
    )
    return response.json() if response else {}