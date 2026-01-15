import json
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

from gitea import (clone_to, create_issue, create_repo_from_exploit_template,
                   ensure_clientlib_label, ensure_exploit_label,
                   ensure_odin_label, ensure_overview_label,
                   ensure_severity_labels, repo_clone_path, respond_to_issue,
                   update_issue_labels)
from gp import get_latest_nop_flagids
from queueing import JobContext
from template import get_static_analysis_issue_body

from odin.constants import VulnerabilityFinding
from odin.runtime import Environment
from odin.workflows import (ClientlibWorkflow, ExploitWorkflow,
                            OverviewWorkflow, StaticWorkflow)

CODE_BLOCK_RE = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)
FLAG_7N_OUT_RE = re.compile(r'^Flags\s*\((\d+)\):', re.MULTILINE)

def create_issue_helper(owner: str, repo: str, finding: VulnerabilityFinding, sev_labels: dict, odin_label_id: int) -> None:
    return create_issue(
        owner,
        repo,
        title=finding.title,
        body=get_static_analysis_issue_body(finding),
        label_ids=[odin_label_id, sev_labels.get(finding.severity)],
    )

def extract_code_block(text: Optional[str]) -> str:
    if not text:
        return ""
    match = CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else ""


def publish_clientlib(owner: str, report: Optional[str], prefix: str = None) -> None:
    code = extract_code_block(report)
    if not code:
        print(f"No client library code to push for {owner}")
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir, "exploit")
            clone_to(owner, "exploit", str(repo_path))

            git_dir = repo_path / ".git"
            if not git_dir.is_dir():
                raise RuntimeError(f"Exploit repository {owner}/exploit not available at {repo_path}")

            if prefix:
                clientlib_path = repo_path / f"{prefix}_clientlib.py"
            else:
                clientlib_path = repo_path / f"clientlib.py"

            clientlib_path.write_text(code.rstrip() + "\n", encoding="utf-8")

            git_cmd = ["git", "-C", str(repo_path)]
            subprocess.run([*git_cmd, "add", clientlib_path.name], check=True)

            message = f"Add client library analysis for {owner}"
            subprocess.run([*git_cmd, "commit", "-m", message], check=True)
            subprocess.run([*git_cmd, "push", "origin", "main"], check=True)
    except Exception as exc:
        print(f"Error pushing client library to exploit repo: {exc}")

def test_and_upload_exploit(owner: str, exploit: str) -> Tuple[bool, str, str]:
    code = extract_code_block(exploit)

    if not code:
        print(f"No exploit code to test for {owner}")
        return False, "No exploit code found", ""

    exp_repo_name = f"{owner}_odin_exploit_{uuid.uuid4().hex[:8]}"
    resp = create_repo_from_exploit_template(exp_repo_name, dest_owner=owner)
    html_url = resp.get("html_url") if resp else None
    print(f"Created exploit repo {resp}")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            clone_to(owner, exp_repo_name, str(repo_path))

            git_dir = repo_path / ".git"
            if not git_dir.is_dir():
                raise RuntimeError(f"Exploit repository {owner}/exploit not available at {repo_path}")

            exploit_path = repo_path / f"exploit.py"
            exploit_path.write_text(code.rstrip() + "\n", encoding="utf-8")

            git_cmd = ["git", "-C", str(repo_path)]
            subprocess.run([*git_cmd, "add", exploit_path.name], check=True)

            message = f"Add exploit for {owner}/{exp_repo_name}"
            subprocess.run([*git_cmd, "commit", "-m", message], check=True)
            subprocess.run([*git_cmd, "push", "origin", "main"], check=True)

            print(f"Testing exploit for {owner}/{exp_repo_name}")

            wd = os.getcwd()
            os.chdir(repo_path)
            result = subprocess.run(
                ["python3", str(repo_path / "7n.py"), "test"],
                capture_output=True, text=True
            )
            os.chdir(wd)

            if result.returncode == 0:
                print(f"Exploit for {owner}/{exp_repo_name} passed:\n{result.stdout}")
        
                matches = FLAG_7N_OUT_RE.findall(result.stdout)
                count = int(matches[-1]) if matches else 0
                print(f"Extracted {count} flags from exploit output")

                return count > 0, result.stdout, html_url
            else:
                print(f"Exploit for {owner}/{exp_repo_name} failed with return code {result.returncode}:\n{result.stderr}")
                return False, result.stderr, html_url
    except Exception as exc:
        print(f"Error testing/pushing exploit to exploit repo: {exc}")
        return False, str(exc), html_url

def static_analysis(ctx: JobContext) -> None:
    owner = ctx.payload.get("owner")
    repo = ctx.payload.get("repo")

    if not owner or not repo:
        raise ValueError("Static analysis jobs require 'owner' and 'repo'")

    print(f"Running static analysis for {owner}/{repo}")

    repo_dir = repo_clone_path(owner, repo)

    if not (repo_dir / ".git").is_dir():
        raise RuntimeError(f"Repository {owner}/{repo} not available at {repo_dir}")

    sev_labels = ensure_severity_labels(owner, repo)
    odin_label_id = ensure_odin_label(owner, repo)

    issues_created = []
    def finding_callback(finding: VulnerabilityFinding) -> None:
        issue = create_issue_helper(owner, repo, finding, sev_labels, odin_label_id)
        issues_created.append(issue.get("number"))

    with Environment(code_path=repo_dir) as env:
        workflow = StaticWorkflow(
            code_path=repo_dir,
            model_cfg=ctx.payload.get("model_cfg", {
                "model_class": "OpenAIResponsesModel",
                "arguments": {
                    "model_id": "gpt-5",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "requests_per_minute": 60,
                    "reasoning": {"summary": "detailed", "effort": "high"}
                },
            }),
            max_steps=150,
            finding_callback=finding_callback if ctx.payload.get("add_findings") else None,
            trace_callback=ctx.update_trace,
        )
        findings = workflow.run(env)
        print(f"Static analysis found {len(findings)} issues")

    issues_created = list(filter(None, issues_created))
    ctx.set_output({"findings": [asdict(f) for f in findings], "issues_created": issues_created})

def combiner_analysis(ctx: JobContext) -> None:
    owner = ctx.payload.get("owner")
    repo = ctx.payload.get("repo")

    findings = ctx.payload.get("findings", [])

    if not findings:
        findings_from_jobids = ctx.payload.get("findings_from_jobids", [])
        for job_id in findings_from_jobids:
            job_output = ctx.get_output(job_id)
            if job_output and "findings" in job_output:
                findings.extend(job_output["findings"])

    if not owner or not repo:
        raise ValueError("Combiner analysis jobs require 'owner' and 'repo'")

    if not findings:
        raise ValueError("Combiner analysis jobs require 'findings'")

    print(f"Running combiner analysis for {owner}/{repo} with {len(findings)} findings")

    repo_dir = repo_clone_path(owner, repo)

    if not (repo_dir / ".git").is_dir():
        raise RuntimeError(f"Repository {owner}/{repo} not available at {repo_dir}")

    sev_labels = ensure_severity_labels(owner, repo)
    odin_label_id = ensure_odin_label(owner, repo)

    issues_created = []
    def finding_callback(finding: VulnerabilityFinding) -> None:
        issue = create_issue_helper(owner, repo, finding, sev_labels, odin_label_id)
        issues_created.append(issue.get("number"))

        print(f"Created issue {json.dumps(issue)} for finding {finding.title}")
        if issue.get("number") and ctx.payload.get("enqueue_exploit_analysis"):
            print(f"Enqueuing exploit analysis for issue #{issue['number']}")
            ctx.enqueue_child(
                job_type="preprocess-exploit-analysis",
                payload={
                    "owner": owner,
                    "repo": repo,
                    "issue_number": issue["number"],
                    "finding": asdict(finding),
                },
                depends_on=[],
                priority=200,
            )

    with Environment(code_path=repo_dir) as env:
        workflow = StaticWorkflow(
            code_path=repo_dir,
            model_cfg=ctx.payload.get("model_cfg", {
                "model_class": "OpenAIResponsesModel",
                "arguments": {
                    "model_id": "gpt-5",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "requests_per_minute": 60,
                    "reasoning": {"summary": "detailed", "effort": "high"}
                },
            }),
            max_steps=100,
            finding_callback=finding_callback if ctx.payload.get("add_findings") else None,
            additional_task_info=f"The following findings were discovered in a previous analysis. You must deduplicate findings and create one enriched report with the combined results and information from each report and finding. You may choose to exclude findings that aren't correct or do not have impact.\nPrevious findings:\n{json.dumps(findings, indent=2)}\n\nIMPORTANT: Remember to use `add_vulnerability_finding` to record all findings and only have a summary in your final message.",
            trace_callback=ctx.update_trace,
        )
        new_findings = workflow.run(env)
        print(f"Combiner analysis found {len(new_findings)} new issues")

    issues_created = list(filter(None, issues_created))
    ctx.set_output({"findings": [asdict(f) for f in new_findings], "issues_created": issues_created})

def preprocess_exploit_analysis(ctx: JobContext) -> None:
    owner = ctx.payload.get("owner")
    repo = ctx.payload.get("repo")

    if not owner or not repo:
        raise ValueError("Exploit analysis jobs require 'owner' and 'repo'")

    print(f"Preprocessing exploit analysis for {owner}/{repo}")
    try:
        flag_regex, service_ip, flagids = get_latest_nop_flagids(owner)
    except Exception as e:
        print(f"Error fetching flagids for {owner}: {e}")
        flag_regex, service_ip, flagids = None, None, [None]

    jobs_created = []
    for flagid in flagids:
        print(f"Enqueuing exploit analysis for flagid {flagid}")
        jobid = ctx.enqueue_child(
            job_type="exploit-analysis",
            payload={
                "owner": owner,
                "repo": repo,
                "issue_number": ctx.payload.get("issue_number"),
                "finding": ctx.payload.get("finding"),
                "flag_regex": flag_regex,
                "service_ip": service_ip,
                "flagid": flagid,
                "model_cfg": ctx.payload.get("model_cfg"),
            },
            depends_on=[],
            priority=200,
        )
        jobs_created.append(jobid)
    ctx.set_output({"jobs_created": jobs_created})

def exploit_analysis(ctx: JobContext) -> None:
    owner = ctx.payload.get("owner")
    repo = ctx.payload.get("repo")

    finding = ctx.payload.get("finding")
    issue_number = ctx.payload.get("issue_number")

    if not issue_number:
        raise ValueError("Exploit analysis jobs require 'issue_number'")

    if not owner or not repo:
        raise ValueError("Exploit analysis jobs require 'owner' and 'repo'")
    
    if not finding:
        raise ValueError("Exploit analysis jobs require 'finding'")

    print(f"Running exploit analysis for {owner}/{repo} on finding {finding.get('title') if finding else 'N/A'} linked to issue #{issue_number}")

    repo_dir = repo_clone_path(owner, repo)
    if not (repo_dir / ".git").is_dir():
        raise RuntimeError(f"Repository {owner}/{repo} not available at {repo_dir}")

    exploit_label_id = ensure_exploit_label(owner, repo)
    vuln = finding.report if isinstance(finding, VulnerabilityFinding) else finding.get('report')

    if not vuln:
        raise ValueError("Finding must have a report")
        
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_repo_dir = Path(temp_dir) / repo
        subprocess.run(["cp", "-r", str(repo_dir) + "/.", str(temp_repo_dir)], check=True)

        with Environment(code_path=temp_repo_dir, deploy_service=True) as env:
            workflow = ExploitWorkflow(
                code_path=temp_repo_dir,
                model_cfg=ctx.payload.get("model_cfg", {
                    "model_class": "OpenAIResponsesModel",
                    "arguments": {
                        "model_id": "gpt-5",
                        "api_key": os.environ.get("OPENAI_API_KEY"),
                        "requests_per_minute": 60,
                        "reasoning": {"summary": "detailed", "effort": "high"}
                    },
                }),
                vuln=vuln,
                max_steps=150,
                flag_id=ctx.payload.get("flagid"),
                flag_regex=ctx.payload.get("flag_regex"),
                service_ip=ctx.payload.get("service_ip"),
                trace_callback=ctx.update_trace,
            )
            exploit = workflow.run(env)

            print(f"Exploit analysis completed, adding to issue #{issue_number}")

            if exploit:
                update_issue_labels(
                    owner,
                    repo,
                    issue_id=issue_number,
                    label_ids=[exploit_label_id],
                    replace=False,
                )

            exploit_success, n7_output, exploit_url = test_and_upload_exploit(owner, exploit) if exploit else False

            additional_info = ""
            if exploit_success:
                additional_info = "\n\nThe exploit was successfully tested and found to retrieve flags."
            if n7_output:
                additional_info += f"\nOutput from testing the exploit:\n```\n{n7_output}\n```"
            if exploit_url:
                additional_info += f"\nThe exploit code has been pushed to a repository: [{exploit_url}]({exploit_url})"

            respond_to_issue(
                owner,
                repo,
                issue_id=issue_number,
                body=(exploit or "Exploit analysis completed, but no exploit was found.") + additional_info,
            )

            ctx.set_output({"exploit_added_to_issue": bool(exploit), "exploit_success": exploit_success, "exploit_test_output": n7_output, "exploit_repo_url": exploit_url})

def overview_analysis(ctx: JobContext) -> None:
    owner = ctx.payload.get("owner")
    repo = ctx.payload.get("repo")

    if not owner or not repo:
        raise ValueError("Overview analysis jobs require 'owner' and 'repo'")

    print(f"Running overview analysis for {owner}/{repo}")

    repo_dir = repo_clone_path(owner, repo)

    if not (repo_dir / ".git").is_dir():
        raise RuntimeError(f"Repository {owner}/{repo} not available at {repo_dir}")

    overview_label_id = ensure_overview_label(owner, repo)
    odin_label_id = ensure_odin_label(owner, repo)

    with Environment(code_path=repo_dir, deploy_service=True) as env:
        workflow = OverviewWorkflow(
            code_path=repo_dir,
            model_cfg=ctx.payload.get("model_cfg", {
                "model_class": "OpenAIResponsesModel",
                "arguments": {
                    "model_id": "gpt-5",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "requests_per_minute": 60,
                    "reasoning": {"summary": "detailed", "effort": "high"}
                },
            }),
            max_steps=100,
            trace_callback=ctx.update_trace,
        )
        report = workflow.run(env)

        print(f"Overview analysis completed, creating issue in {owner}/{repo}")
        issue = create_issue(
            owner,
            repo,
            title="Codebase Overview Report",
            body=report or "Overview analysis completed, but no report was generated.",
            label_ids=[odin_label_id, overview_label_id],
        )
        print(f"Created overview issue {json.dumps(issue)}")
        ctx.set_output({"issue_created": issue.get("number")})

def clientlib_analysis(ctx: JobContext) -> None:
    owner = ctx.payload.get("owner")
    repo = ctx.payload.get("repo")

    if not owner or not repo:
        raise ValueError("Client library analysis jobs require 'owner' and 'repo'")

    print(f"Running client library analysis for {owner}/{repo}")

    repo_dir = repo_clone_path(owner, repo)

    if not (repo_dir / ".git").is_dir():
        raise RuntimeError(f"Repository {owner}/{repo} not available at {repo_dir}")

    clientlib_label_id = ensure_clientlib_label(owner, repo)
    odin_label_id = ensure_odin_label(owner, repo)

    with Environment(code_path=repo_dir, deploy_service=True) as env:
        workflow = ClientlibWorkflow(
            code_path=repo_dir,
            model_cfg=ctx.payload.get("model_cfg", {
                "model_class": "OpenAIResponsesModel",
                "arguments": {
                    "model_id": "gpt-5",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "requests_per_minute": 60,
                    "reasoning": {"summary": "detailed", "effort": "high"}
                },
            }),
            max_steps=100,
            trace_callback=ctx.update_trace,
        )
        report = workflow.run(env)

        print(f"Client library analysis completed, creating issue in {owner}/{repo}")
        issue = create_issue(
            owner,
            repo,
            title="Client Library Analysis Report",
            body=report or "Client library analysis completed, but no report was generated.",
            label_ids=[odin_label_id, clientlib_label_id],
        )
        print(f"Created client library issue {json.dumps(issue)}")

        publish_clientlib("templates", report, prefix=owner)
        publish_clientlib(owner, report)

        ctx.set_output({"issue_created": issue.get("number")})

def trigger_analysis(ctx: JobContext) -> None:
    git_clone_job_id = ctx.enqueue_child(
        job_type="git-clone",
        payload={
            "owner": ctx.payload.get("owner"),
            "repo": ctx.payload.get("repo"),
        },
        priority=1
    )

    overview_job_id = ctx.enqueue_child(
        job_type="overview-analysis",
        payload={
            "owner": ctx.payload.get("owner"),
            "repo": ctx.payload.get("repo"),
        },
        priority=10,
        depends_on=[git_clone_job_id],
    )

    clientlib_job_id = ctx.enqueue_child(
        job_type="clientlib-analysis",
        payload={
            "owner": ctx.payload.get("owner"),
            "repo": ctx.payload.get("repo"),
        },
        priority=5,
        depends_on=[git_clone_job_id],
    )

    static_analysis_jobs_ids = [
        ctx.enqueue_child(
            job_type="static-analysis",
            payload={
                "owner": ctx.payload.get("owner"),
                "repo": ctx.payload.get("repo"),
                "add_findings": False,
            },
            depends_on=[git_clone_job_id],
        )
        for _ in range(2)
    ]

    combiner_job_id = ctx.enqueue_child(
        job_type="combiner-analysis",
        payload={
            "owner": ctx.payload.get("owner"),
            "repo": ctx.payload.get("repo"),
            "findings_from_jobids": static_analysis_jobs_ids,
            "add_findings": True,
            "enqueue_exploit_analysis": True,
            "exploit_analysis_model_deps": [
                overview_job_id,
                clientlib_job_id,
            ]
        },
        depends_on=static_analysis_jobs_ids + [git_clone_job_id],
    )

    ctx.set_output({"triggered_jobs": [static_analysis_jobs_ids, combiner_job_id, overview_job_id, clientlib_job_id, git_clone_job_id]})
