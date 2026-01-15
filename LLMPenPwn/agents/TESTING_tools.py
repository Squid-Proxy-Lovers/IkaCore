#!/usr/bin/env python3
"""
Tools for penetration testing agents.
"""

import json
import base64
import os
import re
import sqlite3
import socket
import subprocess
from datetime import datetime
from pathlib import Path
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
import logging

# Add parent directory to path to import adapter
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import adapter using importlib to avoid name collision with this module
import importlib.util
_adapter_file = _parent_dir / "tools" / "adapter.py"
_adapter_spec = importlib.util.spec_from_file_location("adapter_module", _adapter_file)
_adapter = importlib.util.module_from_spec(_adapter_spec)
_adapter_spec.loader.exec_module(_adapter)
tool = _adapter.tool
function_to_odin_tool = _adapter.function_to_odin_tool

tool_logger = logging.getLogger("tools")


_DENY_CMD_PATTERNS = [
    r"\brm\b",
    r"\breboot\b",
    r"\bshutdown\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bkill\b\s+-9\b",
    r"\bpoweroff\b",
    r"\binit\b\s+0\b",
    r"\bhalt\b",
]


def _truncate_output(text: str, max_output_kb: int) -> str:
    max_bytes = max(1, int(max_output_kb) * 1024)
    data = text.encode("utf-8", errors="replace")
    if len(data) <= max_bytes:
        return text
    truncated = data[:max_bytes].decode("utf-8", errors="replace")
    return truncated + "\n[truncated]\n"


def _is_command_denied(command: str) -> bool:
    cmd = command.lower()
    for pat in _DENY_CMD_PATTERNS:
        if re.search(pat, cmd):
            return True
    return False


@tool
def nmap_scan(target: str, scan_type: str = "default") -> str:
    """
    Runs nmap against a host/subnet with preset scan profiles.
    Uses -Pn (skip ping) since ICMP is often blocked on VPN networks.
    Uses TCP connect scans (-sT) which work reliably without root.

    Args:
        target (str): IP/hostname/CIDR to scan.
        scan_type (str): quick|default|full|stealth|udp.
    """
    # Safety: conservative timing, limited retries, shorter host timeout for efficiency
    safe_flags = ["-T3", "--max-retries", "1", "--host-timeout", "30s"]
    
    # Determine if this is a subnet scan
    is_subnet = "/" in target
    
    if is_subnet:
        # For subnet scans: use fewer ports, faster timeouts, only show open ports
        scan_commands = {
            "quick": ["nmap", *safe_flags, "-Pn", "-sT", "-p", "22,80,443,445,3389", "--open", target],  # Fast discovery: 5 common ports, only open
            "default": ["nmap", *safe_flags, "-Pn", "-sT", "-p", "22,80,443,445,3389,21,25,53,135,139,1433,3306,5432,8080,8443", "--open", "-sV", target],  # Common ports with version detection
            "full": ["nmap", *safe_flags, "-Pn", "-sT", "--top-ports", "1000", "-sV", "-sC", target],  # Top 1000 ports with scripts
            "stealth": ["nmap", *safe_flags, "-Pn", "-sS", "-sV", target],  # SYN scan (requires root)
            "udp": ["nmap", *safe_flags, "-Pn", "-sU", "-sV", target],  # UDP scan
        }
        timeout_seconds = 900  # 15 minutes for subnet scans
    else:
        # For single host scans: can be more thorough
        scan_commands = {
            "quick": ["nmap", *safe_flags, "-Pn", "-sT", "--top-ports", "100", "--open", target],  # Top 100 ports, only open
            "default": ["nmap", *safe_flags, "-Pn", "-sT", "-sV", "--top-ports", "1000", target],  # Top 1000 ports with version
            "full": ["nmap", *safe_flags, "-Pn", "-sT", "-sV", "-sC", "-p-", target],  # All ports with scripts
            "stealth": ["nmap", *safe_flags, "-Pn", "-sS", "-sV", target],  # SYN scan (requires root)
            "udp": ["nmap", *safe_flags, "-Pn", "-sU", "-sV", target],  # UDP scan
        }
        timeout_seconds = 600  # 10 minutes for single host full scans
    
    cmd = scan_commands.get(scan_type, scan_commands["default"])
    
    try:
        tool_logger.debug(f"nmap_scan: target={target}, scan_type={scan_type}, cmd={' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        tool_logger.debug(f"nmap_scan: completed, returncode={result.returncode}")
        output = result.stdout + result.stderr
        # If scan found hosts but timed out on some, still return results
        if result.returncode != 0 and "Nmap scan report" in output:
            tool_logger.warning(f"nmap_scan: scan completed with warnings but has results")
        return output
    except subprocess.TimeoutExpired:
        return "Error: nmap scan timed out. Try scanning individual hosts or using a smaller port range."
    except FileNotFoundError:
        return "Error: nmap not found. Please install nmap."
    except Exception as e:
        return f"Error running nmap: {str(e)}"


@tool
def execute_bash(command: str, timeout_sec: int = 60, max_output_kb: int = 256) -> str:
    """
    Executes a bash command and returns stdout+stderr.

    Args:
        command (str): Command to execute on host.
        timeout_sec (int): Timeout in seconds (default: 60).
        max_output_kb (int): Max output to return in KB (default: 256).
    """
    if _is_command_denied(command):
        tool_logger.warning(f"execute_bash: Command blocked - {command[:100]}")
        return "Error: Command blocked by safety policy."
    try:
        tool_logger.debug(f"execute_bash: command={command[:200]}, timeout={timeout_sec}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
        )
        tool_logger.debug(f"execute_bash: completed, returncode={result.returncode}")
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return _truncate_output(output, max_output_kb=max_output_kb)
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout_sec} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def url_join(base: str, path: str) -> str:
    """
    Joins a base URL with a path safely.

    Args:
        base (str): Base URL, e.g., http://10.0.0.1:8080
        path (str): Path, e.g., /admin
    """
    return urllib.parse.urljoin(base.rstrip("/") + "/", path.lstrip("/"))


@tool
def url_encode(value: str) -> str:
    """
    URL-encodes a value.

    Args:
        value (str): String to encode.
    """
    return urllib.parse.quote(value, safe="")


@tool
def url_decode(value: str) -> str:
    """
    URL-decodes a value.

    Args:
        value (str): Encoded string to decode.
    """
    return urllib.parse.unquote(value)


@tool
def base64_encode(value: str) -> str:
    """
    Base64-encodes a UTF-8 string.

    Args:
        value (str): String to encode.
    """
    return base64.b64encode(value.encode("utf-8")).decode("utf-8")


@tool
def base64_decode(value: str) -> str:
    """
    Base64-decodes a string into UTF-8 if possible.

    Args:
        value (str): Base64-encoded string.
    """
    try:
        return base64.b64decode(value.encode("utf-8"), validate=True).decode("utf-8", errors="replace")
    except Exception as e:
        return f"Error: invalid base64 - {e}"


@tool
def regex_extract(text: str, pattern: str) -> str:
    """
    Extracts regex matches from text.

    Args:
        text (str): Input text to search.
        pattern (str): Regex pattern.
    """
    try:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        return json.dumps(matches, indent=2)
    except Exception as e:
        return f"Error: regex failed - {e}"


def _safe_json_loads(text: str, fallback):
    try:
        data = json.loads(text)
        return data
    except Exception:
        return fallback


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


@tool
def poll_notifications(target_id: str, limit: int = 20, db_path: str = "pentest_memory.db") -> str:
    """
    Polls notifications for a target (by IP or target_id) from SQLite.

    Args:
        target_id (str): Target identifier or IP address.
        limit (int): Max notifications to return (default: 20).
        db_path (str): Path to SQLite database.
    """
    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT note_id, target_ip, message, created_at FROM notifications WHERE target_ip = ? ORDER BY note_id DESC LIMIT ?",
        (target_id, int(limit)),
    )
    rows = [
        {"notification_id": r[0], "target_ip": r[1], "message": r[2], "created_at": r[3]}
        for r in cur.fetchall()
    ]
    conn.close()
    return json.dumps(rows, indent=2)


@tool
def ack_notification(notification_id: int, db_path: str = "pentest_memory.db") -> str:
    """
    Acknowledges a notification by deleting it from SQLite.

    Args:
        notification_id (int): Notification ID to delete.
        db_path (str): Path to SQLite database.
    """
    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    cur = conn.cursor()
    cur.execute("DELETE FROM notifications WHERE note_id = ?", (int(notification_id),))
    conn.commit()
    conn.close()
    return "Notification acknowledged."


@tool
def broadcast_notification(message: str, target_ids_json: str = "[]", db_path: str = "pentest_memory.db") -> str:
    """
    Broadcasts a notification to specific targets, or all sleeping targets if none provided.

    Args:
        message (str): Message to send.
        target_ids_json (str): JSON array of target IPs/IDs to notify (default: []).
        db_path (str): Path to SQLite database.
    """
    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    now = datetime.utcnow().isoformat()
    target_ids = _safe_json_loads(target_ids_json, [])
    cur = conn.cursor()
    sent = 0
    if isinstance(target_ids, list) and target_ids:
        for tid in target_ids:
            if not isinstance(tid, str) or not tid.strip():
                continue
            cur.execute(
                "INSERT INTO notifications (target_ip, message, created_at) VALUES (?, ?, ?)",
                (tid.strip(), message, now),
            )
            sent += 1
        conn.commit()
    else:
        sent = _wake_sleeping_targets(conn, message)
    conn.close()
    return f"Notification sent to {sent} targets."


@tool
def list_vulnerabilities(risk_filter: str = "", ip_filter: str = "", limit: int = 200, db_path: str = "pentest_memory.db") -> str:
    """
    Lists vulnerability records from SQLite with optional filters.

    Args:
        risk_filter (str): Optional risk filter: CRITICAL|HIGH|MEDIUM|LOW|INFORMATIONAL.
        ip_filter (str): Optional substring filter on assets (e.g., 10.0.1.14).
        limit (int): Max rows (default: 200).
        db_path (str): Path to SQLite database.
    """
    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    cur = conn.cursor()
    risk = risk_filter.strip().upper()
    ip = ip_filter.strip()
    query = "SELECT vuln_id, bug_name, risk, impact, sophistication, assets, created_at FROM vulnerabilities"
    where: list[str] = []
    params: list[object] = []
    if risk:
        where.append("risk = ?")
        params.append(risk)
    if ip:
        where.append("assets LIKE ?")
        params.append(f"%{ip}%")
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY vuln_id DESC LIMIT ?"
    params.append(int(limit))
    cur.execute(query, tuple(params))
    rows = [
        {
            "vuln_id": r[0],
            "bug_name": r[1],
            "risk": r[2],
            "impact": r[3],
            "sophistication": r[4],
            "assets": r[5],
            "created_at": r[6],
        }
        for r in cur.fetchall()
    ]
    conn.close()
    return json.dumps(rows, indent=2)


@tool
def list_interesting_data(type_filter: str = "", ip_filter: str = "", limit: int = 200, db_path: str = "pentest_memory.db") -> str:
    """
    Lists interesting_data rows from SQLite with optional filters.

    Args:
        type_filter (str): Optional category filter: PASSWORD|EMAIL|USERNAME|HASH|NOTE.
        ip_filter (str): Optional target_ip filter.
        limit (int): Max rows (default: 200).
        db_path (str): Path to SQLite database.
    """
    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    cur = conn.cursor()
    cat = type_filter.strip().upper()
    ip = ip_filter.strip()
    query = "SELECT data_id, target_ip, category, service, username, password, hash, email, notes, timestamp FROM interesting_data"
    where: list[str] = []
    params: list[object] = []
    if cat:
        where.append("category = ?")
        params.append(cat)
    if ip:
        where.append("target_ip = ?")
        params.append(ip)
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY data_id DESC LIMIT ?"
    params.append(int(limit))
    cur.execute(query, tuple(params))
    rows = [
        {
            "data_id": r[0],
            "target_ip": r[1],
            "category": r[2],
            "service": r[3],
            "username": r[4],
            "password": r[5],
            "hash": r[6],
            "email": r[7],
            "notes": r[8],
            "timestamp": r[9],
        }
        for r in cur.fetchall()
    ]
    conn.close()
    return json.dumps(rows, indent=2)


@tool
def get_latest_activity(limit: int = 200, db_path: str = "pentest_memory.db") -> str:
    """
    Returns a combined feed of vulnerabilities, interesting data, and notifications.

    Args:
        limit (int): Max rows per category (default: 200).
        db_path (str): Path to SQLite database.
    """
    vulns = _safe_json_loads(list_vulnerabilities(limit=limit, db_path=db_path), [])
    data = _safe_json_loads(list_interesting_data(limit=limit, db_path=db_path), [])
    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT note_id, target_ip, message, created_at FROM notifications ORDER BY note_id DESC LIMIT ?",
        (int(limit),),
    )
    notes = [
        {"notification_id": r[0], "target_ip": r[1], "message": r[2], "created_at": r[3]}
        for r in cur.fetchall()
    ]
    conn.close()
    return json.dumps({"vulnerabilities": vulns, "interesting_data": data, "notifications": notes}, indent=2)


@tool
def delegate_to_subagent(agent_type: str, target_id: str, task_json: str, db_path: str = "pentest_memory.db") -> str:
    """
    Enqueues a task for a specialized sub-agent to pick up and execute.

    Args:
        agent_type (str): One of WEB|WINDOWS|MISC|CODE_REVIEW.
        target_id (str): Target ID (e.g., target1) or IP.
        task_json (str): JSON payload with schema: {\"goal\": str, \"scope\": {...}, \"allowed_tools\": [...], \"safety_limits\": {...}}.
        db_path (str): Path to SQLite database.
    """
    at = agent_type.strip().upper()
    if at not in {"WEB", "WINDOWS", "MISC", "CODE_REVIEW"}:
        return "Error: agent_type must be one of WEB|WINDOWS|MISC|CODE_REVIEW"
    payload = _safe_json_loads(task_json, None)
    if not isinstance(payload, dict) or not isinstance(payload.get("goal"), str) or not payload.get("goal", "").strip():
        return "Error: task_json must be JSON object with at least a non-empty 'goal' string"

    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    _ensure_agent_tasks_table(conn)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO agent_tasks (agent_type, target_id, task_json, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (at, target_id, json.dumps(payload), "PENDING", now, now),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()
    return f"Task enqueued: id={task_id} agent_type={at} target_id={target_id}"


@tool
def credential_reuse_plan(data_id: int, targets_json_path: str = "network_scan.json", db_path: str = "pentest_memory.db") -> str:
    """
    Generates a safe credential reuse test plan for a piece of interesting data.

    Args:
        data_id (int): interesting_data.data_id row to base plan on.
        targets_json_path (str): Path to network_scan.json.
        db_path (str): Path to SQLite database.
    """
    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT data_id, target_ip, category, service, username, password, hash, email, notes, timestamp FROM interesting_data WHERE data_id = ?",
        (int(data_id),),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return "Error: data_id not found"
    item = {
        "data_id": row[0],
        "target_ip": row[1],
        "category": row[2],
        "service": row[3],
        "username": row[4],
        "password": row[5],
        "hash": row[6],
        "email": row[7],
        "notes": row[8],
        "timestamp": row[9],
    }
    try:
        net = _safe_json_loads(read_json_file(targets_json_path), {})
    except Exception:
        net = {}
    candidates = []
    if isinstance(net, dict):
        for tid, tdata in net.items():
            if not isinstance(tdata, dict):
                continue
            ip = tdata.get("ip", "")
            ports = tdata.get("ports", {})
            if not isinstance(ports, dict):
                continue
            port_list = sorted({int(p) for p in ports.keys() if str(p).isdigit()})
            candidates.append({"target_id": tid, "ip": ip, "ports": port_list})
    plan = {
        "credential": item,
        "safety": {
            "max_attempts_per_service": 3,
            "delay_seconds_between_attempts": 3,
            "stop_on_lockout_signals": True,
        },
        "candidates": candidates,
        "notes": "Test reuse conservatively on services that match the credential context (ssh/rdp/smb/web logins/db).",
    }
    return json.dumps(plan, indent=2)


@tool
def generate_network_report(network_json_path: str = "network_scan.json") -> str:
    """
    Generates a concise network report from network_scan.json.

    Args:
        network_json_path (str): Path to network_scan.json.
    """
    data = _safe_json_loads(read_json_file(network_json_path), {})
    if not isinstance(data, dict):
        return "Error: network JSON invalid"
    targets = []
    for tid, tdata in data.items():
        if not isinstance(tdata, dict):
            continue
        ip = tdata.get("ip", "")
        ports = tdata.get("ports", {})
        port_list = sorted([int(p) for p in ports.keys() if str(p).isdigit()])
        targets.append({"target_id": tid, "ip": ip, "ports": port_list, "summary": tdata.get("summary", "")})
    return json.dumps({"total_targets": len(targets), "targets": targets}, indent=2)


@tool
def generate_attack_path_summary(limit: int = 20, db_path: str = "pentest_memory.db") -> str:
    """
    Generates a short summary of top vulnerabilities and latest credentials.

    Args:
        limit (int): Max entries (default: 20).
        db_path (str): Path to SQLite database.
    """
    vulns = _safe_json_loads(list_vulnerabilities(limit=limit, db_path=db_path), [])
    creds = _safe_json_loads(list_interesting_data(type_filter="PASSWORD", limit=limit, db_path=db_path), [])
    return json.dumps({"top_vulnerabilities": vulns, "latest_passwords": creds}, indent=2)


@tool
def banner_grab_tcp(target_ip: str, port: int, bytes_limit: int = 4096, timeout_sec: int = 3) -> str:
    """
    Connects to target_ip:port and reads up to bytes_limit bytes for banner grabbing.

    Args:
        target_ip (str): Target IP address.
        port (int): Port to connect to.
        bytes_limit (int): Max bytes to read (default: 4096).
        timeout_sec (int): Socket timeout seconds (default: 3).
    """
    try:
        with socket.create_connection((target_ip, int(port)), timeout=max(1, int(timeout_sec))) as s:
            s.settimeout(max(1, int(timeout_sec)))
            data = s.recv(max(1, int(bytes_limit)))
            return data.decode("utf-8", errors="replace")
    except Exception as e:
        return f"Error: banner grab failed - {e}"


@tool
def http_request(
    url: str,
    method: str = "GET",
    headers_json: str = "{}",
    body: str = "",
    timeout_sec: int = 15,
) -> str:
    """
    Makes an HTTP request using urllib with controlled timeouts.

    Args:
        url (str): Full URL to request.
        method (str): HTTP method (GET, HEAD, POST, PUT, DELETE).
        headers_json (str): JSON object of headers.
        body (str): Request body for POST/PUT.
        timeout_sec (int): Timeout in seconds (default: 15).
    """
    m = method.strip().upper()
    if m not in {"GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS"}:
        return "Error: unsupported HTTP method"
    headers = _safe_json_loads(headers_json, {})
    if not isinstance(headers, dict):
        headers = {}
    data = body.encode("utf-8") if body and m in {"POST", "PUT"} else None
    req = urllib.request.Request(url=url, data=data, method=m)
    for k, v in headers.items():
        if isinstance(k, str) and isinstance(v, str):
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
            resp_body = resp.read(16 * 1024)
            out = {
                "final_url": resp.geturl(),
                "status": resp.status,
                "headers": dict(resp.headers.items()),
                "body_preview": resp_body.decode("utf-8", errors="replace"),
            }
            return json.dumps(out, indent=2)
    except Exception as e:
        return f"Error: http_request failed - {e}"


@tool
def http_headers(url: str, timeout_sec: int = 15) -> str:
    """
    Fetches HTTP headers (HEAD request) for a URL.

    Args:
        url (str): Full URL.
        timeout_sec (int): Timeout in seconds (default: 15).
    """
    return http_request(url=url, method="HEAD", timeout_sec=timeout_sec)


@tool
def robots_and_sitemap(base_url: str, timeout_sec: int = 15) -> str:
    """
    Fetches /robots.txt and /sitemap.xml from a base URL.

    Args:
        base_url (str): Base URL (scheme+host+optional port).
        timeout_sec (int): Timeout in seconds (default: 15).
    """
    robots = http_request(url=url_join(base_url, "/robots.txt"), method="GET", timeout_sec=timeout_sec)
    sitemap = http_request(url=url_join(base_url, "/sitemap.xml"), method="GET", timeout_sec=timeout_sec)
    return json.dumps({"robots_txt": _safe_json_loads(robots, robots), "sitemap_xml": _safe_json_loads(sitemap, sitemap)}, indent=2)


@tool
def tech_fingerprint(base_url: str) -> str:
    """
    Performs a light tech fingerprint using headers and HTML heuristics.

    Args:
        base_url (str): Base URL.
    """
    hdrs = _safe_json_loads(http_headers(base_url), {})
    body = _safe_json_loads(http_request(base_url), {})
    server = ""
    powered = ""
    if isinstance(hdrs, dict):
        server = str(hdrs.get("headers", {}).get("Server", ""))
        powered = str(hdrs.get("headers", {}).get("X-Powered-By", ""))
    title = ""
    if isinstance(body, dict):
        html = str(body.get("body_preview", ""))
        m = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            title = m.group(1).strip()[:200]
    return json.dumps({"server": server, "x_powered_by": powered, "title": title}, indent=2)


@tool
def dir_enum_safe(base_url: str, wordlist: str = "common", rps: int = 5, max_paths: int = 1000) -> str:
    """
    Performs safe directory enumeration with rate limiting.

    Args:
        base_url (str): Base URL.
        wordlist (str): Wordlist selector (default: common).
        rps (int): Requests per second cap (default: 5, max: 10).
        max_paths (int): Maximum paths to test (default: 1000).
    """
    rate = max(1, min(int(rps), 10))
    paths = []
    if wordlist == "common":
        paths = [
            "admin", "administrator", "login", "robots.txt", "sitemap.xml",
            ".git/HEAD", ".env", "backup", "backups", "test", "dev", "api", "swagger", "swagger-ui", "api-docs",
        ]
    else:
        return "Error: only wordlist=common supported in safe mode"

    results = []
    tested = 0
    delay = 1.0 / float(rate)
    for p in paths:
        if tested >= int(max_paths):
            break
        tested += 1
        url = url_join(base_url, "/" + p)
        resp = _safe_json_loads(http_request(url=url, method="GET", timeout_sec=10), {})
        status = None
        if isinstance(resp, dict):
            status = resp.get("status")
        if status in {200, 301, 302, 401, 403}:
            results.append({"path": "/" + p, "status": status})
        time.sleep(delay)
    return json.dumps({"tested": tested, "results": results}, indent=2)


@tool
def http_methods_probe(base_url: str) -> str:
    """
    Probes HTTP methods via an OPTIONS request.

    Args:
        base_url (str): Base URL.
    """
    return http_request(url=base_url, method="OPTIONS", timeout_sec=15)


@tool
def openapi_swagger_discover(base_url: str) -> str:
    """
    Checks common OpenAPI/Swagger endpoints.

    Args:
        base_url (str): Base URL.
    """
    candidates = [
        "/swagger.json",
        "/openapi.json",
        "/api-docs",
        "/swagger",
        "/swagger-ui",
    ]
    found = []
    for p in candidates:
        resp = _safe_json_loads(http_request(url=url_join(base_url, p), timeout_sec=10), {})
        if isinstance(resp, dict) and resp.get("status") in {200, 301, 302}:
            found.append({"path": p, "status": resp.get("status")})
        time.sleep(0.2)
    return json.dumps({"found": found}, indent=2)


@tool
def graphql_introspection_probe(base_url: str) -> str:
    """
    Performs a minimal GraphQL introspection probe (detection only).

    Args:
        base_url (str): Base URL (GraphQL endpoint should be /graphql in most stacks).
    """
    url = url_join(base_url, "/graphql")
    body = json.dumps({"query": "{__schema{queryType{name}}}"})
    hdrs = json.dumps({"Content-Type": "application/json"})
    return http_request(url=url, method="POST", headers_json=hdrs, body=body, timeout_sec=15)


@tool
def git_exposure_check(base_url: str) -> str:
    """
    Checks if .git exposure is present by requesting /.git/HEAD.

    Args:
        base_url (str): Base URL.
    """
    return http_request(url=url_join(base_url, "/.git/HEAD"), method="GET", timeout_sec=10)


@tool
def nikto_safe(base_url: str, tuning: str = "x") -> str:
    """
    Runs nikto in a conservative mode if installed.

    Args:
        base_url (str): Base URL.
        tuning (str): Nikto tuning string (default: x).
    """
    cmd = f"nikto -Tuning {tuning} -maxtime 2m -host '{base_url}'"
    return execute_bash(cmd, timeout_sec=140, max_output_kb=256)


@tool
def smb_os_discovery(target_ip: str) -> str:
    """
    Runs nmap SMB OS discovery scripts.

    Args:
        target_ip (str): Target IP address.
    """
    cmd = f"nmap -T3 --max-retries 2 --host-timeout 2m -p 445 --script smb-os-discovery,smb-security-mode '{target_ip}'"
    return execute_bash(cmd, timeout_sec=180, max_output_kb=256)


@tool
def smb_list_shares(target_ip: str, username: str = "", password: str = "") -> str:
    """
    Lists SMB shares using smbclient.

    Args:
        target_ip (str): Target IP address.
        username (str): Optional username.
        password (str): Optional password.
    """
    if username and password:
        cmd = f"smbclient -L //{target_ip} -U '{username}%{password}'"
    else:
        cmd = f"smbclient -L //{target_ip} -N"
    return execute_bash(cmd, timeout_sec=90, max_output_kb=256)


@tool
def smb_download_file(target_ip: str, username: str = "", password: str = "", share: str, remote_path: str, local_path: str) -> str:
    """
    Downloads a file from an SMB share (best-effort, no creds).

    Args:
        target_ip (str): Target IP address.
        username (str): Username to authenticate with.
        password (str): Password to authenticate with.
        share (str): Share name.
        remote_path (str): Remote file path in the share.
        local_path (str): Local output path.
    """
    out = Path(local_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if username and password:
        cmd = f"smbclient //{target_ip}/{share} -U '{username}%{password}' -c 'get \"{remote_path}\" \"{local_path}\"'"
    else:
        cmd = f"smbclient //{target_ip}/{share} -N -c 'get \"{remote_path}\" \"{local_path}\"'"
    return execute_bash(cmd, timeout_sec=120, max_output_kb=256)


@tool
def windows_password_policy(target_ip: str, username: str = "", password: str = "") -> str:
    """
    Attempts to retrieve Windows password policy using nxc for authenticated enumeration and enum4linux for unauthenticated (if installed).

    Args:
        target_ip (str): Target IP address.
        username (str): Username to authenticate with.
        password (str): Password to authenticate with.
    """
    if username and password:
        cmd = f"nxc smb '{target_ip}' -u '{username}' -p '{password}' --pass-pol"
    else:
        cmd = f"enum4linux -P '{target_ip}'"
    return execute_bash(cmd, timeout_sec=180, max_output_kb=256)


@tool
def rdp_enum_encryption(target_ip: str) -> str:
    """
    Enumerates RDP encryption settings using nmap.

    Args:
        target_ip (str): Target IP address.
    """
    cmd = f"nmap -T3 --max-retries 2 --host-timeout 2m -p 3389 --script rdp-enum-encryption '{target_ip}'"
    return execute_bash(cmd, timeout_sec=180, max_output_kb=256)


@tool
def ldap_rootdse(target_ip: str) -> str:
    """
    Queries LDAP rootDSE using nmap script.

    Args:
        target_ip (str): Target IP address.
    """
    cmd = f"nmap -T3 --max-retries 2 --host-timeout 2m -p 389 --script ldap-rootdse '{target_ip}'"
    return execute_bash(cmd, timeout_sec=180, max_output_kb=256)


@tool
def kerberos_user_enum_safe(domain: str, dc_ip: str, usernames_json: str, rps: int = 1) -> str:
    """
    Performs a conservative Kerberos username enumeration plan (best-effort).

    Args:
        domain (str): Domain name.
        dc_ip (str): Domain controller IP.
        usernames_json (str): JSON list of usernames.
        rps (int): Requests per second (default: 1).
    """
    users = _safe_json_loads(usernames_json, [])
    if not isinstance(users, list) or not users:
        return "Error: usernames_json must be a JSON list"
    rate = max(1, min(int(rps), 2))
    delay = 1.0 / float(rate)
    results = []
    for u in users[:50]:
        if not isinstance(u, str) or not u.strip():
            continue
        # Detection-only placeholder: real kerbrute may not be installed.
        results.append({"username": u.strip(), "note": "Use kerbrute if installed: kerbrute userenum"})
        time.sleep(delay)
    return json.dumps({"domain": domain, "dc_ip": dc_ip, "results": results}, indent=2)


@tool
def asrep_roast_detect_only(domain: str, dc_ip: str, usernames_json: str) -> str:
    """
    Detection-only placeholder for AS-REP roasting opportunities.

    Args:
        domain (str): Domain name.
        dc_ip (str): Domain controller IP.
        usernames_json (str): JSON list of usernames.
    """
    users = _safe_json_loads(usernames_json, [])
    if not isinstance(users, list) or not users:
        return "Error: usernames_json must be a JSON list"
    #return json.dumps(
        {
            #"note": "Detection only. If impacket is installed, use GetNPUsers.py to check AS-REP roastable accounts.",
            #"domain": domain,
            #"dc_ip": dc_ip,
            #"user_count": len(users),
        #},
        #indent=2,
    #)
    cmd = f"GetNPUsers.py -dc-ip '{target_ip}' '{domain_name}'/ -usersfile '{usernames_json}' -format hashcat"
    return execute_bash(cmd, timeout_sec=180, max_output_kb=600)
    

@tool
def kerberoast_detect_only(domain: str, dc_ip: str, username: str, password: str) -> str:
    """
    Kerberoast the domain with valid credentials.

    Args:
        domain (str): Domain name.
        dc_ip (str): Domain controller IP.
        username (str): Username.
        password (str): Password.
    """
    #return json.dumps(
        #{
            #"note": "Detection only. If impacket is installed, use GetUserSPNs.py to request TGS for SPNs.",
            #"domain": domain,
            #"dc_ip": dc_ip,
            #"username": username,
        #},
        #indent=2,
    #)
    cmd = f"GetUserSPNs.py -dc-ip '{dc_ip}' '{domain}'/'{username}':'{password}' -request -format hashcat"
    return execute_bash(cmd, timeout_sec=180, max_output_kb=600)


@tool
def gpp_decrypt(cpassword: str) -> str:
    """
    Decrypts a Group Policy Preferences cpassword using gpp-decrypt if installed.

    Args:
        cpassword (str): Encrypted cpassword value.
    """
    cmd = f"gpp-decrypt '{cpassword}'"
    return execute_bash(cmd, timeout_sec=30, max_output_kb=64)


@tool
def ssh_banner(target_ip: str) -> str:
    """
    Grabs SSH banner from port 22.

    Args:
        target_ip (str): Target IP address.
    """
    return banner_grab_tcp(target_ip=target_ip, port=22, bytes_limit=4096, timeout_sec=3)


@tool
def ssh_auth_test(target_ip: str, username: str, password: str, max_attempts: int = 3, delay_sec: int = 3) -> str:
    """
    Safe SSH auth test placeholder (does not brute force). Requires sshpass if doing password auth.

    Args:
        target_ip (str): Target IP address.
        username (str): Username to test.
        password (str): Password to test.
        max_attempts (int): Max attempts (default: 3).
        delay_sec (int): Delay between attempts seconds (default: 3).
    """
    attempts = max(1, min(int(max_attempts), 3))
    delay = max(1, int(delay_sec))
    out = []
    for _ in range(attempts):
        cmd = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 '{username}@{target_ip}' 'echo ok' 2>&1"
        out.append(execute_bash(cmd, timeout_sec=15, max_output_kb=64))
        time.sleep(delay)
    return "\n".join(out)


@tool
def ftp_anon_test(target_ip: str) -> str:
    """
    Tests anonymous FTP listing using curl if installed.

    Args:
        target_ip (str): Target IP address.
    """
    cmd = f"curl -sS 'ftp://{target_ip}/' --user 'anonymous:' --connect-timeout 5 --max-time 15"
    return execute_bash(cmd, timeout_sec=20, max_output_kb=128)


@tool
def mysql_connect_readonly(target_ip: str, username: str, password: str) -> str:
    """
    Tests MySQL connection with a harmless query.

    Args:
        target_ip (str): Target IP address.
        username (str): Username.
        password (str): Password.
    """
    cmd = f"mysql -h '{target_ip}' -u '{username}' -p'{password}' -e 'SELECT 1;' 2>&1"
    return execute_bash(cmd, timeout_sec=30, max_output_kb=128)


@tool
def mysql_query_readonly(target_ip: str, username: str, password: str, query: str) -> str:
    """
    Executes a READ ONLY MySQL query. Blocks common write keywords.

    Args:
        target_ip (str): Target IP address.
        username (str): Username.
        password (str): Password.
        query (str): Query to execute (must be read-only).
    """
    q = query.strip().lower()
    if any(k in q for k in ["insert", "update", "delete", "drop", "truncate", "alter", "create"]):
        return "Error: write queries are blocked"
    cmd = f"mysql -h '{target_ip}' -u '{username}' -p'{password}' -e {json.dumps(query)} 2>&1"
    return execute_bash(cmd, timeout_sec=60, max_output_kb=256)


@tool
def postgres_connect_readonly(target_ip: str, username: str, password: str) -> str:
    """
    Tests PostgreSQL connection with a harmless query.

    Args:
        target_ip (str): Target IP address.
        username (str): Username.
        password (str): Password.
    """
    cmd = f"PGPASSWORD='{password}' psql -h '{target_ip}' -U '{username}' -c 'SELECT 1;' 2>&1"
    return execute_bash(cmd, timeout_sec=30, max_output_kb=128)


@tool
def mongo_connect_readonly(target_ip: str) -> str:
    """
    Tests MongoDB connectivity (best-effort).

    Args:
        target_ip (str): Target IP address.
    """
    cmd = f"mongosh 'mongodb://{target_ip}:27017' --eval 'db.adminCommand({{ping:1}})' 2>&1"
    return execute_bash(cmd, timeout_sec=30, max_output_kb=128)


@tool
def redis_info_readonly(target_ip: str, password: str = "") -> str:
    """
    Fetches Redis INFO output (read-only).

    Args:
        target_ip (str): Target IP address.
        password (str): Optional password.
    """
    auth = f"-a '{password}'" if password else ""
    cmd = f"redis-cli -h '{target_ip}' {auth} INFO 2>&1"
    return execute_bash(cmd, timeout_sec=20, max_output_kb=256)


@tool
def snmpwalk_lite(target_ip: str, community: str = "public", oid: str = "1.3.6.1.2.1.1") -> str:
    """
    Performs a limited SNMP walk starting at a small OID.

    Args:
        target_ip (str): Target IP address.
        community (str): SNMP community string (default: public).
        oid (str): OID root (default: 1.3.6.1.2.1.1).
    """
    cmd = f"snmpwalk -v2c -c '{community}' '{target_ip}' '{oid}' 2>&1 | head -n 50"
    return execute_bash(cmd, timeout_sec=30, max_output_kb=128)


@tool
def dns_zone_xfer_once(server_ip: str, domain: str) -> str:
    """
    Attempts a single DNS zone transfer via dig.

    Args:
        server_ip (str): DNS server IP.
        domain (str): Domain name.
    """
    cmd = f"dig @{server_ip} {domain} AXFR +time=3 +tries=1"
    return execute_bash(cmd, timeout_sec=20, max_output_kb=256)


@tool
def nfs_exports(target_ip: str) -> str:
    """
    Lists NFS exports via showmount.

    Args:
        target_ip (str): Target IP address.
    """
    cmd = f"showmount -e '{target_ip}' 2>&1"
    return execute_bash(cmd, timeout_sec=20, max_output_kb=128)


@tool
def detect_secrets_in_text(text: str) -> str:
    """
    Performs regex-based secret detection on a text blob.

    Args:
        text (str): Text to scan.
    """
    patterns = {
        "aws_access_key_id": r"AKIA[0-9A-Z]{16}",
        "github_token": r"gh[pous]_[A-Za-z0-9]{20,}",
        "generic_api_key": r"(?i)api[_-]?key\\s*[:=]\\s*['\\\"][^'\\\"]{8,}['\\\"]",
        "password_assignment": r"(?i)password\\s*[:=]\\s*['\\\"][^'\\\"]{4,}['\\\"]",
        "jwt_like": r"eyJ[a-zA-Z0-9_\\-]+\\.eyJ[a-zA-Z0-9_\\-]+\\.[a-zA-Z0-9_\\-]+",
    }
    found = {}
    for name, pat in patterns.items():
        try:
            m = re.findall(pat, text)
            if m:
                found[name] = m[:10]
        except Exception:
            continue
    return json.dumps(found, indent=2)


@tool
def scan_repo_for_secrets(repo_dir: str, max_files: int = 2000) -> str:
    """
    Scans a directory for common secret patterns.

    Args:
        repo_dir (str): Directory to scan.
        max_files (int): Max files to read (default: 2000).
    """
    root = Path(repo_dir)
    if not root.exists() or not root.is_dir():
        return "Error: repo_dir does not exist"
    scanned = 0
    findings: list[dict[str, object]] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if scanned >= int(max_files):
                break
            path = Path(dirpath) / name
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz"}:
                continue
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            hits = _safe_json_loads(detect_secrets_in_text(text), {})
            if isinstance(hits, dict) and hits:
                findings.append({"file": str(path), "matches": hits})
            scanned += 1
    return json.dumps({"scanned": scanned, "findings": findings}, indent=2)


@tool
def parse_env_file(env_text: str) -> str:
    """
    Parses .env file content into JSON.

    Args:
        env_text (str): Raw .env content.
    """
    out = {}
    for line in env_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip("'").strip('"')
    return json.dumps(out, indent=2)


@tool
def dependency_inventory(repo_dir: str) -> str:
    """
    Extracts a lightweight dependency inventory from common manifest files.

    Args:
        repo_dir (str): Directory containing project files.
    """
    root = Path(repo_dir)
    if not root.exists() or not root.is_dir():
        return "Error: repo_dir does not exist"
    files = {
        "requirements.txt": root / "requirements.txt",
        "pyproject.toml": root / "pyproject.toml",
        "package.json": root / "package.json",
        "composer.json": root / "composer.json",
        "pom.xml": root / "pom.xml",
        "Gemfile": root / "Gemfile",
    }
    out = {}
    for name, path in files.items():
        if path.exists():
            try:
                out[name] = path.read_text(errors="ignore")[:5000]
            except Exception:
                continue
    return json.dumps(out, indent=2)

@tool
def safe_ping_sweep(subnet_cidr: str, rate_per_sec: int = 10, timeout_sec: int = 2) -> str:
    """
    Performs a conservative ping sweep using nmap. Note: ICMP ping may be blocked on VPN networks.
    For better results, use safe_tcp_connect_probe or nmap_scan with -Pn flag.

    Args:
        subnet_cidr (str): CIDR (e.g., 192.168.1.0/24).
        rate_per_sec (int): Delay-based rate cap (default: 10, not currently enforced).
        timeout_sec (int): Per-ping timeout seconds (default: 2, not currently enforced).
    """
    return execute_bash(
        f"nmap -T3 --max-retries 2 --host-timeout 2m -sn {subnet_cidr}",
        timeout_sec=300,
        max_output_kb=512,
    )


@tool
def safe_tcp_connect_probe(subnet_cidr: str, ports_json: str = "[22,80,443,445,3389]", rate_per_sec: int = 10) -> str:
    """
    Performs a conservative TCP connect probe for a few ports using nmap.
    Uses -Pn to skip ping discovery (works on VPN networks where ICMP is blocked).

    Args:
        subnet_cidr (str): CIDR range (e.g., 10.10.110.0/24).
        ports_json (str): JSON list of ports (default: [22,80,443,445,3389]).
        rate_per_sec (int): Unused placeholder for future rate control; kept for schema stability.
    """
    ports = _safe_json_loads(ports_json, [])
    if not isinstance(ports, list) or not ports:
        return "Error: ports_json must be a JSON list of ports"
    port_str = ",".join(str(int(p)) for p in ports if str(p).isdigit())
    if not port_str:
        return "Error: No valid ports found in ports_json"
    # Increase timeout for subnet scans
    timeout = 600 if "/" in subnet_cidr else 300
    return execute_bash(
        f"nmap -T3 --max-retries 1 --host-timeout 30s -sT -Pn --open -p {port_str} {subnet_cidr}",
        timeout_sec=timeout,
        max_output_kb=512,
    )


@tool
def download_url_to_file(url: str, output_path: str, max_bytes: int = 5_000_000, timeout_sec: int = 20) -> str:
    """
    Downloads a URL to a local file with a size cap.

    Args:
        url (str): URL to download.
        output_path (str): Local file path to write.
        max_bytes (int): Maximum bytes to write (default: 5,000,000).
        timeout_sec (int): Timeout in seconds (default: 20).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
            data = resp.read(max(1, int(max_bytes)) + 1)
        if len(data) > int(max_bytes):
            return "Error: download exceeded max_bytes"
        out.write_bytes(data)
        return f"Downloaded {len(data)} bytes to {output_path}"
    except Exception as e:
        return f"Error: download failed - {e}"


def _safe_extract_path(root: Path, member_name: str) -> Path | None:
    root_abs = root.resolve()
    dest = (root / member_name).resolve()
    if str(dest).startswith(str(root_abs)):
        return dest
    return None


@tool
def unarchive_to_dir(archive_path: str, output_dir: str, max_files: int = 500) -> str:
    """
    Extracts zip or tar archives to a directory with file-count and path traversal protections.

    Args:
        archive_path (str): Path to archive file.
        output_dir (str): Directory to extract into.
        max_files (int): Max number of files to extract (default: 500).
    """
    src = Path(archive_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    extracted = 0
    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src, "r") as zf:
            for info in zf.infolist():
                if extracted >= int(max_files):
                    break
                dest = _safe_extract_path(out, info.filename)
                if dest is None:
                    continue
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as rf:
                    dest.write_bytes(rf.read())
                extracted += 1
        return f"Extracted {extracted} files from zip"
    if tarfile.is_tarfile(src):
        with tarfile.open(src, "r:*") as tf:
            for member in tf.getmembers():
                if extracted >= int(max_files):
                    break
                if not member.name:
                    continue
                dest = _safe_extract_path(out, member.name)
                if dest is None:
                    continue
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                if member.isfile():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    dest.write_bytes(f.read())
                    extracted += 1
        return f"Extracted {extracted} files from tar"
    return "Error: unsupported archive type"


@tool
def grep_in_dir(directory: str, pattern: str, max_hits: int = 200) -> str:
    """
    Greps recursively for a regex pattern in text files under a directory.

    Args:
        directory (str): Directory to search.
        pattern (str): Regex pattern.
        max_hits (int): Max matches (default: 200).
    """
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        return "Error: directory does not exist"
    try:
        rx = re.compile(pattern)
    except Exception as e:
        return f"Error: invalid regex - {e}"
    hits: list[dict[str, object]] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if len(hits) >= int(max_hits):
                break
            path = Path(dirpath) / name
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append({"file": str(path), "line": idx, "text": line[:300]})
                    if len(hits) >= int(max_hits):
                        break
    return json.dumps({"hits": hits, "count": len(hits)}, indent=2)


@tool
def read_json_file(file_path: str) -> str:
    """
    Reads a JSON file and returns formatted JSON or {} if missing.

    Args:
        file_path (str): Path to JSON file.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({})
        
        with open(path, 'r') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file - {str(e)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_json_file(file_path: str, json_content: str) -> str:
    """
    Writes JSON content (string) to a file.

    Args:
        file_path (str): Path to JSON file to write.
        json_content (str): JSON string to persist.
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = json.loads(json_content)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"Successfully wrote JSON to {file_path}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON content - {str(e)}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def add_target(json_file: str, target_id: str, ip: str, summary: str) -> str:
    """
    Adds a target entry to the network JSON.

    Args:
        json_file (str): Path to JSON file.
        target_id (str): Identifier (e.g., target1).
        ip (str): Target IP.
        summary (str): Target summary (device/OS/services).
    """
    try:
        path = Path(json_file)
        
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        
        if target_id in data:
            return f"Error: Target {target_id} already exists. Use update_target or add_port instead."
        
        data[target_id] = {
            "ip": ip,
            "summary": summary,
            "ports": {}
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"Successfully added target {target_id} with IP {ip}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file - {str(e)}"
    except Exception as e:
        return f"Error adding target: {str(e)}"


@tool
def add_port(json_file: str, target_id: str, port: str, port_summary: str) -> str:
    """
    Adds a port entry to an existing target in the JSON.

    Args:
        json_file (str): Path to JSON file.
        target_id (str): Target identifier.
        port (str): Port number.
        port_summary (str): Description of service on port.
    """
    try:
        path = Path(json_file)
        
        if not path.exists():
            return f"Error: JSON file {json_file} does not exist. Create target first with add_target."
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if target_id not in data:
            return f"Error: Target {target_id} does not exist. Create it first with add_target."
        
        if "ports" not in data[target_id]:
            data[target_id]["ports"] = {}
        
        data[target_id]["ports"][port] = port_summary
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"Successfully added port {port} to {target_id}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file - {str(e)}"
    except Exception as e:
        return f"Error adding port: {str(e)}"


@tool
def update_target_summary(json_file: str, target_id: str, summary: str) -> str:
    """
    Updates the summary for a target in the JSON.

    Args:
        json_file (str): Path to JSON file.
        target_id (str): Target identifier.
        summary (str): New summary text.
    """
    try:
        path = Path(json_file)
        
        if not path.exists():
            return f"Error: JSON file {json_file} does not exist."
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if target_id not in data:
            return f"Error: Target {target_id} does not exist."
        
        data[target_id]["summary"] = summary
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"Successfully updated summary for {target_id}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file - {str(e)}"
    except Exception as e:
        return f"Error updating target: {str(e)}"


# -----------------------------
# SQLite helpers for vulns/data
# -----------------------------

def _ensure_extra_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vulnerabilities (
            vuln_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bug_name TEXT,
            risk TEXT,
            impact TEXT,
            sophistication TEXT,
            assets TEXT,
            description TEXT,
            risk_justification TEXT,
            mitre_techniques TEXT,
            mitigations TEXT,
            replication TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interesting_data (
            data_id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_ip TEXT,
            category TEXT,
            service TEXT,
            username TEXT,
            password TEXT,
            hash TEXT,
            email TEXT,
            notes TEXT,
            timestamp TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_ip TEXT,
            message TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()


def _wake_sleeping_targets(conn: sqlite3.Connection, message: str) -> int:
    cur = conn.cursor()
    try:
        cur.execute("SELECT ip FROM targets WHERE current_stage = 'sleeping'")
        sleeping = [row[0] for row in cur.fetchall()]
    except sqlite3.OperationalError:
        sleeping = []
    count = 0
    if sleeping:
        now = datetime.utcnow().isoformat()
        for ip in sleeping:
            cur.execute(
                "INSERT INTO notifications (target_ip, message, created_at) VALUES (?, ?, ?)",
                (ip, message, now),
            )
            count += 1
        conn.commit()
    return count


_RISK_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"}


def _build_risk_justification(impact: str, sophistication: str) -> str:
    return (
        f"Risk Justification: The impact is {impact.title()}. Successful exploitation could lead to "
        f"significant compromise. The sophistication is {sophistication.title()}. Successful exploitation requires "
        f"an attacker with capabilities aligned to this sophistication level."
    )


@tool
def write_vulnerability(
    bug_name: str,
    risk: str,
    impact: str,
    sophistication: str,
    assets: str,
    description: str,
    mitre_techniques: str = "",
    mitigations: str = "",
    replication: str = "",
    db_path: str = "pentest_memory.db",
) -> str:
    """
    Inserts a structured vulnerability record into SQLite and notifies sleeping agents.

    Args:
        bug_name (str): The vulnerability name.
        risk (str): One of CRITICAL|HIGH|MEDIUM|LOW|INFORMATIONAL.
        impact (str): One of CRITICAL|HIGH|MEDIUM|LOW|INFORMATIONAL.
        sophistication (str): One of CRITICAL|HIGH|MEDIUM|LOW|INFORMATIONAL.
        assets (str): Comma-separated assets, e.g., "http://10.0.1.14:5001/,example.local:600".
        description (str): Narrative description of the issue.
        mitre_techniques (str): Comma-separated MITRE techniques.
        mitigations (str): Comma-separated mitigations.
        replication (str): Commands/steps to reproduce.
        db_path (str): Path to SQLite database.
    """
    risk_u = risk.strip().upper()
    impact_u = impact.strip().upper()
    soph_u = sophistication.strip().upper()
    for val in (risk_u, impact_u, soph_u):
        if val not in _RISK_LEVELS:
            return f"Error: risk/impact/sophistication must be one of {_RISK_LEVELS}"

    asset_list = [a.strip() for a in assets.split(",") if a.strip()]
    mitre_list = [m.strip() for m in mitre_techniques.split(",") if m.strip()]
    mitigation_list = [m.strip() for m in mitigations.split(",") if m.strip()]

    justification = _build_risk_justification(impact_u, soph_u)

    db_logger = logging.getLogger("database")
    db_logger.info(f"write_vulnerability: {bug_name} (risk={risk_u}, impact={impact_u})")
    
    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO vulnerabilities
        (bug_name, risk, impact, sophistication, assets, description, risk_justification, mitre_techniques, mitigations, replication, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            bug_name,
            risk_u,
            impact_u,
            soph_u,
            json.dumps(asset_list),
            description,
            justification,
            json.dumps(mitre_list),
            json.dumps(mitigation_list),
            replication,
            now,
        ),
    )
    conn.commit()
    notified = _wake_sleeping_targets(conn, "New vulnerability data was added. Review and see if you can use it.")
    conn.close()
    
    db_logger.info(f"write_vulnerability: recorded vuln_id={cur.lastrowid}, notified={notified} agents")
    return f"Vulnerability recorded. Assets={len(asset_list)}. Sleeping agents notified={notified}."


_DATA_TYPES = {"PASSWORD", "EMAIL", "USERNAME", "HASH", "NOTE"}


@tool
def store_interesting_data(
    data_type: str,
    data_value: str,
    target_ip: str = "",
    service: str = "",
    notes: str = "",
    db_path: str = "pentest_memory.db",
) -> str:
    """
    Stores a single piece of interesting data in SQLite and notifies sleeping agents.

    Args:
        data_type (str): One of PASSWORD|EMAIL|USERNAME|HASH|NOTE.
        data_value (str): The value to store for the given data_type.
        target_ip (str): Optional target IP this data relates to.
        service (str): Optional service name.
        notes (str): Optional notes; if data_type is NOTE, data_value is stored as the note.
        db_path (str): Path to SQLite database.
    """
    dt = data_type.strip().upper()
    if dt not in _DATA_TYPES:
        return f"Error: data_type must be one of {_DATA_TYPES}"
    if not data_value:
        return "Error: data_value is required"

    # Map to columns
    username = data_value if dt == "USERNAME" else ""
    password = data_value if dt == "PASSWORD" else ""
    hash_val = data_value if dt == "HASH" else ""
    email_val = data_value if dt == "EMAIL" else ""
    notes_val = notes if notes else ""
    if dt == "NOTE":
        notes_val = data_value if data_value else notes_val

    db_logger = logging.getLogger("database")
    db_logger.info(f"store_interesting_data: type={dt}, target_ip={target_ip}, service={service}")
    
    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO interesting_data
        (target_ip, category, service, username, password, hash, email, notes, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            target_ip,
            dt,
            service,
            username,
            password,
            hash_val,
            email_val,
            notes_val,
            now,
        ),
    )
    conn.commit()
    notified = _wake_sleeping_targets(conn, "New interesting data was added. Review and see if you can use it.")
    conn.close()
    
    db_logger.info(f"store_interesting_data: recorded data_id={cur.lastrowid}, notified={notified} agents")
    return f"Interesting data stored (type={dt}). Sleeping agents notified={notified}."


def get_recon_tools():
    """Get all tools for the recon agent as Odin Tool instances."""
    return get_recon_toolset()


def _wrap_all(funcs: list) -> list:
    return [function_to_odin_tool(f) for f in funcs]


def get_core_toolset() -> list:
    funcs = [
        execute_bash,
        read_json_file,
        write_json_file,
        add_target,
        add_port,
        update_target_summary,
        write_vulnerability,
        store_interesting_data,
        poll_notifications,
        ack_notification,
        url_join,
        url_encode,
        url_decode,
        base64_encode,
        base64_decode,
        regex_extract,
    ]
    return _wrap_all(funcs)


def get_recon_toolset() -> list:
    funcs = [
        nmap_scan,
        safe_ping_sweep,
        safe_tcp_connect_probe,
    ]
    return get_core_toolset() + _wrap_all(funcs)


def get_manager_toolset() -> list:
    funcs = [
        list_vulnerabilities,
        list_interesting_data,
        get_latest_activity,
        credential_reuse_plan,
        generate_network_report,
        generate_attack_path_summary,
        broadcast_notification,
        delegate_to_subagent,
    ]
    return get_core_toolset() + _wrap_all(funcs)


def get_targeted_toolset() -> list:
    funcs = [
        nmap_scan,
        banner_grab_tcp,
        http_request,
        http_headers,
        robots_and_sitemap,
        tech_fingerprint,
        dir_enum_safe,
        http_methods_probe,
        openapi_swagger_discover,
        graphql_introspection_probe,
        git_exposure_check,
        nikto_safe,
        smb_os_discovery,
        smb_list_shares,
        smb_download_file,
        windows_password_policy,
        rdp_enum_encryption,
        ldap_rootdse,
        kerberos_user_enum_safe,
        asrep_roast_detect_only,
        kerberoast_detect_only,
        gpp_decrypt,
        ssh_banner,
        ssh_auth_test,
        ftp_anon_test,
        mysql_connect_readonly,
        mysql_query_readonly,
        postgres_connect_readonly,
        mongo_connect_readonly,
        redis_info_readonly,
        snmpwalk_lite,
        dns_zone_xfer_once,
        nfs_exports,
        download_url_to_file,
        unarchive_to_dir,
        grep_in_dir,
        enter_wait_mode,
    ]
    return get_core_toolset() + _wrap_all(funcs)


def get_web_toolset() -> list:
    funcs = [
        http_request,
        http_headers,
        robots_and_sitemap,
        tech_fingerprint,
        dir_enum_safe,
        http_methods_probe,
        openapi_swagger_discover,
        graphql_introspection_probe,
        git_exposure_check,
        nikto_safe,
        download_url_to_file,
    ]
    return get_core_toolset() + _wrap_all(funcs)


def get_windows_toolset() -> list:
    funcs = [
        nmap_scan,
        banner_grab_tcp,
        smb_os_discovery,
        smb_list_shares,
        smb_download_file,
        windows_password_policy,
        rdp_enum_encryption,
        ldap_rootdse,
        kerberos_user_enum_safe,
        asrep_roast_detect_only,
        kerberoast_detect_only,
        gpp_decrypt,
    ]
    return get_core_toolset() + _wrap_all(funcs)


def get_misc_toolset() -> list:
    funcs = [
        nmap_scan,
        banner_grab_tcp,
        ssh_banner,
        ssh_auth_test,
        ftp_anon_test,
        mysql_connect_readonly,
        mysql_query_readonly,
        postgres_connect_readonly,
        mongo_connect_readonly,
        redis_info_readonly,
        snmpwalk_lite,
        dns_zone_xfer_once,
        nfs_exports,
    ]
    return get_core_toolset() + _wrap_all(funcs)


def get_code_review_toolset() -> list:
    funcs = [
        download_url_to_file,
        unarchive_to_dir,
        grep_in_dir,
        detect_secrets_in_text,
        scan_repo_for_secrets,
        parse_env_file,
        dependency_inventory,
    ]
    return get_core_toolset() + _wrap_all(funcs)


@tool
def enter_wait_mode(target_id: str, reason: str = "", db_path: str = "pentest_memory.db") -> str:
    """
    Enter wait mode when the agent has exhausted its capabilities.
    The agent will wait for new data (credentials, vulnerabilities) from other agents
    before resuming work.
    
    Args:
        target_id (str): Target identifier (IP or target ID).
        reason (str): Reason for entering wait mode (e.g., "No credentials available", "All stages complete").
        db_path (str): Path to SQLite database.
    """
    db_logger = logging.getLogger("database")
    db_logger.info(f"enter_wait_mode: {target_id} entering wait mode - {reason}")
    
    conn = sqlite3.connect(db_path)
    _ensure_extra_tables(conn)
    cur = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO notifications (target_ip, message, created_at)
        VALUES (?, ?, ?)
        """,
        (target_id, f"Agent entered wait mode: {reason}", now)
    )
    
    conn.commit()
    conn.close()
    
    return f"Entered wait mode. Waiting for new data from other agents. Reason: {reason}"
