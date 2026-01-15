#!/usr/bin/env python3
"""
Shared memory/database for multi-agent penetration testing system.
Uses SQLite to store targets, findings, credentials, and other shared state.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import asdict
import threading

from base import Target, Finding, AgentStage, RiskLevel

logger = logging.getLogger(__name__)


class SharedMemory:
    """SQLite-based shared memory for coordinating multiple pentest agents."""
    
    def __init__(self, db_path: str = "pentest_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn
    
    def _init_db(self):
        """Initialize the database schema."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            # Targets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS targets (
                    ip TEXT PRIMARY KEY,
                    hostname TEXT,
                    open_ports TEXT,  -- JSON array
                    services TEXT,    -- JSON dict
                    os_guess TEXT,
                    summary TEXT,
                    current_stage TEXT,
                    last_updated TEXT,
                    discovered_at TEXT
                )
            """)
            
            # Findings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS findings (
                    finding_id TEXT PRIMARY KEY,
                    target_ip TEXT,
                    title TEXT,
                    risk TEXT,
                    service TEXT,
                    description TEXT,
                    remediation TEXT,
                    timestamp TEXT,
                    discovered_by TEXT,
                    FOREIGN KEY (target_ip) REFERENCES targets(ip)
                )
            """)
            
            # Credentials table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS credentials (
                    credential_id TEXT PRIMARY KEY,
                    target_ip TEXT,
                    service TEXT,
                    username TEXT,
                    password TEXT,
                    hash TEXT,
                    email TEXT,
                    timestamp TEXT,
                    discovered_by TEXT,
                    FOREIGN KEY (target_ip) REFERENCES targets(ip)
                )
            """)
            
            # Network summary
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS network_summary (
                    id INTEGER PRIMARY KEY,
                    network_diagram TEXT,
                    updated_at TEXT
                )
            """)
            
            conn.commit()
            conn.close()
    
    def add_target(self, target: Target) -> None:
        """Add or update a target."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            discovered_at = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO targets 
                (ip, hostname, open_ports, services, os_guess, summary, 
                 current_stage, last_updated, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                target.ip,
                target.hostname,
                json.dumps(target.open_ports),
                json.dumps(target.services),
                target.os_guess,
                target.summary,
                target.current_stage.value,
                target.last_updated.isoformat(),
                discovered_at
            ))
            
            conn.commit()
            conn.close()
    
    def get_target(self, ip: str) -> Optional[Target]:
        """Get a target by IP."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM targets WHERE ip = ?", (ip,))
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return Target(
                ip=row[0],
                hostname=row[1],
                open_ports=json.loads(row[2]) if row[2] else [],
                services=json.loads(row[3]) if row[3] else {},
                os_guess=row[4],
                summary=row[5] or "",
                current_stage=AgentStage(row[6]) if row[6] else AgentStage.SERVICE_DISCOVERY,
                last_updated=datetime.fromisoformat(row[7]),
                findings=self.get_findings_for_target(ip)
            )
    
    def get_all_targets(self) -> List[Target]:
        """Get all targets."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("SELECT ip FROM targets")
            ips = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return [self.get_target(ip) for ip in ips if self.get_target(ip)]
    
    def add_finding(self, finding: Finding) -> None:
        """Add a finding."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO findings
                (finding_id, target_ip, title, risk, service, description, 
                 remediation, timestamp, discovered_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                finding.finding_id,
                finding.target_ip,
                finding.title,
                finding.risk.value,
                finding.service,
                finding.description,
                finding.remediation,
                finding.timestamp.isoformat(),
                finding.discovered_by
            ))
            
            conn.commit()
            conn.close()
    
    def get_findings_for_target(self, target_ip: str) -> List[Finding]:
        """Get all findings for a target."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM findings WHERE target_ip = ?", (target_ip,))
            rows = cursor.fetchall()
            conn.close()
            
            findings = []
            for row in rows:
                findings.append(Finding(
                    finding_id=row[0],
                    target_ip=row[1],
                    title=row[2],
                    risk=RiskLevel(row[3]),
                    service=row[4],
                    description=row[5] or "",
                    remediation=row[6] or "",
                    timestamp=datetime.fromisoformat(row[7]),
                    discovered_by=row[8]
                ))
            
            return findings
    
    def add_credential(self, target_ip: str, service: str, username: Optional[str] = None,
                      password: Optional[str] = None, hash_value: Optional[str] = None,
                      email: Optional[str] = None, discovered_by: Optional[str] = None) -> str:
        """Add a credential (password, hash, email, username)."""
        credential_id = f"{target_ip}_{service}_{datetime.now().timestamp()}"
        
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO credentials
                (credential_id, target_ip, service, username, password, hash, email, 
                 timestamp, discovered_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                credential_id,
                target_ip,
                service,
                username,
                password,
                hash_value,
                email,
                datetime.now().isoformat(),
                discovered_by
            ))
            
            conn.commit()
            conn.close()
        
        return credential_id
    
    def get_credentials_for_target(self, target_ip: str) -> List[Dict[str, Any]]:
        """Get all credentials for a target."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM credentials WHERE target_ip = ?", (target_ip,))
            rows = cursor.fetchall()
            conn.close()
            
            credentials = []
            for row in rows:
                credentials.append({
                    'credential_id': row[0],
                    'target_ip': row[1],
                    'service': row[2],
                    'username': row[3],
                    'password': row[4],
                    'hash': row[5],
                    'email': row[6],
                    'timestamp': row[7],
                    'discovered_by': row[8]
                })
            
            return credentials
    
    def update_target_stage(self, ip: str, stage: AgentStage) -> None:
        """Update the current stage of a target."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE targets 
                SET current_stage = ?, last_updated = ?
                WHERE ip = ?
            """, (stage.value, datetime.now().isoformat(), ip))
            
            conn.commit()
            conn.close()
    
    def set_network_diagram(self, diagram: str) -> None:
        """Set the network diagram."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            # Delete old summary and insert new one
            cursor.execute("DELETE FROM network_summary")
            cursor.execute("""
                INSERT INTO network_summary (network_diagram, updated_at)
                VALUES (?, ?)
            """, (diagram, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
    
    def get_network_summary_json(self) -> Dict[str, Any]:
        """Get network summary as JSON."""
        targets = self.get_all_targets()
        summary = {}
        
        for target in targets:
            summary[target.ip] = {
                'ip': target.ip,
                'hostname': target.hostname,
                'open_ports': target.open_ports,
                'services': target.services,
                'os_guess': target.os_guess,
                'summary': target.summary,
                'current_stage': target.current_stage.value,
                'findings_count': len(target.findings)
            }
        
        return summary

