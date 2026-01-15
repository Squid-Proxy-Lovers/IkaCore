#!/usr/bin/env python3
"""
Database dump utility for penetration testing SQLite database.
Exports data in multiple formats: SQL, JSON, CSV.
"""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def export_sql(db_path: str, output_path: str) -> None:
    """Export entire database as SQL dump."""
    conn = sqlite3.connect(db_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in conn.iterdump():
            f.write(f"{line}\n")
    
    conn.close()
    print(f"SQL dump written to {output_path}")


def export_json(db_path: str, output_path: str, table: str = None) -> None:
    """Export database tables as JSON."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    
    if table:
        tables = [table]
    else:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
    
    data = {}
    
    for table_name in tables:
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        data[table_name] = [dict(row) for row in rows]
    
    conn.close()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"JSON export written to {output_path}")
    for table_name, rows in data.items():
        print(f"  {table_name}: {len(rows)} rows")


def export_csv(db_path: str, output_dir: str, table: str = None) -> None:
    """Export database tables as CSV files."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    
    if table:
        tables = [table]
    else:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for table_name in tables:
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        if not rows:
            continue
        
        csv_path = output_path / f"{table_name}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows([dict(row) for row in rows])
        
        print(f"  {table_name}.csv: {len(rows)} rows")
    
    conn.close()
    print(f"CSV exports written to {output_dir}")


def export_vulnerabilities(db_path: str, output_path: str, format: str = "json") -> None:
    """Export vulnerabilities in a readable format."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            vuln_id,
            bug_name,
            risk,
            impact,
            sophistication,
            assets,
            description,
            risk_justification,
            mitre_techniques,
            mitigations,
            replication,
            created_at
        FROM vulnerabilities
        ORDER BY 
            CASE risk
                WHEN 'CRITICAL' THEN 1
                WHEN 'HIGH' THEN 2
                WHEN 'MEDIUM' THEN 3
                WHEN 'LOW' THEN 4
                ELSE 5
            END,
            created_at DESC
    """)
    
    vulns = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vulns, f, indent=2, default=str)
    elif format == "markdown":
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Vulnerability Report\n\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")
            f.write(f"Total vulnerabilities: {len(vulns)}\n\n")
            
            for vuln in vulns:
                f.write(f"## {vuln['bug_name']}\n\n")
                f.write(f"**Risk:** {vuln['risk']} | **Impact:** {vuln['impact']} | **Sophistication:** {vuln['sophistication']}\n\n")
                f.write(f"**Affected Assets:** {vuln['assets']}\n\n")
                f.write(f"**Description:**\n{vuln['description']}\n\n")
                f.write(f"**Risk Justification:**\n{vuln['risk_justification']}\n\n")
                if vuln['mitre_techniques']:
                    f.write(f"**MITRE ATT&CK Techniques:** {vuln['mitre_techniques']}\n\n")
                if vuln['mitigations']:
                    f.write(f"**Mitigations:** {vuln['mitigations']}\n\n")
                if vuln['replication']:
                    f.write(f"**Replication Steps:**\n```\n{vuln['replication']}\n```\n\n")
                f.write(f"*Discovered: {vuln['created_at']}*\n\n")
                f.write("---\n\n")
    
    print(f"Vulnerabilities exported to {output_path} ({len(vulns)} vulnerabilities)")


def export_credentials(db_path: str, output_path: str, format: str = "json") -> None:
    """Export credentials and interesting data."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            data_id,
            target_ip,
            category,
            service,
            username,
            password,
            hash,
            email,
            notes,
            timestamp
        FROM interesting_data
        ORDER BY timestamp DESC
    """)
    
    data = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    elif format == "csv":
        if not data:
            print("No credentials to export")
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    
    print(f"Credentials exported to {output_path} ({len(data)} entries)")


def show_summary(db_path: str) -> None:
    """Show database summary statistics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n=== Database Summary ===\n")
    
    cursor.execute("SELECT COUNT(*) FROM vulnerabilities")
    vuln_count = cursor.fetchone()[0]
    print(f"Vulnerabilities: {vuln_count}")
    
    cursor.execute("SELECT COUNT(*) FROM interesting_data")
    data_count = cursor.fetchone()[0]
    print(f"Interesting Data Entries: {data_count}")
    
    cursor.execute("SELECT COUNT(*) FROM notifications")
    notif_count = cursor.fetchone()[0]
    print(f"Notifications: {notif_count}")
    
    try:
        cursor.execute("SELECT COUNT(*) FROM agent_tasks")
        task_count = cursor.fetchone()[0]
        print(f"Agent Tasks: {task_count}")
    except sqlite3.OperationalError:
        pass
    
    cursor.execute("""
        SELECT risk, COUNT(*) 
        FROM vulnerabilities 
        GROUP BY risk 
        ORDER BY 
            CASE risk
                WHEN 'CRITICAL' THEN 1
                WHEN 'HIGH' THEN 2
                WHEN 'MEDIUM' THEN 3
                WHEN 'LOW' THEN 4
                ELSE 5
            END
    """)
    print("\nVulnerabilities by Risk Level:")
    for risk, count in cursor.fetchall():
        print(f"  {risk}: {count}")
    
    cursor.execute("""
        SELECT category, COUNT(*) 
        FROM interesting_data 
        GROUP BY category
    """)
    print("\nInteresting Data by Category:")
    for category, count in cursor.fetchall():
        print(f"  {category}: {count}")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Dump penetration testing database in various formats"
    )
    parser.add_argument(
        "--db",
        default="pentest_memory.db",
        help="Path to SQLite database (default: pentest_memory.db)"
    )
    parser.add_argument(
        "--output",
        help="Output file or directory path"
    )
    parser.add_argument(
        "--format",
        choices=["sql", "json", "csv", "markdown"],
        default="json",
        help="Export format (default: json)"
    )
    parser.add_argument(
        "--table",
        help="Export specific table only (for json/csv formats)"
    )
    parser.add_argument(
        "--vulns-only",
        action="store_true",
        help="Export only vulnerabilities"
    )
    parser.add_argument(
        "--creds-only",
        action="store_true",
        help="Export only credentials/interesting data"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show database summary statistics"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    
    if args.summary:
        show_summary(str(db_path))
        return
    
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.vulns_only:
            args.output = f"vulnerabilities_{timestamp}.{args.format}"
        elif args.creds_only:
            args.output = f"credentials_{timestamp}.{args.format}"
        elif args.format == "csv":
            args.output = f"db_export_{timestamp}"
        else:
            args.output = f"db_export_{timestamp}.{args.format}"
    
    if args.vulns_only:
        export_vulnerabilities(str(db_path), args.output, args.format)
    elif args.creds_only:
        export_credentials(str(db_path), args.output, args.format)
    elif args.format == "sql":
        export_sql(str(db_path), args.output)
    elif args.format == "json":
        export_json(str(db_path), args.output, args.table)
    elif args.format == "csv":
        export_csv(str(db_path), args.output, args.table)
    elif args.format == "markdown":
        export_vulnerabilities(str(db_path), args.output, "markdown")
    else:
        print(f"Error: Unsupported format: {args.format}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
