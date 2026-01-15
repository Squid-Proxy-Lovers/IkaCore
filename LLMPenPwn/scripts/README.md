# Database Dump Scripts

## dump_db.py

Utility script to export the penetration testing SQLite database in various formats.

### Usage

```bash
# Show database summary
python scripts/dump_db.py --db pentest_memory.db --summary

# Export all data as JSON
python scripts/dump_db.py --db pentest_memory.db --format json --output export.json

# Export only vulnerabilities as Markdown report
python scripts/dump_db.py --db pentest_memory.db --vulns-only --format markdown --output vulns.md

# Export only credentials as CSV
python scripts/dump_db.py --db pentest_memory.db --creds-only --format csv --output creds.csv

# Export entire database as SQL dump
python scripts/dump_db.py --db pentest_memory.db --format sql --output backup.sql

# Export all tables as separate CSV files
python scripts/dump_db.py --db pentest_memory.db --format csv --output csv_export/

# Export specific table as JSON
python scripts/dump_db.py --db pentest_memory.db --format json --table vulnerabilities --output vulns.json
```

### Options

- `--db`: Path to SQLite database (default: `pentest_memory.db`)
- `--output`: Output file or directory path (auto-generated if not specified)
- `--format`: Export format - `sql`, `json`, `csv`, or `markdown`
- `--table`: Export specific table only (for json/csv formats)
- `--vulns-only`: Export only vulnerabilities table
- `--creds-only`: Export only interesting_data table
- `--summary`: Show database summary statistics

### Output Formats

- **SQL**: Complete database dump, can be restored with `sqlite3 < db.sql`
- **JSON**: Structured JSON with all tables
- **CSV**: Separate CSV file per table
- **Markdown**: Human-readable vulnerability report (vulns-only)
