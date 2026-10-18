#!/usr/bin/env python3
"""Generate a Markdown table from allowlisted live reports; never copy model text."""

import argparse
import json
from pathlib import Path

from live_provider_config import CASES, PROVIDERS, REASONS, clean_usage, safe_seconds


def write_summary(directory):
    rows = ['| Provider | Case | Status | Seconds | Total tokens | Reason |',
            '| --- | --- | --- | ---: | ---: | --- |']
    for provider in PROVIDERS:
        path = directory / (provider + '.json')
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for result in report.get('results', []):
            case, status = result.get('case'), result.get('status')
            if case not in CASES or status not in {'passed', 'failed', 'skipped'}:
                continue
            tokens = clean_usage(result.get('usage')).get('total_tokens', '—')
            reason = result.get('reason')
            reason = reason if isinstance(reason, str) and reason in REASONS else '—'
            rows.append(f"| {provider} | {case} | {status} | {safe_seconds(result.get('seconds'))} | {tokens} | {reason} |")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'summary.md').write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('artifacts/live'))
    write_summary(parser.parse_args().directory)
