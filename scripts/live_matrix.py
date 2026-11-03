#!/usr/bin/env python3
"""Build once, install in a temporary runtime-only venv, run configured providers."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--providers', help='comma-separated provider IDs, or all-configured')
    parser.add_argument("--retention-eval", action="store_true", help="run bounded Codex compaction fact-retention evaluation")
    options, forwarded = parser.parse_known_args()
    if options.providers:
        from live_provider_config import PROVIDERS
        selected = options.providers.split(',')
        if options.providers == 'all-configured':
            forwarded = ['--all-configured', *forwarded]
        elif all(name in PROVIDERS for name in selected):
            forwarded = [item for name in dict.fromkeys(selected) for item in ('--provider', name)] + forwarded
        else:
            parser.error('invalid provider selection')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory(prefix='ikacore-live-wheel-') as directory:
        temp = Path(directory)
        # Build outside the checkout: setuptools must not leave stale build/ files.
        source = temp / 'source'
        source.mkdir()
        for filename in ('pyproject.toml', 'README.md', 'LICENSE', 'NOTICE'):
            shutil.copy2(ROOT / filename, source / filename)
        shutil.copytree(ROOT / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '.', '--no-deps', '--no-build-isolation',
                        '-w', str(temp / 'dist')], cwd=source, env=env, check=True, stdout=subprocess.DEVNULL, timeout=180)
        wheel, = (temp / 'dist').glob('*.whl')
        runtime = temp / 'runtime'
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(runtime)], env=env, check=True, timeout=60)
        python = runtime / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install', '--retries', '0', '--timeout', '30', str(wheel)],
                       env=env, check=True, stdout=subprocess.DEVNULL, timeout=180)
        args = forwarded if options.retention_eval else forwarded or ["--all-configured"]
        runner = "live_compaction_eval.py" if options.retention_eval else "live_provider_check.py"
        return subprocess.run([str(python), str(ROOT / "scripts" / runner), *args],
                              cwd=ROOT, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
