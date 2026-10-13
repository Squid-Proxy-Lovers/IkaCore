"""Ensure the configured lint gate actually rejects representative correctness hazards."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(importlib.util.find_spec('ruff') is None, reason='install the dev extra for lint-policy tests')


@pytest.mark.parametrize('rule, code', [
    ('B006', 'def f(values=[]):\n    return values\n'),
    ('B023', 'callbacks = [lambda: index for index in range(3)]\n'),
    ('B904', 'def f():\n    try:\n        int("bad")\n    except ValueError:\n        raise RuntimeError("failed")\n'),
    ('RUF006', 'import asyncio\nasync def f():\n    asyncio.create_task(asyncio.sleep(0))\n'),
    ('ASYNC230', 'async def f():\n    with open("data.txt") as source:\n        return source.read()\n'),
    ('C901', 'def f(value):\n' + ''.join(f'    if value == {i}:\n        return {i}\n' for i in range(16))),
])
def test_lint_policy_detects_correctness_hazards(rule, code):
    result = subprocess.run([sys.executable, '-m', 'ruff', 'check', '--config', str(ROOT / 'pyproject.toml'),
                             '--stdin-filename', 'quality_fixture.py', '--output-format', 'json', '-'],
                            input=code, text=True, capture_output=True, check=False)
    assert result.returncode == 1, result.stderr
    assert rule in {item['code'] for item in json.loads(result.stdout)}
