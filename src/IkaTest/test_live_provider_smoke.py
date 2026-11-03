"""Optional installed-wheel smoke entry point; the normal test suite is offline."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.getenv('IKACORE_LIVE_PROVIDER_TESTS') != '1',
                               reason='set IKACORE_LIVE_PROVIDER_TESTS=1 for installed-wheel live smoke')


def test_live_configured_provider_smoke(monkeypatch):
    root = Path(__file__).parents[2]
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    from live_provider_config import PROVIDERS
    assert any(os.getenv(config['key']) for config in PROVIDERS.values()), 'no live provider configured'
    subprocess.run([sys.executable, str(root / 'scripts/live_matrix.py'), '--all-configured',
                    '--case', 'sync_text', '--case', 'async_text'], cwd=root, check=True, timeout=1100)
